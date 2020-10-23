# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

""" CNN for architecture search """
import torch
import torch.nn as nn
import torch.nn.functional as F
from .search_cells import SearchCell
import fad_core.genotypes as gt
from torch.nn.parallel._functions import Broadcast
import logging
import pdb
import random
import math

from fcos_core.config import cfg as cfg_det
from fcos_core.modeling.backbone import build_backbone
from fcos_core.modeling.detector.detectors import build_detection_model


def broadcast_list(l, device_ids):
    """ Broadcasting list """

    l_copies = Broadcast.apply(device_ids, *l)

    if len(l) == 0:   
        return l_copies

    l_copies = [l_copies[i:i+len(l)] for i in range(0, len(l_copies), len(l))]
        
    return l_copies

class SearchRCNN(nn.Module):
    """ Search detector model """
    def __init__(self, C_in, C, n_layers, n_nodes=4, norm=True, C_node=None):
        """
        Args:
            C_in: # of input channels
            C: # of starting model channels
            n_classes: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell
            stem_multiplier
        """
        super().__init__()

        self.C_in = C_in
        self.C = C
        self.n_layers = n_layers
        bottleNeck = C_in / C if not C_node else C_node / C

        # ------------ change stem to backbone CNN
        C_cur = C_in # output channel size of backbone

        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.
        C_pp, C_p, C_cur = C_cur, C_cur, C

        self.cells = nn.ModuleList()

        for i in range(n_layers):
            cell = SearchCell(n_nodes, C_pp, C_p, C_cur, False, False, bottleNeck=bottleNeck, norm=norm, relu=True)

            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out

    def forward(self, x, weights_normal):
        if isinstance(x, list):
            s0, s1 = x[0], x[1]
        else:
            s0 = s1 = x 

        for cell in self.cells:
            weights = weights_normal
            s0, s1 = s1, cell(s0, s1, weights)

        # output the feature map directly
        return s1


class SearchRCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """
    def __init__(self, n_layers, n_nodes=4, stem_multiplier=3,
                 device_ids=None, cfg_det='', n_module=1):
        super().__init__()
        self.n_nodes = n_nodes
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids
        self.cfg_det = cfg_det
        self.C = cfg_det.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        self.n_module = n_module
        self.stackConv = True

        # initialize architect parameters: alphas
        if self.stackConv:
            n_ops = 1 + sum([int(x[7]) for x in gt.PRIMITIVES if x[:5]=='stack' ])
        else:
            n_ops = len(gt.PRIMITIVES)
        
        self.alpha_normal = nn.ParameterList()

        for n_m in range(n_module):
            for i in range(n_nodes):
                self.alpha_normal.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))
            if 'beta' in n: 
                self._alphas.append((n, p))

        # build detector with NAS 
        self.net = build_detection_model(cfg_det) 

    def forward(self, x, targets=None):

        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
             
        if len(self.device_ids) == 1:
                return self.net(x, targets, weights_normal)

        x_tensors = x.tensors
        if len(x_tensors) < len(self.device_ids):
            # rare situation: number of images in batch < number of devices
            factor = math.ceil(len(self.device_ids) / len(x_tensors))
            x_tensors = torch.cat([x_tensors] * factor, dim=0)[:len(self.device_ids)]
            targets = (targets * factor)[:len(self.device_ids)]
        
        xs = nn.parallel.scatter(x_tensors, self.device_ids)
        cnts = list(map(lambda x: len(x), xs))
        for i in range(1, len(cnts)):
            cnts[i] += cnts[i - 1]
        cnts = [0] + cnts
        # scatter targets
        ts = [targets[cnts[i]:cnts[i + 1]] for i in range(len(cnts) - 1)]
        # map targets to corresponding gpu
        ts = list(map(lambda td: [_t.to(f"cuda:{td[1]}") for _t in td[0]], zip(ts, self.device_ids)))

        # broadcast weights
        wnormal_copies = broadcast_list(weights_normal, self.device_ids)
        
        # replicate modules
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas, list(zip(xs, ts, wnormal_copies)), devices=self.device_ids)
         
        return nn.parallel.gather(outputs, self.device_ids[0])

    def loss(self, X, y):
            return self.forward(X,y)

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha in self.alpha_normal:
            logger.info(F.softmax(alpha, dim=-1))

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        geno = []

        gene_normal = gt.parse(self.alpha_normal[:self.n_nodes], k=2, stackConv=self.stackConv)
        concat = range(2, 2+self.n_nodes) # concat all intermediate nodes
        geno.append(gt.Genotype(normal=gene_normal, normal_concat=concat))

        # if search for both tower
        if self.n_module == 2:
            gene_normal = gt.parse(self.alpha_normal[self.n_nodes:], k=2, stackConv=self.stackConv)
            concat = range(2, 2+self.n_nodes) # concat all intermediate nodes
            geno.append(gt.Genotype(normal=gene_normal, normal_concat=concat))

        return geno
 
    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p



