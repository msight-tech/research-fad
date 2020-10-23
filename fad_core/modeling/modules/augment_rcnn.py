# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

""" CNN for network augmentation """
import torch
import torch.nn as nn
from .augment_cells import AugmentCell
from fad_core.modeling import ops as ops
import fad_core.genotypes as gt


class AugmentRCNN(nn.Module):
    """ Augmented CNN model """
    def __init__(self, C_in, C, n_layers, genotype, norm=True, C_node=None):
        """
        Args:
            input_size: size of height and width (assuming height = width)
            C_in: # of input channels
            C: # of starting model channels
        """
        super().__init__()
               
        bottleNeck = C_in / C if not C_node else C_node / C

        self.C_in = C_in
        self.C = C

        self.n_layers = n_layers
        self.genotype = gt.from_str(genotype)
        genotype = self.genotype

        C_cur = C_in
        
        C_pp, C_p, C_cur = C_cur, C_cur, C
        
        self.cells = nn.ModuleList()
        reduction_p = False
        reduction = reduction_p

        for i in range(n_layers):

            cell = AugmentCell(genotype, C_pp, C_p, C_cur, reduction_p, reduction, bottleNeck, norm=norm, relu=True) 

            self.cells.append(cell)
            C_cur_out = int(C_cur * len(cell.concat) * bottleNeck)
            C_pp, C_p = C_p, C_cur_out


    def forward(self, x):
        if isinstance(x ,list):
            s0, s1 = x[0], x[1]
        else:
            s0 = s1 = x
 
        if isinstance(s0,list): s0 = s0[0]
        if isinstance(s1,list): s1 = s1[0]

        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
 
        return s1 
 

    def drop_path_prob(self, p):
        """ Set drop path probability """
        for module in self.modules():
            if isinstance(module, ops.DropPath_):
                module.p = p
