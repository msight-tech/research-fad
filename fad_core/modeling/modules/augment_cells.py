# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

""" CNN cell for network augmentation """
import torch
import torch.nn as nn
from fad_core.modeling import ops as ops
from .search_cells import pad
import fad_core.genotypes as gt


class AugmentCell(nn.Module):
    """ Cell for augmentation
    Each edge is discrete.
    """
    def __init__(self, genotype, C_pp, C_p, C, reduction_p, reduction, bottleNeck, norm=True, relu=True):
        super().__init__()
        self.reduction = reduction
        self.n_nodes = len(genotype.normal)
        self.bottleNeck = bottleNeck
        self.C_node = int(C*self.bottleNeck)
 
        if C_pp != self.C_node:
            self.preproc0 = ops.StdConv(C_pp, self.C_node, 1, 1, 0, norm=norm, relu=True)
        if C_p != self.C_node:
            self.preproc1 = ops.StdConv(C_p, self.C_node, 1, 1, 0, norm=norm, relu=True)

        # generate dag
        if reduction:
            gene = genotype.reduce
            self.concat = genotype.reduce_concat
        else:
            gene = genotype.normal
            self.concat = genotype.normal_concat

        self.dag = gt.to_dag(C, gene, reduction, self.bottleNeck, norm=norm, relu=relu)

    def forward(self, s0, s1):
        if s0.shape[1] != self.C_node:       
            s0 = self.preproc0(s0) 
        if s1.shape[1] != self.C_node: 
            s1 = self.preproc1(s1)

        states = [s0, s1]

        for edges in self.dag:
            s_cur = sum(op(states[op.s_idx]) for op in edges)
            states.append(s_cur)

        s_out = torch.cat([states[i] for i in self.concat], dim=1)

        return s_out

