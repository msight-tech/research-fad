# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

""" CNN cell for architecture search """
import torch
import torch.nn as nn
from .. import ops
import fad_core.genotypes


class SearchCell(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """
    def __init__(self, n_nodes, C_pp, C_p, C, reduction_p, reduction, bottleNeck=1, norm=True, relu=True):
        """
        Args:
            n_nodes: # of intermediate n_nodes
            C_pp: C_out[k-2]
            C_p : C_out[k-1]
            C   : C_in[k] (current)
            reduction_p: flag for whether the previous cell is reduction cell or not
            reduction: flag for whether the current cell is reduction cell or not
        """
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes
        self.bottleNeck = bottleNeck
        self.C_node = int(C*bottleNeck)

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.

        # generate dag
        self.dag = nn.ModuleList()
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            for j in range(2+i): # include 2 input nodes
                # reduction should be used only for input node
                stride = 2 if reduction and j < 2 else 1
                # if normal darts
                op = ops.MixedOp(C, stride, norm=norm, relu=relu)
                if bottleNeck != 1:
                    op_all = nn.ModuleList()
                    op_all.append(nn.Conv2d(self.C_node, C, 1, stride=1, bias=False))
                    op_all.append(op) 
                    op_all.append(nn.Conv2d(C, self.C_node, 1, stride=1, bias=False))    
                    self.dag[i].append(op_all)
                else:  
                    self.dag[i].append(op)


    def forward(self, s0, s1, w_dag):
       
        states = [s0, s1]
 
        for edges, w_list in zip(self.dag, w_dag):
            if self.bottleNeck != 1:
                s_cur = sum(edges[i][2](edges[i][1](edges[i][0](s), w)) for i, (s, w) in enumerate(zip(states, w_list)))
            else:
                s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(states, w_list)))
            states.append(s_cur)

        s_out = torch.cat(states[2:], dim=1)

        return s_out

def pad(x,div):
    w,h = x.shape[2], x.shape[3]
    wp, hp = 0, 0
    if w%div != 0:
        wp = div - w%div
    if h%div != 0:
        hp = div - h%div
    if wp !=0 or hp != 0:
        return nn.functional.pad(x, (0,hp,0,wp), "constant", 0) 
    else:
        return x

 

