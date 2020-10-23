# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

""" Genotypes
    - Genotype: normal/reduce gene + normal/reduce cell output connection (concat)
    - gene: discrete ops information (w/o output connection)
    - dag: real ops (can be mixed or discrete, but Genotype has only discrete information itself)
"""
from collections import namedtuple
import torch
import torch.nn as nn
from fad_core.modeling import ops as ops
import torch.nn.functional as F

Genotype = namedtuple('Genotype', 'normal normal_concat')

PRIMITIVES = [
    'stack_x5_stdConv+stdConv_dilSep+stdConv_dilSep',
    'stack_x5_sinSepConv+sinSepConv_dilSep+sinSepConv_dilSep',
    'none'
]
# Note: different from the paper, we exclude the convolution with dilation rate of 3, since we empirically find that it harms the detection performance.


PRIMSTACK = [
    'std_conv_3x3',
    'std_conv_3x3+dil_conv_3x3',
    'std_conv_3x3_x2',
    'std_conv_3x3_x2+dil_conv_3x3',
    'std_conv_3x3_x3',
    'sinSep_conv_3x3',
    'sinSep_conv_3x3+dil_conv_3x3',
    'sinSep_conv_3x3_x2',
    'sinSep_conv_3x3_x2+dil_conv_3x3',
    'sinSep_conv_3x3_x3',
    'none'
]


def to_dag(C_in, gene, reduction, bottleNeck=1, norm=True, relu=True):
    """ generate discrete ops from gene """
    dag = nn.ModuleList()
    for edges in gene:
        row = nn.ModuleList()
        for op_name, s_idx in edges:
            # reduction cell & from input nodes => stride = 2
            stride = 2 if reduction and s_idx < 2 else 1
            op = ops.OPS[op_name](C_in, stride, True, norm, relu)
            if bottleNeck != 1: 
                op = nn.Sequential(
                    nn.Conv2d(int(C_in*bottleNeck), C_in, 1, stride=1),
                    op,
                    nn.Conv2d(C_in, int(C_in*bottleNeck), 1, stride=1)
                )
            op.s_idx = s_idx
            row.append(op)
        dag.append(row)

    return dag


def from_str(s):
    """ generate genotype from string
    e.g. "Genotype(
            normal=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
                    [('sep_conv_3x3', 1), ('dil_conv_3x3', 2)],
                    [('sep_conv_3x3', 1), ('sep_conv_3x3', 2)],
                    [('sep_conv_3x3', 1), ('dil_conv_3x3', 4)]],
            normal_concat=range(2, 6),
            reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)]],
            reduce_concat=range(2, 6))"
    """

    genotype = eval(s)

    return genotype


def parse(alpha, k, stackConv=False):
    """
    parse continuous alpha to discrete gene.
    alpha is ParameterList:
    ParameterList [
        Parameter(n_edges1, n_ops),
        Parameter(n_edges2, n_ops),
        ...
    ]

    gene is list:
    [
        [('node1_ops_1', node_idx), ..., ('node1_ops_k', node_idx)],
        [('node2_ops_1', node_idx), ..., ('node2_ops_k', node_idx)],
        ...
    ]
    each node has two edges (k=2) in CNN.
    """

    gene = []
    assert 'none' in PRIMITIVES[-1] # assume last PRIMITIVE is 'none'

    # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
    # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
    for i, edges in enumerate(alpha):
        # edges: Tensor(n_edges, n_ops)
        edge_max, primitive_indices = torch.topk(edges[:, :-1], 1) # ignore 'none'
        
        topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
        node_gene = []
        for edge_idx in topk_edge_indices:
            prim_idx = primitive_indices[edge_idx]
            if not stackConv:
                prim = PRIMITIVES[prim_idx]
            else:
                prim = PRIMSTACK[prim_idx]
            node_gene.append((prim, edge_idx.item()))

        gene.append(node_gene)

    return gene



def parseSingle(alpha):
    """
    parse continuous alpha to discrete gene.
    alpha is ParameterList:
    ParameterList [
        Parameter(n_edges1, n_ops),
        Parameter(n_edges2, n_ops),
        ...
    ]

    gene is list:
    [
        [('node1_ops_1', node_idx), ..., ('node1_ops_k', node_idx)],
        [('node2_ops_1', node_idx), ..., ('node2_ops_k', node_idx)],
        ...
    ]
    each node has two edges (k=2) in CNN.
    """

    gene = []
    assert 'none' in PRIMITIVES[-1] # assume last PRIMITIVE is 'none'

    # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
    # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
    edges = alpha
    edge_max, prim_idx = torch.topk(edges[:-1], 1) # ignore 'none'
        
    prim = PRIMITIVES[prim_idx] 
    gene.append((prim, 0))

    return gene


