# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

""" Operations """
import torch
import torch.nn as nn
import fad_core.genotypes as gt
import pdb

OPS = {
    'none': lambda C, stride, affine, norm, relu: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: PoolBN('avg', C, 3, stride, 1, affine=affine),
    'max_pool_3x3': lambda C, stride, affine: PoolBN('max', C, 3, stride, 1, affine=affine),
    'skip_max_pool': lambda C, stride, affine: \
        Identity() if stride == 1 else PoolBN('max', C, 3+2*(stride==4), stride, 1 if stride==2 else (2,2), affine=affine), #
    'skip_connect': lambda C, stride, affine: \
        Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine, stride=stride),
    'sep_conv_3x3': lambda C, stride, affine, norm, relu: SepConv(C, C, 3, stride, 1, affine=affine, norm=norm, relu=relu),
    'sep_conv_5x5': lambda C, stride, affine, norm, relu: SepConv(C, C, 5, stride, 2, affine=affine, norm=norm, relu=relu),
    'sep_conv_7x7': lambda C, stride, affine, norm, relu: SepConv(C, C, 7, stride, 3, affine=affine, norm=norm, relu=relu),
    'dil_conv_3x3': lambda C, stride, affine, norm, relu: DilConv(C, C, 3, stride, 2, 2, affine=affine, norm=norm, relu=relu), # 5x5
    'dil_conv_5x5': lambda C, stride, affine, norm, relu: DilConv(C, C, 5, stride, 4, 2, affine=affine, norm=norm, relu=relu), # 9x9
    'conv_7x1_1x7': lambda C, stride, affine: FacConv(C, C, 7, stride, 3, affine=affine),
    'std_conv_3x3': lambda C, stride, affine, norm, relu: StdConv(C, C, 3, stride, 1, 1, affine=affine, norm=norm, relu=relu),
    'std_conv_5x5': lambda C, stride, affine, norm, relu: StdConv(C, C, 5, stride, 2, 1, affine=affine, norm=norm, relu=relu),
    'dil_std_conv_3x3': lambda C, stride, affine, norm, relu: StdConv(C, C, 3, stride, 2, 2, affine=affine, norm=norm, relu=relu),
    'dil_std_conv_5x5': lambda C, stride, affine, norm, relu: StdConv(C, C, 5, stride, 4, 2, affine=affine, norm=norm, relu=relu),
    'sinSep_conv_3x3': lambda C, stride, affine, norm, relu: SingleSepConv(C, C, 3, stride, 1, affine=affine, norm=norm, relu=relu),

    'stack_x5_stdConv+stdConv_dilSep+stdConv_dilSep': lambda C, stride, affine, norm, relu: RepShare([(0,['stdConv']), (1,['stdConv','dilSep']), (2,['stdConv','dilSep'])], C, C, 3, stride, 1, 1, affine=affine, norm=norm, relu=relu),
    'stack_x5_sinSepConv+sinSepConv_dilSep+sinSepConv_dilSep': lambda C, stride, affine, norm, relu: RepShare([(0,['sinSepConv']), (1,['sinSepConv','dilSep']), (2,['sinSepConv','dilSep'])], C, C, 3, stride, 1, 1, affine=affine, norm=norm, relu=relu), 
    'stack_x6_stdConv+stdConv_dilSep_dilSep3+stdConv_dilSep': lambda C, stride, affine, norm, relu: RepShare([(0,['stdConv']), (1,['stdConv','dilSep','dilSep3']), (2,['stdConv','dilSep'])], C, C, 3, stride, 1, 1, affine=affine, norm=norm, relu=relu),
    'stack_x6_sinSepConv+sinSepConv_dilSep_dilSep3+sinSepConv_dilSep': lambda C, stride, affine, norm, relu: RepShare([(0,['sinSepConv']), (1,['sinSepConv','dilSep','dilSep3']), (2,['sinSepConv','dilSep'])], C, C, 3, stride, 1, 1, affine=affine, norm=norm, relu=relu), 

    'std_conv_3x3_x2': lambda C, stride, affine, norm, relu: StackConv([('stdConv', 2)], C, C, 3, stride, 1, 1, affine=affine, norm=norm, relu=relu),
    'std_conv_3x3_x3': lambda C, stride, affine, norm, relu: StackConv([('stdConv', 3)], C, C, 3, stride, 1, 1, affine=affine, norm=norm, relu=relu),
    'std_conv_3x3+dil_conv_3x3': lambda C, stride, affine, norm, relu: StackConv([('stdConv',1), ('dilSep',1)], C, C, 3, stride, 1, 1, affine=affine, norm=norm, relu=relu),
    'std_conv_3x3+dil_conv_3x3_r3': lambda C, stride, affine, norm, relu: StackConv([('stdConv',1), ('dilSep3',1)], C, C, 3, stride, 1, 1, affine=affine, norm=norm, relu=relu),
    'std_conv_3x3_x2+dil_conv_3x3': lambda C, stride, affine, norm, relu: StackConv([('stdConv',2), ('dilSep',1)], C, C, 3, stride, 1, 1, affine=affine, norm=norm, relu=relu),
    'sep_conv_3x3_x2': lambda C, stride, affine, norm, relu: StackConv([('sepConv', 2)], C, C, 3, stride, 1, 1, affine=affine, norm=norm, relu=relu),
    'sep_conv_3x3_x3': lambda C, stride, affine, norm, relu: StackConv([('sepConv', 3)], C, C, 3, stride, 1, 1, affine=affine, norm=norm, relu=relu),
    'sep_conv_3x3+dil_conv_3x3': lambda C, stride, affine, norm, relu: StackConv([('sepConv',1), ('dilSep',1)], C, C, 3, stride, 1, 1, affine=affine, norm=norm, relu=relu),
    'sep_conv_3x3+dil_conv_3x3_r3': lambda C, stride, affine, norm, relu: StackConv([('sepConv',1), ('dilSep3',1)], C, C, 3, stride, 1, 1, affine=affine, norm=norm, relu=relu),
    'sep_conv_3x3_x2+dil_conv_3x3': lambda C, stride, affine, norm, relu: StackConv([('sepConv',2), ('dilSep',1)], C, C, 3, stride, 1, 1, affine=affine, norm=norm, relu=relu),
    'sinSep_conv_3x3_x2': lambda C, stride, affine, norm, relu: StackConv([('sinSepConv', 2)], C, C, 3, stride, 1, 1, affine=affine, norm=norm, relu=relu),
    'sinSep_conv_3x3_x3': lambda C, stride, affine, norm, relu: StackConv([('sinSepConv', 3)], C, C, 3, stride, 1, 1, affine=affine, norm=norm, relu=relu),
    'sinSep_conv_3x3+dil_conv_3x3': lambda C, stride, affine, norm, relu: StackConv([('sinSepConv',1), ('dilSep',1)], C, C, 3, stride, 1, 1, affine=affine, norm=norm, relu=relu),
    'sinSep_conv_3x3+dil_conv_3x3_r3': lambda C, stride, affine, norm, relu: StackConv([('sinSepConv',1), ('dilSep3',1)], C, C, 3, stride, 1, 1, affine=affine, norm=norm, relu=relu),
    'sinSep_conv_3x3_x2+dil_conv_3x3': lambda C, stride, affine, norm, relu: StackConv([('sinSepConv',2), ('dilSep',1)], C, C, 3, stride, 1, 1, affine=affine, norm=norm, relu=relu)
}


def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # per data point mask; assuming x in cuda.
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)

    return x


class DropPath_(nn.Module):
    def __init__(self, p=0.):
        """ [!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        drop_path_(x, self.p, self.training)

        return x


class PoolBN(nn.Module):
    """
    AvgPool or MaxPool - BN
    """
    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        """
        Args:
            pool_type: 'max' or 'avg'
        """
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

        self.bn = nn.GroupNorm(32, C)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out


class StdConv(nn.Module):
    """ Standard conv
    Conv - GN - ReLU
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation=1, affine=True, norm=True, relu=True):
        super().__init__()
        modules = []
        modules.append(nn.Conv2d(C_in, C_out, kernel_size, stride, padding, dilation=dilation))
        if norm:
            modules.append(nn.GroupNorm(32, C_out))
        if relu:
            modules.append(nn.ReLU())
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class SingleSepConv(nn.Module):
    """ Single sep conv
    Conv - GN - ReLU
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, norm=True, relu=True):
        super().__init__()
        modules = []
        modules.append(nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False))  
        modules.append(nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=True))
        if norm:
            modules.append(nn.GroupNorm(32, C_out))
        if relu:
            modules.append(nn.ReLU())
        self.net = nn.Sequential(*modules)
            
    def forward(self, x):
        return self.net(x)


class FacConv(nn.Module):
    """ Factorized conv
    ReLU - Conv(Kx1) - Conv(1xK) - BN
    """
    def __init__(self, C_in, C_out, kernel_length, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, (kernel_length, 1), stride, padding, bias=False),
            nn.Conv2d(C_in, C_out, (1, kernel_length), stride, padding),
            nn.GroupNorm(32, C_out)
        )

    def forward(self, x):
        return self.net(x)


class DilConv(nn.Module):
    """ (Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True, norm=True, relu=True):
        super().__init__()
        modules = []
        modules.append(nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in, bias=False))
        modules.append(nn.Conv2d(C_in, C_out, 1, stride=1, padding=0))
        if norm:
            modules.append(nn.GroupNorm(32, C_out))
        if relu:
            modules.append(nn.ReLU())
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class SepConv(nn.Module):
    """ Depthwise separable conv
    DilConv(dilation=1) * 2
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, norm=True, relu=True):
        super().__init__()
        self.net = nn.Sequential(
                DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1, affine=affine, norm=norm, relu=relu),
                DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine, norm=norm, relu=relu)
        ) 

    def forward(self, x):
        return self.net(x)


class StackConv(nn.Module):
    """ Stacked conv
    Conv - GN - ReLU
    """
    def __init__(self, ops, C_in, C_out, kernel_size, stride, padding, dilation=1, affine=True, norm=True, relu=True):
        # 'affine' is not used, here bias is always on
        super().__init__()
        cnt = 0    
        self.norm, self.relu = norm, relu
 
        for op, rep in ops:
            for n in range(cnt, rep+cnt):
                if op == 'stdConv':
                    conv = nn.Conv2d(C_in, C_out, kernel_size, stride, padding, dilation=dilation)
 
                elif op == 'dilSep':
                    conv = nn.Sequential(
                        nn.Conv2d(C_in, C_in, kernel_size, stride, 2, dilation=2, groups=C_in),
                        nn.Conv2d(C_in, C_out, 1, stride=1, padding=0)
                    )
                elif op == 'dilSep3':
                    conv = nn.Sequential(
                        nn.Conv2d(C_in, C_in, kernel_size, stride, 3, dilation=3, groups=C_in),
                        nn.Conv2d(C_in, C_out, 1, stride=1, padding=0)
                    )
                elif op == 'sepConv':
                        conv = nn.Sequential(
                            DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1, affine=affine, norm=True, relu=True),
                            DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine, norm=False, relu=False)
                        )
                elif op == 'sinSepConv':
                    conv = SingleSepConv(C_in, C_out, kernel_size, stride, padding, affine=True, norm=False, relu=False) 
                else:
                    pdb.set_trace()

                self.add_module('conv_%d' % n, conv) 
                del conv

                if norm:
                    self.add_module('gn_%d' % n, nn.GroupNorm(32, C_out) )
                if relu:
                    self.add_module('relu_%d' % n, nn.ReLU() )

            cnt = n + 1     
        self.rep = cnt

    def forward(self, x):        
        for n in range(self.rep):
            x = getattr(self, 'conv_%d' % n)(x)    
            if self.norm:
                x = getattr(self, 'gn_%d' % n)(x)
            if self.relu:
                x = getattr(self, 'relu_%d' % n)(x)   
        return x


class RepShare(nn.Module):
    """ 
    Representation Sharing with StackConv
    """
    def __init__(self, ops, C_in, C_out, kernel_size, stride, padding, dilation=1, affine=True, norm=True, relu=True):
        super().__init__()

        self.decouple = True

        self.ops = ops 
        self.num_rep = len(self.ops)

        for rep, opSet in ops:
            for id, op in enumerate(opSet):
                self.add_module('layer_%d_%d' % (rep, id),
                    StackConv([(op, 1)], C_in, C_out, 3, stride, 2 if op=='dilSep' else 3 if op=='dilSep3' else 1, 2 if op=='dilSep' else 3 if op=='dilSep3' else 1, affine=affine, norm=norm, relu=relu) 
                )           
                # if add conv1x1 to decouple 
                if self.decouple and (rep != self.num_rep-1) and (id == 0): 
                    self.add_module('decouple_%d_%d' % (rep, id), nn.Conv2d(C_in, C_out, 1, 1, 0, bias=False)
                    )
 
    def forward(self, x):       
        res = []
        for rep, opSet in self.ops:
            for id in range(len(opSet))[::-1]: # forward pass in reverse order
                if id == 0: # update x if it is the root operation
                    x = getattr(self, 'layer_%d_%d' % (rep, id))(x) 
                    if self.decouple and (rep != self.num_rep-1): 
                        # add a conv1x1 before append 
                        res.append(getattr(self, 'decouple_%d_%d' % (rep, id))(x))
                    else:
                        res.append(x)
                else:
                    res.append(getattr(self, 'layer_%d_%d' % (rep, id))(x)) 

        return res


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class IdentityGN(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.conv = nn.Conv2d(C, C, 1, 1, bias=False)
        self.gn = nn.GroupNorm(32, C)

    def forward(self, x):
        return self.gn(self.conv(x))
      

class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.

        # re-sizing by stride
        return x[:, :, ::self.stride, ::self.stride] * 0.


# channel can vary
class ZeroC(nn.Module):
    def __init__(self, C_in, C_out, stride, affine=True, norm=True, relu=True):
        super().__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.conv = nn.Conv2d(C_in, C_out, 1, stride)
        torch.nn.init.constant_(self.conv.weight, 0)

    def forward(self, x):
        x = self.conv(x)
        return x * 0.


class Interpolate(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.factor, mode="nearest")


class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """
    def __init__(self, C_in, C_out, affine=True, stride=2):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1r = nn.Conv2d(C_in, C_out // 2, 1, stride=stride, padding=0)
        self.conv2r = nn.Conv2d(C_in, C_out // 2, 1, stride=stride, padding=0)
        self.bn = nn.GroupNorm(32, C_out)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1r(x), self.conv2r(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class MixedOp(nn.Module):
    """ Mixed operation """
    def __init__(self, C, stride, C_out=0, upsample=0, norm=True, relu=True):
        super().__init__()
        self._ops = nn.ModuleList()
        self.upsample = upsample            
        self.stackConv = False

        for primitive in gt.PRIMITIVES:
            if 'stack' in primitive:
                self.stackConv = True

            if C_out == 0:
                op = OPS[primitive](C, stride, affine=False, norm=norm, relu=relu)
            else:
                op = OPS[primitive](C, C_out, stride, affine=False, norm=norm, relu=relu)
           
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        if self.upsample > 0:
            x = nn.functional.interpolate(x, scale_factor=self.upsample, mode="nearest")

        if not self.stackConv:
            return sum(w * op(x) for w, op in zip(weights, self._ops))
        else:
            cnt, res = 0, None
            for op in self._ops:
                tmp = op(x)
                # one op may need more than 1 alpha
                if isinstance(tmp, list):
                    for i in range(len(tmp)):
                        res = weights[cnt] * tmp[i] if res is None else res + weights[cnt] * tmp[i]
                        cnt += 1 
                else:
                    res = weights[cnt] * tmp if res is None else res + weights[cnt] * tmp                    
                    cnt += 1

        return res


