# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import torch

from fcos_core.engine.trainer import reduce_loss_dict

class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, w_momentum, w_weight_decay, first_order=False):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay


    def virtual_step(self, trn_X, trn_y, xi, w_optim, features=None):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        if features:
            loss_dict = self.net(trn_X, features, trn_y) # L_trn(w)
        else:
            loss_dict = self.net(trn_X, trn_y) # L_trn(w)
        loss_dict = reduce_loss_dict(loss_dict)

        # ----- combine the losses into one by summing
        loss = sum(loss for loss in loss_dict.values())
        
        # compute gradient
        gradients = torch.autograd.grad(loss.mean(), filter(lambda p: p.requires_grad, self.net.weights()), allow_unused=True)

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(filter(lambda p: p.requires_grad, self.net.weights()), filter(lambda p: p.requires_grad, self.v_net.weights()), gradients):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                if g is None: 
                    #print('g is None') 
                    continue
                vw.copy_(w - xi * (m + g + self.w_weight_decay*w))

            # synchronize alphas
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)

    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim, features=None, features_val=None):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        
        # do virtual step (calc w`)
        if features:
            self.virtual_step(trn_X, trn_y, xi, w_optim, features)
        else: 
            self.virtual_step(trn_X, trn_y, xi, w_optim)

        # calc unrolled loss
        if features:
            loss_dict = self.v_net(val_X, features_val, val_y) # L_val(w`)
        else:
            loss_dict = self.v_net(val_X, val_y) # L_val(w`)
        loss_dict = reduce_loss_dict(loss_dict)

        # ----- combine the losses into one by summing
        loss = sum(loss for loss in loss_dict.values())


        # compute gradient
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(filter(lambda p: p.requires_grad, self.v_net.weights()))

        v_grads = torch.autograd.grad(loss.mean(), v_alphas + v_weights,allow_unused=True)

        dalpha = v_grads[:len(v_alphas)]

 
        # -------------- 2nd-order, better accuracy
        dw = v_grads[len(v_alphas):]

        if features:
                hessian = self.compute_hessian(dw, trn_X, trn_y, features)
        else:
                hessian = self.compute_hessian(dw, trn_X, trn_y)

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                alpha.grad = da - xi*h


    # -------------- 1st-order approximation, much faster
    def first_order_backward(self, val_X, val_y):

        loss_dict = self.v_net(val_X, val_y) # L_val(w`)
        loss_dict = reduce_loss_dict(loss_dict)

        # ----- combine the losses into one by summing
        loss = sum(loss for loss in loss_dict.values())

        # compute gradient
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(filter(lambda p: p.requires_grad, self.v_net.weights()))

        v_grads = torch.autograd.grad(loss.mean(), v_alphas + v_weights,allow_unused=True)

        dalpha = v_grads[:len(v_alphas)]

        with torch.no_grad():
           for alpha, da in zip(self.net.alphas(), dalpha):
               alpha.grad = da

    def compute_hessian(self, dw, trn_X, trn_y, features=None):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """

        norm = torch.cat([w.view(-1) for w in dw if w is not None]).norm()

        eps = 0.01 / norm

        with torch.no_grad():
            for p, d in zip(filter(lambda p: p.requires_grad, self.net.weights()), dw):
                if d is not None:
                    p += eps * d
        if features:
            loss_dict = self.net(trn_X, features, trn_y)
        else:
            loss_dict = self.net(trn_X, trn_y)
        loss_dict = reduce_loss_dict(loss_dict)

        # ----- combine the losses into one by summing
        loss = sum(loss for loss in loss_dict.values())

        dalpha_pos = torch.autograd.grad(loss.mean(), self.net.alphas(),allow_unused=True) # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(filter(lambda p: p.requires_grad, self.net.weights()), dw):
                if d is not None:
                    p -= 2. * eps * d
        if features:
            loss_dict = self.net(trn_X, features, trn_y)
        else:
            loss_dict = self.net(trn_X, trn_y)
        loss_dict = reduce_loss_dict(loss_dict)

        # ----- combine the losses into one by summing
        loss = sum(loss for loss in loss_dict.values())

        dalpha_neg = torch.autograd.grad(loss.mean(), self.net.alphas(), allow_unused=True) # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(filter(lambda p: p.requires_grad, self.net.weights()), dw):
                if d is not None:
                    p += eps * d

        hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian
