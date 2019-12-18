#!/usr/bin/env python3
#
# File: test_smm.py
#
import torch
from structmechmod import rigidbody, utils, models
import numpy as np

torch.set_default_dtype(torch.float64)

def test_delan_smm1():
    utils.set_rng_seed(42)
    mask = torch.tensor([1, 1, 0, 0])
    delan = rigidbody.DeLan(2, 32, 3, mask, activation='Tanh', udim=1, bias=10.0)
    mass_matrix = None
    smm = rigidbody.LearnedRigidBody(2, 1, mask, mass_matrix=mass_matrix, hidden_sizes=[32, 32, 32])
    l1 = delan._mass_matrix_network.layers
    l2 = smm._mass_matrix.embed._lnet._mlp

    l2[0].weight.data.copy_(l1[0].weight.data)
    l2[2].weight.data.copy_(l1[1].weight.data)
    l2[4].weight.data.copy_(l1[2].weight.data)
    smm._mass_matrix.out.weight.data.copy_(l1[3].weight.data)

    l2[0].bias.data.copy_(l1[0].bias.data)
    l2[2].bias.data.copy_(l1[1].bias.data)
    l2[4].bias.data.copy_(l1[2].bias.data)
    smm._mass_matrix.out.bias.data.copy_(l1[3].bias.data)

    for p1, p2 in zip(delan._mass_matrix_network.parameters(), smm._mass_matrix.parameters()):
        assert (p1 == p2).all()



def test_delan_smm2():
    utils.set_rng_seed(42)
    mask = torch.tensor([1, 1, 0, 0])
    delan = rigidbody.DeLan(2, 32, 3, mask, activation='Tanh', udim=1, bias=10.0)
    mass_matrix = models.DelanCholeskyMMNet(2, hidden_sizes=[32, 32, 32])
    smm = rigidbody.LearnedRigidBody(2, 1, mask, mass_matrix=mass_matrix, hidden_sizes=[32, 32, 32])
    l1 = delan._mass_matrix_network.layers
    l2 = smm._mass_matrix.embed
    l2[0].weight.data.copy_(l1[0].weight.data)
    l2[1].weight.data.copy_(l1[1].weight.data)
    l2[2].weight.data.copy_(l1[2].weight.data)
    smm._mass_matrix.out.weight.data.copy_(l1[3].weight.data)

    l2[0].bias.data.copy_(l1[0].bias.data)
    l2[1].bias.data.copy_(l1[1].bias.data)
    l2[2].bias.data.copy_(l1[2].bias.data)
    smm._mass_matrix.out.bias.data.copy_(l1[3].bias.data)

    for p1, p2 in zip(delan._mass_matrix_network.parameters(), smm._mass_matrix.parameters()):
        assert (p1 == p2).all()

    q = torch.rand(1, 2).requires_grad_()
    v = torch.rand(1, 2).requires_grad_()
    cv_d = delan.corriolisforce(q, v)
    cv_s = smm.corriolisforce(q, v)
    assert np.allclose(cv_d.detach(), cv_s.detach())
    M_d = delan.mass_matrix(q)
    M_s = smm.mass_matrix(q)

if __name__ == '__main__':
    test_delan_smm2()
