#
# File: test_trainer.py
#
import torch
import numpy as np

from structmechmod import models, rigidbody
from structmechmod.trainer import HParams, train

def test_basics():
    potential = models.DelanZeroPotential(2)
    mod = rigidbody.DeLan(2, 64, 3, torch.tensor([1, 1, 0, 0]), udim=1, potential=potential)
    tdata = (np.random.rand(128, 4), np.random.rand(128, 1), np.random.rand(128, 4))
    vdata = (np.random.rand(128, 4), np.random.rand(128, 1), np.random.rand(128, 4))

    train(mod, tdata, vdata, HParams(None, 2, 0.001, 32, 0.01, 50, 50, 50.))
    
