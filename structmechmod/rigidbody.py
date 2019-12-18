# File: rigidbody.py
from typing import Iterable

import abc
import torch
torch.set_default_dtype(torch.float64)

from structmechmod import nn_utils, utils
from structmechmod.models import CholeskyMMNet, PotentialNet, GeneralizedForceNet
from structmechmod.dnn import DifferentialNetwork


class AbstractRigidBody:

    @property
    @abc.abstractmethod
    def thetamask(self):
        """Returns theta mask for configuration q.
        These should use utils.diffangles to compute differences
        """

    @abc.abstractmethod
    def mass_matrix(self, q):
        """Return mass matrix for configuration q"""

    @abc.abstractmethod
    def potential(self, q):
        """Return potential for configuration q"""

    @abc.abstractmethod
    def generalized_force(self, q, v, u):
        """Return generalized force for configuration q, velocity v, external torque u"""

    def kinetic_energy(self, q, v):
        mass_matrix = self.mass_matrix(q)
        # TODO(jkg): Check if this works correctly for batched
        kenergy = 0.5 * (v.unsqueeze(1) @ (mass_matrix @ v.unsqueeze(2))).squeeze(2)
        return kenergy

    def lagrangian(self, q, v):
        """ Returns the Lagrangian of a mechanical system
        """
        kenergy = self.kinetic_energy(q, v)
        pot = self.potential(q)
        lag = kenergy - pot
        return lag

    def hamiltonian(self, q, v):
        """ Returns the Hamiltonian of a mechanical system
        """
        kenergy = self.kinetic_energy(q, v)
        pot = self.potential(q)
        ham = kenergy + pot
        return ham

    def corriolisforce(self, q, v, mass_matrix=None):
        """ Computes the corriolis matrix times v
        """
        with torch.enable_grad():
            if mass_matrix is None:
                mass_matrix = self.mass_matrix(q)

            # if self._slow_corriolis_force:
            #     C = self.corriolis(q,v,mass_matrix=mass_matrix)
            #     return C @ v.unsqueeze(2)

            Mv = mass_matrix @ v.unsqueeze(2)

            KE = 0.5 * v.unsqueeze(1) @ Mv

            Cv_KE = torch.autograd.grad(KE.sum(), q, retain_graph=True, create_graph=True)[0]

            gMv = torch.stack([
                torch.autograd.grad(Mv[:, i].sum(), q, retain_graph=True, create_graph=True)[0]
                for i in range(q.size(1))
            ], dim=1)

            Cv = gMv @ v.unsqueeze(2) - Cv_KE.unsqueeze(2)

            return Cv

    def slowest_Cv(self, q, v, mass_matrix=None):
        """ Computes the corriolis matrix times v
        """
        with torch.enable_grad():
            if mass_matrix is None:
                mass_matrix = self.mass_matrix(q)

            qdim = q.size(1)

            Cv = []

            grad = torch.autograd.grad

            for i in range(qdim):
                res = 0.
                for j in range(qdim):
                    for k in range(qdim):

                        dMijdqk = grad(mass_matrix[:, i, j].sum(), q, retain_graph=True)[0][:, k]
                        dMikdqj = grad(mass_matrix[:, i, k].sum(), q, retain_graph=True)[0][:, j]
                        dMkjdqi = grad(mass_matrix[:, k, j].sum(), q, retain_graph=True)[0][:, i]

                        res = res + 0.5 * (dMijdqk + dMikdqj - dMkjdqi) * v[:, k] * v[:, j]

                Cv.append(res)

            Cv = torch.stack(Cv, dim=1).unsqueeze(2)

            return Cv

    def fastest_Cv(self, q, v, mass_matrix=None):
        """ Computes the corriolis matrix times v
        """
        with torch.enable_grad():
            if mass_matrix is None:
                mass_matrix = self.mass_matrix(q)

            Mv = mass_matrix @ v.unsqueeze(2)

            KE = 0.5 * v.unsqueeze(1) @ Mv

            Mvv = Mv * v.unsqueeze(2)

            Cv = torch.autograd.grad(Mvv.sum() - KE.sum(), q, retain_graph=True)[0]

            return Cv.unsqueeze(2)

    def corriolis(self, q, v, mass_matrix=None):
        """ Computes the corriolis matrix
        """
        with torch.enable_grad():
            if mass_matrix is None:
                mass_matrix = self.mass_matrix(q)

            qdim = q.size(1)
            B = mass_matrix.size(0)

            mass_matrix = mass_matrix.reshape(-1, qdim, qdim)

            # TODO vectorize
            rows = []

            for i in range(qdim):
                cols = []
                for j in range(qdim):
                    qgrad = torch.autograd.grad(
                        torch.sum(mass_matrix[:, i, j]), q, retain_graph=True, create_graph=True)[0]
                    cols.append(qgrad)

                rows.append(torch.stack(cols, dim=1))

            dMijk = torch.stack(rows, dim=1)

        corriolis = 0.5 * ((dMijk + dMijk.transpose(2, 3) - dMijk.transpose(1, 3)) @ v.reshape(
            B, 1, qdim, 1)).squeeze(3)
        return corriolis

    def gradpotential(self, q):
        """ Returns the conservative forces acting on the system
        """
        with torch.enable_grad():
            pot = self.potential(q)
            gvec = torch.autograd.grad(torch.sum(pot), q, retain_graph=True, create_graph=True)[0]
        return gvec

    def solve_euler_lagrange(self, q, v, u=None):
        """ Computes `qddot` (generalized acceleration) by solving
        the Euler-Lagrange equation (Eq 7 in the paper)
        \qddot = M^-1 (F - Cv - G)
        """
        with torch.enable_grad():
            with utils.temp_require_grad((q, v)):
                M = self.mass_matrix(q)
                Cv = self.corriolisforce(q, v, M)
                G = self.gradpotential(q)

        F = torch.zeros_like(Cv)

        if u is not None:
            F = self.generalized_force(q, v, u)

        # Solve M \qddot = F - Cv - G
        qddot = torch.solve(F - Cv - G.unsqueeze(2), M)[0].squeeze(2)
        return qddot


class RigidBodyModule(AbstractRigidBody, torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def dynamics(self, q, v, u=None):
        return self.solve_euler_lagrange(q, v, u)

    def kinematics(self, q, v, u=None):
        return v

    def forward(self, t, y, u=None):
        q, v = y
        qdot = self.kinematics(q, v, u)
        vdot = self.dynamics(q, v, u)
        return qdot, vdot

    def get_params(self):
        state = self.state_dict()
        return [v.numpy() for v in state.values()]


class LearnedRigidBody(RigidBodyModule):

    def __init__(self, qdim: int, udim: int, thetamask: torch.tensor, mass_matrix=None,
                 potential=None, generalized_force=None, hidden_sizes=None,
                 slow_corriolis_force=False):
        """

        Arguments:
        - `qdim`:
        - `udim`: [int]
        - `thetamask`: [torch.Tensor (1, qdim)] 1 if angle, 0 otherwise
        - `mass_matrix`: [torch.nn.Module]
        - `potential`: [torch.nn.Module]
        - `generalized_force`: [torch.nn.Module]
        - hidden_sizes: [list]
        """
        self._qdim = qdim
        self._udim = udim

        self._thetamask = thetamask

        super().__init__()

        if mass_matrix is None:
            mass_matrix = CholeskyMMNet(qdim, hidden_sizes=hidden_sizes)

        self._mass_matrix = mass_matrix

        if potential is None:
            potential = PotentialNet(qdim, hidden_sizes=hidden_sizes)

        self._potential = potential

        if generalized_force is None:
            generalized_force = GeneralizedForceNet(qdim, udim, hidden_sizes)

        self._generalized_force = generalized_force

        self._slow_corriolis_force = slow_corriolis_force

    def mass_matrix(self, q):
        return self._mass_matrix(q)

    def potential(self, q):
        return self._potential(q)

    def generalized_force(self, q, v, u):
        return self._generalized_force(q, v, u)

    @property
    def thetamask(self):
        return self._thetamask


class NaiveRigidBody(RigidBodyModule):

    def __init__(self, qdim: int, udim: int, thetamask: torch.tensor, hidden_sizes: Iterable[int]):
        self._qdim = qdim
        self._udim = udim
        self._thetamask = thetamask
        self._hidden_sizes = hidden_sizes
        super().__init__()
        self._net = nn_utils.LNMLP(qdim * 2 + udim, hidden_sizes, qdim)

    @property
    def thetamask(self):
        return self._thetamask

    def dynamics(self, q, v, u=None):
        if u is not None:
            x = torch.cat([q, v, u], dim=1)
        else:
            x = torch.cat([q, v], dim=1)

        return self._net(x)


class DeLan(RigidBodyModule):

    def __init__(self, qdim, n_width, n_hidden, thetamask, activation='ReLu', udim=None, forces=None,
                 bias=10.0, mass_matrix=None, potential=None):
        self._thetamask = thetamask
        self._qdim = qdim
        self._udim = udim
        self._bias = bias

        super().__init__()

        if mass_matrix is None:
            self._mass_matrix_network = DifferentialNetwork(qdim, n_width=n_width, n_hidden=n_hidden,
                                                        activation=activation,
                                                        n_output=int(qdim * (qdim + 1) / 2))
            
        else:
            self._mass_matrix_network = mass_matrix

        if potential is None:
            self._potential_network = DifferentialNetwork(qdim, n_width=n_width, n_hidden=n_hidden,
                                                      activation=activation, n_output=1)
        else:
            self._potential_network = potential

        if forces is None:
            if udim is not None:
                forces = GeneralizedForceNet(qdim, udim, [n_width] * n_hidden)

        self._forces = forces

    @property
    def thetamask(self):
        return self._thetamask


    def embed_to_L(self, L_params, bias=False):
        B = L_params.size(0)
        L_diag = L_params[:, :self._qdim]
        if bias:
            L_diag = L_diag + self._bias
        L_tril = L_params[:, self._qdim:]
        L = L_params.new_zeros(B, self._qdim, self._qdim)
        L = utils.bfill_lowertriangle(L, L_tril)
        L = utils.bfill_diagonal(L, L_diag)

        return L

    def mass_matrix(self, q):
        L_params, dLparamsdq = self._mass_matrix_network(q)
        print("Delan")
        print(L_params)
        L = self.embed_to_L(L_params)
        print(L)
        M = L @ L.transpose(-2, -1)
        return M

    def corriolisforce(self, q, v):
        print("Delan")
        L_params, dLparamsdq = self._mass_matrix_network(q)

        L = self.embed_to_L(L_params, bias=True)

        M = L @ L.transpose(-2, -1)
        # Eqn 12 from Deep Lagrangian Networks
        dLparamsdt = (dLparamsdq @ v.unsqueeze(-1)).squeeze(-1)

        dLdt = self.embed_to_L(dLparamsdt)

        # Eqb 10 from Deep Lagrangian Networks
        dMdt = L @ dLdt.transpose(-2, -1) + dLdt @ L.transpose(-2, -1)

        dMdtv = dMdt @ v.unsqueeze(-1)
        # Eqn 14 from Deep Lagrangian Networks
        dKEdq = []
        for i in range(self._qdim):
            dLdqi = self.embed_to_L(dLparamsdq[..., i])
            _mx = L @ dLdqi.transpose(-2, -1) + dLdqi @ L.transpose(-2, -1)
            dKEdqi = ((_mx @ v.unsqueeze(-1)).squeeze(-1) * v).sum(-1)
            dKEdq.append(dKEdqi)

        dKEdq = torch.stack(dKEdq, dim=1).unsqueeze(-1)

        # Eqn 4 from Deep Lagrangian Networks
        corfor = dMdtv - 0.5 * dKEdq
        return corfor

    def solve_euler_lagrange(self, q, v, u=None):
        """ Computes `qddot` (generalized acceleration) by solving
        the Euler-Lagrange equation (Eq 7 in the paper)
        \qddot = M^-1 (F - Cv - G)
        """

        B = q.size(0)

        L_params, dLparamsdq = self._mass_matrix_network(q)

        L = self.embed_to_L(L_params, bias=True)

        M = L @ L.transpose(-2, -1)

        pot, gradpot = self._potential_network(q)
        gradpot = gradpot.transpose(-2, -1)

        # Eqn 12 from Deep Lagrangian Networks
        dLparamsdt = (dLparamsdq @ v.unsqueeze(-1)).squeeze(-1)

        dLdt = self.embed_to_L(dLparamsdt)

        # Eqb 10 from Deep Lagrangian Networks
        dMdt = L @ dLdt.transpose(-2, -1) + dLdt @ L.transpose(-2, -1)

        dMdtv = dMdt @ v.unsqueeze(-1)
        # Eqn 14 from Deep Lagrangian Networks
        dKEdq = []
        for i in range(self._qdim):
            dLdqi = self.embed_to_L(dLparamsdq[..., i])
            _mx = L @ dLdqi.transpose(-2, -1) + dLdqi @ L.transpose(-2, -1)
            dKEdqi = ((_mx @ v.unsqueeze(-1)).squeeze(-1) * v).sum(-1)
            dKEdq.append(dKEdqi)

        dKEdq = torch.stack(dKEdq, dim=1).unsqueeze(-1)

        # Eqn 4 from Deep Lagrangian Networks
        corfor = dMdtv - 0.5 * dKEdq

        F = self._forces(q, v, u) if self._forces is not None else 0.

        qdd = torch.solve(F - corfor - gradpot, M)[0].squeeze(-1)

        return qdd
