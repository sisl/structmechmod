# File: models.py
#
import torch

from structmechmod import nn_utils, utils


class QuadraticLag(torch.nn.Module):

    def __init__(self, neural_net_kwargs, H, quadlag):
        super().__init__()
        self.net = nn_utils.LNMLP(**neural_net_kwargs)
        self._quadlag = quadlag
        if quadlag:
            self.linear_state = torch.nn.Linear(6, 6)
            self.linear_state_min = torch.nn.Linear(6, 6, bias=False)
            self.linear_state_max = torch.nn.Linear(6, 6, bias=False)
            self.linear_controls = [torch.nn.Linear(4, 6, bias=False) for h in range(H)]
            self.linear_controls_min = [torch.nn.Linear(4, 6, bias=False) for h in range(H)]
            self.linear_controls_max = [torch.nn.Linear(4, 6, bias=False) for h in range(H)]

    def forward(self, data_batch):
        # data batch is size ntrajs * H * ndim.
        # Controls are the first 4 dimensions of ndim, then 3 velocities, then 3 rotation rates
        ntrajs = data_batch.shape[0]
        state_t = data_batch[:, -1, 4:]
        pred = self.net(data_batch.view(ntrajs, -1))
        if self._quadlag:
            pred = (
                self.linear_state(state_t) +
                self.linear_state_min(torch.min(state_t, torch.tensor(0.))**2) +
                self.linear_state_max(torch.max(state_t**2, torch.tensor(0.))**2))
            h = 0
            for lc, lcmin, lcmax in zip(self.linear_controls, self.linear_controls_min,
                                        self.linear_controls_max):
                pred = pred + (
                    lc(data_batch[:, h, :4]) +
                    lcmin(torch.min(data_batch[:, h, :4], torch.tensor(0.))**2) +
                    lcmax(torch.max(data_batch[:, h, :4], torch.tensor(0.))**2))
        return pred


class SharedMMVEmbed(torch.nn.Module):

    def __init__(self, qdim, hidden_sizes):
        self._qdim = qdim
        self._hidden_sizes = hidden_sizes
        super().__init__()
        self._lnet = nn_utils.LNMLP(qdim, hidden_sizes[:-1], hidden_sizes[-1])

    def forward(self, q):
        embed = self._lnet(q)
        return embed


class CholeskyMMNet(torch.nn.Module):

    def __init__(self, qdim, embed=None, hidden_sizes=None, bias=10.0, pos_enforce=lambda x: x):
        self._qdim = qdim
        self._bias = bias
        self._pos_enforce = pos_enforce
        super().__init__()
        if embed is None:
            if hidden_sizes is None:
                raise ValueError("embed and hidden_sizes; both can't be None")
            embed = SharedMMVEmbed(qdim, hidden_sizes)

        self.embed = embed
        self.out = torch.nn.Linear(hidden_sizes[-1], int(qdim * (qdim + 1) / 2))

    def forward(self, q):
        B = q.size(0)
        if self._qdim > 1:
            L_params = self.out(self.embed(q))

            L_diag = self._pos_enforce(L_params[:, :self._qdim])
            L_diag += self._bias
            L_tril = L_params[:, self._qdim:]
            L = q.new_zeros(B, self._qdim, self._qdim)
            L = utils.bfill_lowertriangle(L, L_tril)
            L = utils.bfill_diagonal(L, L_diag)
            M = L @ L.transpose(-2, -1)

        else:
            M = self._pos_enforce((self.out(self.embed(q)) + self._bias).unsqueeze(1))

        return M

class DelanCholeskyMMNet(torch.nn.Module):

    def __init__(self, qdim, embed=None, hidden_sizes=None, bias=10.0, pos_enforce=lambda x: x):
        from structmechmod.dnn import DifferentialLayer
        self._qdim = qdim
        self._bias = bias
        self._pos_enforce = pos_enforce
        super().__init__()
        if embed is None:
            if hidden_sizes is None:
                raise ValueError("embed and hidden_sizes; both can't be None")
            #embed = SharedMMVEmbed(qdim, hidden_sizes)
            embed = torch.nn.ModuleList([DifferentialLayer(qdim, hidden_sizes[0], "Tanh"),
                                         DifferentialLayer(hidden_sizes[0], hidden_sizes[1], "Tanh"),
                                         DifferentialLayer(hidden_sizes[1], hidden_sizes[2], "Tanh")])

        self.embed = embed
        #self.out = torch.nn.Linear(hidden_sizes[-1], int(qdim * (qdim + 1) / 2))
        self.out = DifferentialLayer(hidden_sizes[2], int(qdim * (qdim + 1) / 2), "Linear")
        self._eye = torch.eye(qdim).view(1, qdim, qdim)

    def forward(self, q):
        B = q.size(0)
        if self._qdim > 1:
            qd_dq = self._eye.repeat(B, 1, 1)
            # Compute the Network:
            qd, qd_dq = self.embed[0](q, qd_dq)
            for i in range(1, len(self.embed)):
                qd, qd_dq = self.embed[i](qd, qd_dq)

            L_params, _ = self.out(qd, qd_dq)
            print("DelanCholesky")
            print(L_params)
            L_diag = self._pos_enforce(L_params[:, :self._qdim])
            L_diag = L_diag + self._bias
            L_tril = L_params[:, self._qdim:]
            L = q.new_zeros(B, self._qdim, self._qdim)
            L = utils.bfill_lowertriangle(L, L_tril)
            L = utils.bfill_diagonal(L, L_diag)
            M = L @ L.transpose(-2, -1)

        else:
            M = self._pos_enforce((self.out(self.embed(q)) + self._bias).unsqueeze(1))

        return M


class PotentialNet(torch.nn.Module):

    def __init__(self, qdim, embed=None, hidden_sizes=None):
        self._qdim = qdim
        super().__init__()
        if embed is None:
            if hidden_sizes is None:
                raise ValueError("embed and hidden_sizes; both can't be None")

            embed = SharedMMVEmbed(qdim, hidden_sizes)

        self.embed = embed
        self.out = torch.nn.Linear(hidden_sizes[-1], 1)

    def forward(self, q):
        return self.out(self.embed(q))

class ZeroPotential(torch.nn.Module):
    def __init__(self, qdim, *args):
        self._qdim = qdim
        super().__init__()

    def forward(self, q):
        return torch.zeros(q.size(0), 1)
    
class DelanZeroPotential(torch.nn.Module):
    def __init__(self, qdim, *args):
        self._qdim = qdim
        super().__init__()

    def forward(self, q):
        return (torch.zeros(q.size(0), 1), torch.zeros(q.size(0), 1, q.size(1)))


class GeneralizedForceNet(torch.nn.Module):

    def __init__(self, qdim, udim, hidden_sizes):
        self._qdim = qdim
        self._udim = udim
        self._hidden_sizes = hidden_sizes
        super().__init__()
        self._net = nn_utils.LNMLP(self._qdim * 2 + self._udim, hidden_sizes, qdim)

    def forward(self, q, v, u):
        B = q.size(0)
        x = torch.cat([q, v, u], dim=-1)
        F = self._net(x)
        F = F.unsqueeze(2)
        assert F.shape == (B, self._qdim, 1), F.shape
        return F

class QVForceNet(torch.nn.Module):
    def __init__(self, qdim, hidden_sizes):
        self._qdim = qdim
        self._hidden_sizes = hidden_sizes
        super().__init__()
        self._net = nn_utils.LNMLP(self._qdim*2, hidden_sizes, qdim)

    def forward(self, q,v, u):
        B = q.size(0)
        x = torch.cat([q, v], dim=-1)
        F = self._net(x)
        F = F.unsqueeze(2)
        assert F.shape == (B, self._qdim, 1), F.shape
        return F

class ControlAffineForceNet(torch.nn.Module):

    def __init__(self, qdim, udim, hidden_sizes):
        self._qdim = qdim
        self._udim = udim
        self._hidden_sizes = hidden_sizes
        super().__init__()
        self._net = nn_utils.LNMLP(self._qdim, hidden_sizes, qdim * udim)

    def forward(self, q, v, u):
        B = q.size(0)
        Bmat = self._net(q).view(B, self._qdim, self._udim)
        F = Bmat @ u.unsqueeze(2)
        assert F.shape == (B, self._qdim, 1), F.shape
        return F


class ControlAffineLinearForce(torch.nn.Module):

    def __init__(self, B, diag_embed=False):
        """
        B needs to be shaped (qdim, udim) usually diagonal
        """
        super().__init__()
        if not isinstance(B, torch.nn.Parameter):
            B = torch.nn.Parameter(B)

        self._B = B
        self._diag_embed = diag_embed

    def forward(self, q, v, u):
        N = q.size(0)
        assert u.size(0) == N
        if not self._diag_embed:
            assert self._B.shape == (q.size(1), u.size(1)), self._B.shape
            B = self._B.unsqueeze(0)
            F = B @ u.unsqueeze(2)
        else:
            F = (self._B * u).unsqueeze(2)
        assert F.shape == (N, q.size(1), 1), F.shape
        return F


class ViscousJointDampingForce(torch.nn.Module):

    def __init__(self, eta):
        """
        eta needs to be shaped (1, qdim)
        """
        super().__init__()
        if not isinstance(eta, torch.nn.Parameter):
            eta = torch.nn.Parameter(eta)

        self._eta = eta

    def forward(self, q, v, u):
        N = q.size(0)
        assert self._eta.size(1) == v.size(1)
        F = (self._eta * v).unsqueeze(2)
        assert F.shape == (N, q.size(1), 1), F.shape
        return F


class GeneralizedForces(torch.nn.Module):

    def __init__(self, forces):
        super().__init__()
        self.forces = torch.nn.ModuleList(forces)

    def forward(self, q, v, u):
        F = torch.zeros(q.size(0), q.size(1), 1)
        for f in self.forces:
            f_ = f(q, v, u)
            F += f_

        return F


class NamedGeneralizedForces(torch.nn.Module):

    def __init__(self, forces_dict):
        super().__init__()
        self.forces = torch.nn.ModuleDict(forces_dict)

    def forward(self, q, v, u):
        force_names = list(self.forces.keys())
        F = self.forces[force_names[0]](q, v, u)
        for f in force_names[1:]:
            f_ = self.forces[f](q, v, u)
            F += f_
        return F


class DeLanCholeskyMMNet(torch.nn.Module):

    def __init__(self, qdim, embed, bias):
        self._qdim = qdim
        super().__init__()
        self.embed = embed
        self.bias = bias
        self.ld = torch.nn.Sequential(
            torch.nn.Linear(self.embed._hidden_sizes[-1], self._qdim), torch.nn.ReLU())
        self.lo = torch.nn.Linear(self.embed._hidden_sizes[-1], int(qdim * (qdim - 1) / 2))

    def forward(self, q):
        B = q.size(0)
        embed = self.embed(q)
        ld = self.ld(embed)
        ld = ld + self.bias
        lo = self.lo(embed)
        L = q.new_zeros(B, self._qdim, self._qdim)
        L = utils.bfill_diagonal(L, ld)
        L = utils.bfill_lowertriangle(L, lo)
        M = L @ L.transpose(-2, -1)
        return M
