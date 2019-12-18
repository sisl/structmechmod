import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


## Code below modified from
# https://git.ias.informatik.tu-darmstadt.de/lutter/deep_differential_network/blob/master/deep_differential_network/differentialNetwork.py
# See https://arxiv.org/pdf/1907.04490.pdf


class SoftplusDer(nn.Module):

    def __init__(self, beta=1.):
        super(SoftplusDer, self).__init__()
        self._beta = beta

    def forward(self, x):
        cx = torch.clamp(x, -20., 20.)
        exp_x = torch.exp(self._beta * cx)
        out = exp_x / (exp_x + 1.0)

        if torch.isnan(out).any():
            print("SoftPlus Forward output is NaN.")
        return out


class ReLUDer(nn.Module):

    def __init__(self):
        super(ReLUDer, self).__init__()

    def forward(self, x):
        return torch.ceil(torch.clamp(x, 0, 1))


class Linear(nn.Module):

    def __init__(self):
        super(Linear, self).__init__()

    def forward(self, x):
        return x


class LinearDer(nn.Module):

    def __init__(self):
        super(LinearDer, self).__init__()

    def forward(self, x):
        return torch.clamp(x, 1, 1)


class Cos(nn.Module):

    def __init__(self):
        super(Cos, self).__init__()

    def forward(self, x):
        return torch.cos(x)


class CosDer(nn.Module):

    def __init__(self):
        super(CosDer, self).__init__()

    def forward(self, x):
        return -torch.sin(x)

class Tanh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.tanh(x)

class TanhDer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1.0 - torch.tanh(x)**2

class DifferentialLayer(nn.Module):

    def __init__(self, input_size, n_dof, activation="ReLu"):
        super(DifferentialLayer, self).__init__()

        # Create layer weights and biases:
        self.n_dof = n_dof
        self.weight = nn.Parameter(torch.Tensor(n_dof, input_size))
        self.bias = nn.Parameter(torch.Tensor(n_dof))

        # Initialize activation function and its derivative:
        if activation == "ReLu" or activation == "ReLU":
            self.g = nn.ReLU()
            self.g_prime = ReLUDer()

        elif activation == "SoftPlus":
            self.softplus_beta = 1.0
            self.g = nn.Softplus(beta=self.softplus_beta)
            self.g_prime = SoftplusDer(beta=self.softplus_beta)

        elif activation == "Cos":
            self.g = Cos()
            self.g_prime = CosDer()

        elif activation == "Linear":
            self.g = Linear()
            self.g_prime = LinearDer()

        elif activation == "Tanh":
            self.g = Tanh()
            self.g_prime = TanhDer()

        else:
            raise ValueError(
                "Activation Type must be in ['Linear', 'ReLu', 'SoftPlus', 'Cos'] but is {0}"
                .format(self.activation))

    def forward(self, q, der_prev):
        """
        Calculates forward pass of the Lagrangian Layer
        :param q: input to the current layer
        :param der_prev: derivative of the previous layer w.r.t. the network input
        :return: output tensor, the derivative of the output tensor w.r.t. q
        """

        # Apply Affine Transformation:
        a = F.linear(q, self.weight, self.bias)
        out = self.g(a)
        der = torch.matmul(self.g_prime(a).view(-1, self.n_dof, 1) * self.weight, der_prev)
        return out, der


class DifferentialNetwork(nn.Module):

    def __init__(self, n_dof, **kwargs):
        super(DifferentialNetwork, self).__init__()

        # Read optional arguments:
        self._n_dof = n_dof
        self.n_width = kwargs.get("n_width", 128)
        self.n_hidden = kwargs.get("n_hidden", 1)
        self.n_output = kwargs.get("n_output", 1)
        non_linearity = kwargs.get("activation", "ReLu")

        # Initialization of the layers:
        self._w_init = kwargs.get("w_init", "xavier_normal")
        self._b0 = kwargs.get("b_init", 0.1)
        self._g_hidden = kwargs.get("g_hidden", np.sqrt(2.))
        self._g_output = kwargs.get("g_output", 1.0)
        self._mean_hidden = kwargs.get("mean_hidden", 0.0)
        self._mean_output = kwargs.get("mean_output", 0.0)
        self._p_sparse = kwargs.get("p_sparse", 0.2)

        # Construct Weight Initialization:
        if self._w_init == "xavier_normal":

            # Construct initialization function:
            def init_hidden(layer):

                # Set the Hidden Gain:
                if self._g_hidden <= 0.0:
                    hidden_gain = torch.nn.init.calculate_gain('relu')
                else:
                    hidden_gain = self._g_hidden

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.xavier_normal_(layer.weight, hidden_gain)

                with torch.no_grad():
                    layer.weight = torch.nn.Parameter(layer.weight + self._mean_hidden)

            def init_output(layer):
                # Set Output Gain:
                if self._g_output <= 0.0:
                    output_gain = torch.nn.init.calculate_gain('linear')
                else:
                    output_gain = self._g_output

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.xavier_normal_(layer.weight, output_gain)
                # torch.nn.init.uniform_(layer.weight, a=-output_gain, b=output_gain)

                # with torch.no_grad():
                #     layer.weight = torch.nn.Parameter(layer.weight + self._mean_output)

        elif self._w_init == "orthogonal":

            # Construct initialization function:
            def init_hidden(layer):
                # Set the Hidden Gain:
                if self._g_hidden <= 0.0:
                    hidden_gain = torch.nn.init.calculate_gain('relu')
                else:
                    hidden_gain = self._g_hidden

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.orthogonal_(layer.weight, hidden_gain)

            def init_output(layer):
                # Set Output Gain:
                if self._g_output <= 0.0:
                    output_gain = torch.nn.init.calculate_gain('linear')
                else:
                    output_gain = self._g_output

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.orthogonal_(layer.weight, output_gain)

        elif self._w_init == "sparse":
            assert self._p_sparse < 1. and self._p_sparse >= 0.0

            # Construct initialization function:
            def init_hidden(layer):
                p_non_zero = self._p_sparse
                hidden_std = self._g_hidden

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.sparse_(layer.weight, p_non_zero, hidden_std)

            def init_output(layer):
                p_non_zero = self._p_sparse
                output_std = self._g_output

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.sparse_(layer.weight, p_non_zero, output_std)

        else:
            raise ValueError(
                "Weight Initialization Type must be in ['xavier_normal', 'orthogonal', 'sparse'] "
                "but is {0}".format(self._w_init))

        # Create Network:
        self.layers = nn.ModuleList()

        # Create Input Layer:f
        self.layers.append(DifferentialLayer(self._n_dof, self.n_width, activation=non_linearity))
        init_hidden(self.layers[-1])

        # Create Hidden Layer:
        for _ in range(1, self.n_hidden):
            self.layers.append(
                DifferentialLayer(self.n_width, self.n_width, activation=non_linearity))
            init_hidden(self.layers[-1])

        # Create output Layer:
        self.layers.append(DifferentialLayer(self.n_width, self.n_output, activation="Linear"))
        # self.layers.append(LagrangianLayer(self.n_width, self.n_output, activation=non_linearity))
        init_output(self.layers[-1])

        self._eye = torch.eye(self._n_dof).view(1, self._n_dof, self._n_dof)
        self.device = self._eye.device

    def forward(self, q):

        # Create initial derivative of dq/ dq.
        # qd_dq = self._eye.repeat(q.shape[0], 1, 1).type_as(q)
        qd_dq = self._eye.repeat(q.shape[0], 1, 1)

        # Compute the Network:
        qd, qd_dq = self.layers[0](q, qd_dq)
        for i in range(1, len(self.layers)):
            qd, qd_dq = self.layers[i](qd, qd_dq)

        return qd, qd_dq

    def cuda(self, device=None):

        # Move the Network to the GPU:
        super(DifferentialNetwork, self).cuda(device=device)

        # Move the eye matrix to the GPU:
        self._eye = self._eye.cuda()
        self.device = self._eye.device
        return self

    def cpu(self):

        # Move the Network to the CPU:
        super(DifferentialNetwork, self).cpu()

        # Move the eye matrix to the CPU:
        self._eye = self._eye.cpu()
        self.device = self._eye.device
        return self
