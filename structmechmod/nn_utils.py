import re
import torch
from torch import nn


class MinNormBarrier:

    def __init__(self, alpha: float) -> None:
        self._alpha = alpha

    def __call__(self, parameter: torch.Tensor) -> torch.Tensor:
        val = torch.clamp(torch.sum(torch.pow(parameter, 2)) / self._alpha, 0.0, 1.0)
        return -torch.log(val)


class WeightNormValue:

    def __call__(self, parameter: torch.Tensor) -> torch.Tensor:
        val = torch.sum(torch.pow(parameter, 2))
        return val


class Swish(nn.Module):

    def __init__(self, inplace=False):
        super().__init__()
        self._inplace = inplace

    def forward(self, x):
        if self._inplace:
            x.mul_(torch.sigmoid(x))
            return x

        return x * torch.sigmoid(x)


class BentIdentity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (torch.sqrt(x**2 + 1) - 1) / 2 + x


class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


ACTIVATIONS = {
    'bent': BentIdentity,
    'tanh': nn.Tanh,
    'swish': Swish,
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'identity': Identity
}


class LNMLP(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size, activation='tanh', gain=1.0,
                 ln=False):
        super().__init__()
        Activation = ACTIVATIONS[activation]
        if len(hidden_sizes) > 0:
            layers = [nn.Linear(input_size, hidden_sizes[0])]
            layers.append(Activation())
            if ln:
                layers.append(nn.LayerNorm(hidden_sizes[0]))
            for i in range(len(hidden_sizes) - 1):
                layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
                layers.append(Activation())
                if ln:
                    layers.append(nn.LayerNorm(hidden_sizes[i + 1]))

            layers.append(nn.Linear(hidden_sizes[-1], output_size))
        else:
            layers = [nn.Linear(input_size, output_size)]
        self._layers = layers
        self._mlp = nn.Sequential(*layers)
        self.reset_params(gain=gain)

    def forward(self, input_):
        return self._mlp(input_)

    def reset_params(self, gain=1.0):
        self.apply(lambda x: weights_init_mlp(x, gain=gain))


def softabs(x):
    return torch.sqrt(x**2 + 1e-3)


def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


def weights_init_mlp(m, gain=1.0):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init_normc_(m.weight.data, gain)
        if m.bias is not None:
            m.bias.data.fill_(0)


def disable_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def print_model_params(m):
    print('-' * 10)
    for name, param in m.named_parameters():
        if param.requires_grad:
            print(name)
            print(param)

    print('-' * 10)


class ApplyRegularizer:

    def __init__(self, regularizers):
        self._regularizers = regularizers

    def __call__(self, module: torch.nn.Module) -> torch.Tensor:
        accumulator = 0.0
        for name, parameter in module.named_parameters():
            for regex, regularizer in self._regularizers:
                if re.search(regex, name):
                    penalty = regularizer(parameter)
                    accumulator += penalty
                    break

        return accumulator


def enable_activation_logging(model, logger):
    """
    Log activations

    We register a forward hook to capture the output tensors
    """

    for name, module in model.named_modules():
        if getattr(module, 'should_not_log_activations', False):
            # allow skipping certain modules
            continue

        def hook(module_, inputs, outputs):
            log_prefix = 'activation_histogram/{}_{}'.format(name, module_.__class__.__name__)
            log_prefixmu = 'activation_mean/{}_{}'.format(name, module_.__class__.__name__)
            log_prefixstd = 'activation_std/{}_{}'.format(name, module_.__class__.__name__)
            if isinstance(outputs, torch.Tensor):
                log_name = log_prefix
                out = outputs.clone().detach().numpy()
                logger.add_hist(log_name, out)
                logger.logkv(log_prefixmu, out.mean())
                logger.logkv(log_prefixstd, out.std())

            elif isinstance(outputs, (list, tuple)):
                for i, output in enumerate(outputs):
                    log_name = "{0}_{1}".format(log_prefix, i)
                    out = output.clone().detach().numpy()
                    logger.add_hist(log_name, out)
                    logger.logkv("{0}_{1}".format(log_prefixmu, i), out.mean())
                    logger.logkv("{0}_{1}".format(log_prefixstd, i), out.std())

            elif isinstance(outputs, dict):
                for k, output in outputs.items():
                    log_name = "{0}_{1}".format(log_prefix, k)
                    out = output.clone().detach().numpy()
                    logger.add_hist(log_name, out)
                    logger.logkv("{0}_{1}".format(log_prefixmu, k), out.mean())
                    logger.logkv("{0}_{1}".format(log_prefixstd, k), out.std())

            else:
                pass

        module.register_forward_hook(hook)
