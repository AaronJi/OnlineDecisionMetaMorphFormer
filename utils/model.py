import torch
import torch.nn as nn


def w_init(module, gain=1):
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0)
    return module


def make_mlp(dim_list):
    init_ = lambda m: w_init(m)

    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(init_(nn.Linear(dim_in, dim_out)))
        layers.append(nn.Tanh())

    return nn.Sequential(*layers)


def make_mlp_default(dim_list, final_nonlinearity=True, nonlinearity="relu"):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if nonlinearity == "relu":
            layers.append(nn.ReLU())
        elif nonlinearity == "tanh":
            layers.append(nn.Tanh())

    if not final_nonlinearity:
        layers.pop()
    return nn.Sequential(*layers)


def num_params(model, only_trainable=True):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = model.parameters()
    if only_trainable:
        parameters = list(p for p in parameters if p.requires_grad)
    unique = dict((p.data_ptr(), p) for p in parameters).values()
    return sum(p.numel() for p in unique)

def load_from_pt(model, pt_path, cfg, key_state_dict=None, load_full_model=True):
    if cfg.DEVICE == 'cpu':
        checkpoint = torch.load(pt_path, map_location=torch.device('cpu'))  # TODO
    else:
        try:
            checkpoint = torch.load(pt_path)
        except:
            print('Map checkpoint to DEVICE %s.' % cfg.DEVICE)
            checkpoint = torch.load(pt_path, map_location=torch.device(cfg.DEVICE))

    if key_state_dict is None:
        #model.load_state_dict(checkpoint)
        state_dict0 = checkpoint
    else:
        #model.load_state_dict(checkpoint[key_state_dict])
        state_dict0 = checkpoint[key_state_dict]

    #state_dict0 = model_ckpt.state_dict()
    if load_full_model:
        model.load_state_dict(state_dict0)
    else:
        state_dict = model.state_dict()
        same_params = {k: v for k, v in state_dict0.items() if k in state_dict}
        state_dict.update(same_params)
        model.load_state_dict(state_dict)

    if 'ob_rms' in checkpoint:
        ob_rms = checkpoint['ob_rms']
    else:
        ob_rms = None
    return ob_rms

def load_from_pt_old(model, pt_path, cfg, key_model=None, load_full_model=True):
    if cfg.DEVICE == 'cpu':
        checkpoint = torch.load(pt_path, map_location=torch.device('cpu'))  # TODO
    else:
        checkpoint = torch.load(pt_path)
    if key_model is None:
        model_ckpt = checkpoint['model']
    else:
        model_ckpt = checkpoint[key_model]

    state_dict0 = model_ckpt.state_dict()
    if load_full_model:
        model.load_state_dict(state_dict0)
    else:
        state_dict = model.state_dict()
        same_params = {k: v for k, v in state_dict0.items() if k in state_dict}
        state_dict.update(same_params)
        model.load_state_dict(state_dict)

    if 'ob_rms' in checkpoint:
        ob_rms = checkpoint['ob_rms']
    else:
        ob_rms = None
    return ob_rms


def load_partial_from_pt(model, pt_path, cfg, key_state_dict=None):
    if cfg.DEVICE == 'cpu':
        checkpoint = torch.load(pt_path, map_location=torch.device('cpu'))  # TODO
    else:
        checkpoint = torch.load(pt_path)
    if key_state_dict is None:
        state_dict0 = checkpoint
    else:
        state_dict0 = checkpoint[key_state_dict]

    state_dict = model.state_dict()
    same_params = {k: v for k, v in state_dict0.items() if k in state_dict}
    state_dict.update(same_params)
    model.load_state_dict(state_dict)

    if 'ob_rms' in checkpoint:
        ob_rms = checkpoint['ob_rms']
    else:
        ob_rms = None
    return ob_rms



def get_param_norm(model):
    model_state_dict = model.state_dict()
    norm_sum = 0.0
    dict_norm = {}
    for name, param in model_state_dict.items():
        param_norm = torch.norm(param.float(), p=2)
        dict_norm[name] = param_norm
        norm_sum += param_norm

    return norm_sum, dict_norm
