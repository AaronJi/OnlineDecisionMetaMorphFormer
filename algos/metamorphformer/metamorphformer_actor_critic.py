
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from config import cfg
from utils import model as tu
from algos.transformer.positional_encoder import PositionalEncoding, PositionalEncoding1D
from algos.transformer.encoder import TransformerEncoder
from algos.transformer.transformer_layer import TransformerEncoderLayerResidual, TransformerDecoderLayer
from .metamorphformer_actor import MetamorphformerActor
#from transformers.modeling_utils import PreTrainedModel

import envs.converters.ragdoll_space_converter as rg_spec

# TODO post-calculation in source code! need to find a better way
#cfg.MODEL.MAX_JOINTS = 16
#cfg.MODEL.MAX_LIMBS = 12

def get_last_from_seq(seq):
    if seq is None:
        return seq

    if isinstance(seq, dict):
        last = {}
        for ot in seq:
            last[ot] = seq[ot][:, -1, :]
    else:
        last = seq[:, -1, :]
    return last

class MetamorphformerActorCritic(nn.Module):
    def __init__(self, method='metamorphformer', device=None):
        super(MetamorphformerActorCritic, self).__init__()
        self.device = device
        self.config = cfg

        ## arguments
        self.model_args = cfg.MODEL.TRANSFORMER

        self.morph_emb_dim = cfg.MODEL.LIMB_EMBED_SIZE  # 128
        #self.latent_emb_dim = self.morph_emb_dim
        self.latent_emb_dim = 512
        #self.latent_emb_dim = 128

        self.nums_joint = {}
        self.dims_proprioceptive = {}
        self.dims_bodypart_proprioceptive = {}
        self.dims_exteroceptive = {}
        self.dims_action = {}
        self.dims_bodypart_act = {}

        # unimal
        self.nums_joint['unimal'] = 12
        self.dims_bodypart_proprioceptive['unimal'] = 52
        self.dims_proprioceptive['unimal'] = 624
        self.dims_exteroceptive['unimal'] = 1410
        self.dims_bodypart_act['unimal'] = 2
        self.dims_action['unimal'] = 24

        # ragdoll (walker)
        self.nums_joint[rg_spec.env] = rg_spec.n_bodyparts
        self.dims_bodypart_proprioceptive[rg_spec.env] = rg_spec.dim_bodypart_proprioceptive
        self.dims_proprioceptive[rg_spec.env] = rg_spec.dim_proprioceptive
        self.dims_exteroceptive[rg_spec.env] = rg_spec.dim_exteroceptive
        self.dims_bodypart_act[rg_spec.env] = rg_spec.dim_bodypart_act
        self.dims_action[rg_spec.env] = rg_spec.dim_act

        # ant
        self.nums_joint['ant'] = 8
        self.dims_bodypart_proprioceptive['ant'] = 2
        self.dims_proprioceptive['ant'] = 16
        self.dims_exteroceptive['ant'] = 95
        self.dims_bodypart_act['ant'] = 1
        self.dims_action['ant'] = 8

        # halfcheetah
        self.nums_joint['halfcheetah'] = 6
        self.dims_bodypart_proprioceptive['halfcheetah'] = 2
        self.dims_proprioceptive['halfcheetah'] = 12
        self.dims_exteroceptive['halfcheetah'] = 5
        self.dims_bodypart_act['halfcheetah'] = 1
        self.dims_action['halfcheetah'] = 6

        # hopper
        self.nums_joint['hopper'] = 3
        self.dims_bodypart_proprioceptive['hopper'] = 2
        self.dims_proprioceptive['hopper'] = 6
        self.dims_exteroceptive['hopper'] = 5
        self.dims_bodypart_act['hopper'] = 1
        self.dims_action['hopper'] = 3

        # walker2d
        self.nums_joint['walker2d'] = 6
        self.dims_bodypart_proprioceptive['walker2d'] = 2
        self.dims_proprioceptive['walker2d'] = 12
        self.dims_exteroceptive['walker2d'] = 5
        self.dims_bodypart_act['walker2d'] = 1
        self.dims_action['walker2d'] = 6

        # humanoid
        self.nums_joint['humanoid'] = 9
        self.dims_bodypart_proprioceptive['humanoid'] = 6
        self.dims_proprioceptive['humanoid'] = 54
        self.dims_exteroceptive['humanoid'] = 11+140+84+23+84
        self.dims_bodypart_act['humanoid'] = 3
        self.dims_action['humanoid'] = 27

        # swimmer
        self.nums_joint['swimmer'] = 2
        self.dims_bodypart_proprioceptive['swimmer'] = 2
        self.dims_proprioceptive['swimmer'] = 4
        self.dims_exteroceptive['swimmer'] = 4
        self.dims_bodypart_act['swimmer'] = 1
        self.dims_action['swimmer'] = 2

        # reacher
        self.nums_joint['reacher'] = 2
        self.dims_bodypart_proprioceptive['reacher'] = 3
        self.dims_proprioceptive['reacher'] = 6
        self.dims_exteroceptive['reacher'] = 5
        self.dims_bodypart_act['reacher'] = 1
        self.dims_action['reacher'] = 2

        self.model_args.morph_emb_dim = self.morph_emb_dim
        self.model_args.latent_emb_dim = self.latent_emb_dim

        self.model_args.dim_specs = (self.nums_joint, self.dims_proprioceptive, self.dims_bodypart_proprioceptive, self.dims_exteroceptive, self.dims_action, self.dims_bodypart_act)

        ## envs
        self.envs = list(self.nums_joint.keys())
        print('Init Metamorphformer for envs %s.' % str(self.envs))
        self.model_args.envs = self.envs

        ## shared modules
        if cfg.MODEL.USE_TIMESTEPS:
            timestep_embeder = nn.Embedding(cfg.MODEL.TIMESTEP_BUCKETS, self.latent_emb_dim)
        else:
            timestep_embeder = None

        # Transformer Encoder
        encoder_layers = TransformerEncoderLayerResidual(
            cfg.MODEL.LIMB_EMBED_SIZE,  # 128
            self.model_args.NHEAD,  # 2
            self.model_args.DIM_FEEDFORWARD,  # 1024
            self.model_args.DROPOUT,  # 0.0
        )
        morphology_encoder = TransformerEncoder(encoder_layers, self.model_args.NLAYERS, norm=None, )

        ## task-specific modules
        self.limb_embeders = nn.ModuleDict({
            'unimal': nn.Linear(self.dims_bodypart_proprioceptive['unimal'], self.morph_emb_dim),
            'walker': nn.Linear(self.dims_bodypart_proprioceptive['walker'], self.morph_emb_dim),
            'ant': nn.Linear(self.dims_bodypart_proprioceptive['ant'], self.morph_emb_dim),
            'walker2d': nn.Linear(self.dims_bodypart_proprioceptive['walker2d'], self.morph_emb_dim),
            'halfcheetah': nn.Linear(self.dims_bodypart_proprioceptive['halfcheetah'], self.morph_emb_dim),
            'swimmer': nn.Linear(self.dims_bodypart_proprioceptive['swimmer'], self.morph_emb_dim),
            'reacher': nn.Linear(self.dims_bodypart_proprioceptive['reacher'], self.morph_emb_dim),
            'hopper': nn.Linear(self.dims_bodypart_proprioceptive['hopper'], self.morph_emb_dim),
            'humanoid': nn.Linear(self.dims_bodypart_proprioceptive['humanoid'], self.morph_emb_dim)
        })
        extreo_embeders = nn.ModuleDict({
            'unimal': MLPObsEncoder(self.dims_exteroceptive['unimal'], out_dim=self.latent_emb_dim),
            'walker': MLPObsEncoder(self.dims_exteroceptive['walker'], out_dim=self.latent_emb_dim),
            'ant': MLPObsEncoder(self.dims_exteroceptive['ant'], out_dim=self.latent_emb_dim),
            'walker2d': MLPObsEncoder(self.dims_exteroceptive['walker2d'], out_dim=self.latent_emb_dim),
            'halfcheetah': MLPObsEncoder(self.dims_exteroceptive['halfcheetah'], out_dim=self.latent_emb_dim),
            'swimmer': MLPObsEncoder(self.dims_exteroceptive['swimmer'], out_dim=self.latent_emb_dim),
            'reacher': MLPObsEncoder(self.dims_exteroceptive['reacher'], out_dim=self.latent_emb_dim),
            'hopper': MLPObsEncoder(self.dims_exteroceptive['hopper'], out_dim=self.latent_emb_dim),
            'humanoid': MLPObsEncoder(self.dims_exteroceptive['humanoid'], out_dim=self.latent_emb_dim)
        })
        pos_embedders = nn.ModuleDict({
            'unimal': PositionalEncoding(self.morph_emb_dim, self.nums_joint['unimal']) if self.model_args.POS_EMBEDDING == "learnt" else PositionalEncoding1D(self.morph_emb_dim, self.nums_joint['unimal']),
            'walker': PositionalEncoding(self.morph_emb_dim, self.nums_joint['walker']) if self.model_args.POS_EMBEDDING == "learnt" else PositionalEncoding1D(self.morph_emb_dim, self.nums_joint['walker']),
            'ant': PositionalEncoding(self.morph_emb_dim, self.nums_joint['ant']) if self.model_args.POS_EMBEDDING == "learnt" else PositionalEncoding1D(self.morph_emb_dim, self.nums_joint['ant']),
            'walker2d': PositionalEncoding(self.morph_emb_dim, self.nums_joint['walker2d']) if self.model_args.POS_EMBEDDING == "learnt" else PositionalEncoding1D(self.morph_emb_dim, self.nums_joint['walker2d']),
            'halfcheetah': PositionalEncoding(self.morph_emb_dim, self.nums_joint['halfcheetah']) if self.model_args.POS_EMBEDDING == "learnt" else PositionalEncoding1D(self.morph_emb_dim, self.nums_joint['halfcheetah']),
            'swimmer': PositionalEncoding(self.morph_emb_dim, self.nums_joint['swimmer']) if self.model_args.POS_EMBEDDING == "learnt" else PositionalEncoding1D(self.morph_emb_dim, self.nums_joint['swimmer']),
            'reacher': PositionalEncoding(self.morph_emb_dim, self.nums_joint['reacher']) if self.model_args.POS_EMBEDDING == "learnt" else PositionalEncoding1D(self.morph_emb_dim, self.nums_joint['reacher']),
            'hopper': PositionalEncoding(self.morph_emb_dim, self.nums_joint['hopper']) if self.model_args.POS_EMBEDDING == "learnt" else PositionalEncoding1D(self.morph_emb_dim, self.nums_joint['hopper']),
            'humanoid': PositionalEncoding(self.morph_emb_dim, self.nums_joint['humanoid']) if self.model_args.POS_EMBEDDING == "learnt" else PositionalEncoding1D(self.morph_emb_dim, self.nums_joint['humanoid'])
        })

        #if self.env_name == 'unimal':
        #    self.num_action = 24

        self.log_stds = {}
        #self.stds = {}
        for env in self.envs:
            if cfg.MODEL.ACTION_STD_FIXED:
                log_std = nn.Parameter(np.log(cfg.MODEL.ACTION_STD) * torch.ones(1, self.dims_action[env]), requires_grad=False)
            else:
                log_std = nn.Parameter(torch.zeros(1, self.num_action))
            self.log_stds[env] = log_std
            #self.stds[env] = torch.exp(log_std)

        ##actor & critic
        self.mu_net = MetamorphformerActor(self.model_args, self.limb_embeders, pos_embedders, timestep_embeder, morphology_encoder, extreo_embeders=extreo_embeders, method=method, device=self.device)

        #if not cfg.MODEL.ACTOR_CRITIC_SHARE:
        #    self.v_net = MetamorphformerCritic(self.limb_embeder, pos_embedding, timestep_embeder, morphology_encoder, hfield_encoder, method=method, use_act_mask=use_act_mask)

        ## init weight
        self.init_weights()
        return


    def init_weights(self):
        initrange = self.model_args.EMBED_INIT
        for env in self.envs:
            self.limb_embeders[env].weight.data.uniform_(-initrange, initrange)
        return

    def forward(
            self,
            obs=None,
            act_prev=None,
            rew_prev=None,
            timemask=None,
            timestep=None,
            #act=None,
            #source_lengths=None,
            env=None):

        proprioceptive, exteroceptive, obs_mask, act_mask, obs_joint_mask, act_joint_mask = (
            obs["proprioceptive"],  # shape = [batch_size, T, 624]
            obs["exteroceptive"],  # shape = [batch_size, T, 624]
            obs["obs_mask"],
            obs["act_mask"],
            obs["obs_joint_mask"],
            obs["act_joint_mask"],
        )

        if cfg.MODE == 'pretrain':  # and cfg.MODEL.ACTOR_CRITIC_SHARE:
            action_preds, obs_preds = self.mu_net(proprioceptive, exteroceptive, act_prev, rew_prev, obs_mask, act_mask, obs_joint_mask, act_joint_mask, timemask, timestep, env=env)
            return action_preds, obs_preds
        else:
            mus, obs_preds, limb_vals = self.mu_net(proprioceptive, exteroceptive, act_prev, rew_prev, obs_mask, act_mask, obs_joint_mask, act_joint_mask, timemask, timestep, env=env)

            obs_joint_mask_t = get_last_from_seq(obs_joint_mask) if cfg.MODEL.TIME_SEQ else obs_joint_mask
            limb_vals_t = get_last_from_seq(limb_vals) if cfg.MODEL.TIME_SEQ else limb_vals
            # Zero out mask values
            limb_vals_t = limb_vals_t * (1 - obs_joint_mask_t.int())  # shape = [batch_size, 12]
            # Use avg/max to keep the magnitidue same instead of sum
            num_limbs = self.nums_joint[env] - torch.sum(obs_joint_mask_t.int(), dim=1, keepdim=True)  # shape = [batch_size, 1]
            val = torch.divide(torch.sum(limb_vals_t, dim=1, keepdim=True), num_limbs)  # shape = [batch_size, 1]

            return val, mus, obs_preds


class MLPObsEncoder(nn.Module):
    """Encoder for env obs like hfield."""

    def __init__(self, obs_dim, out_dim=None):
        super(MLPObsEncoder, self).__init__()
        mlp_dims = [obs_dim] + cfg.MODEL.TRANSFORMER.EXT_HIDDEN_DIMS
        if out_dim is not None:
            mlp_dims += [out_dim]
        self.encoder = tu.make_mlp_default(mlp_dims)
        self.obs_feat_dim = mlp_dims[-1]

    def forward(self, obs):
        return self.encoder(obs)