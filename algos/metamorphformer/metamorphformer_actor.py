import math
import torch
import torch.nn as nn
import transformers

from config import cfg
from utils import model as tu
from algos.transformer.transformer_layer import TransformerEncoderLayerResidual, TransformerDecoderLayer
from algos.transformer.decoder import TransformerDecoder
from algos.transformer.encoder import TransformerEncoder
from algos.gpt2.gpt2_model import GPT2Model

class MetamorphformerActor(nn.Module):
    def __init__(self, model_args, limb_embeders, pos_embedders, timestep_embeder, morphology_encoder, extreo_embeders=None, method='metamorphformer', device=None):
        super(MetamorphformerActor, self).__init__()
        self.device = device
        self.method = method
        self.model_args = model_args

        self.envs = self.model_args.envs
        self.morph_emb_dim = self.model_args.morph_emb_dim
        self.latent_emb_dim = self.model_args.latent_emb_dim
        #print('self.latent_emb_dim = %i in actor' % self.latent_emb_dim)

        self.nums_joint, self.dims_proprioceptive, self.dims_bodypart_proprioceptive, self.dims_exteroceptive, self.dims_action, self.dims_bodypart_act = self.model_args.dim_specs

        self.limb_embeders = limb_embeders
        self.extreo_embeders = extreo_embeders
        self.pos_embedders = pos_embedders
        self.timestep_embeder = timestep_embeder
        self.morphology_encoder = morphology_encoder

        # task-specific modules
        self.limb_act_embeders = nn.ModuleDict({
            'unimal': nn.Linear(self.dims_bodypart_act['unimal'], self.morph_emb_dim),
            'walker': nn.Linear(self.dims_bodypart_act['walker'], self.morph_emb_dim),
            'ant': nn.Linear(self.dims_bodypart_act['ant'], self.morph_emb_dim),
            'walker2d': nn.Linear(self.dims_bodypart_act['walker2d'], self.morph_emb_dim),
            'halfcheetah': nn.Linear(self.dims_bodypart_act['halfcheetah'], self.morph_emb_dim),
            'swimmer': nn.Linear(self.dims_bodypart_act['swimmer'], self.morph_emb_dim),
            'reacher': nn.Linear(self.dims_bodypart_act['reacher'], self.morph_emb_dim),
            'hopper': nn.Linear(self.dims_bodypart_act['hopper'], self.morph_emb_dim),
            'humanoid': nn.Linear(self.dims_bodypart_act['humanoid'], self.morph_emb_dim)
        })
        self.action_projectors = nn.ModuleDict({
            'unimal': tu.make_mlp_default([self.latent_emb_dim] + self.model_args.DECODER_DIMS + [self.dims_action['unimal']], final_nonlinearity=False, ),
            'walker': tu.make_mlp_default([self.latent_emb_dim] + self.model_args.DECODER_DIMS + [self.dims_action['walker']], final_nonlinearity=False, ),
            'ant': tu.make_mlp_default([self.latent_emb_dim] + self.model_args.DECODER_DIMS + [self.dims_action['ant']], final_nonlinearity=False, ),
            'walker2d': tu.make_mlp_default([self.latent_emb_dim] + self.model_args.DECODER_DIMS + [self.dims_action['walker2d']], final_nonlinearity=False, ),
            'halfcheetah': tu.make_mlp_default([self.latent_emb_dim] + self.model_args.DECODER_DIMS + [self.dims_action['halfcheetah']], final_nonlinearity=False, ),
            'swimmer': tu.make_mlp_default([self.latent_emb_dim] + self.model_args.DECODER_DIMS + [self.dims_action['swimmer']], final_nonlinearity=False, ),
            'reacher': tu.make_mlp_default([self.latent_emb_dim] + self.model_args.DECODER_DIMS + [self.dims_action['reacher']], final_nonlinearity=False, ),
            'hopper': tu.make_mlp_default([self.latent_emb_dim] + self.model_args.DECODER_DIMS + [self.dims_action['hopper']], final_nonlinearity=False, ),
            'humanoid': tu.make_mlp_default([self.latent_emb_dim] + self.model_args.DECODER_DIMS + [self.dims_action['humanoid']], final_nonlinearity=False, )
        })
        self.obs_projectors = nn.ModuleDict({
            'unimal': tu.make_mlp_default([self.latent_emb_dim] + self.model_args.DECODER_DIMS + [self.dims_proprioceptive['unimal']], final_nonlinearity=False, ),
            'walker': tu.make_mlp_default([self.latent_emb_dim] + self.model_args.DECODER_DIMS + [self.dims_proprioceptive['walker']], final_nonlinearity=False, ),
            'ant': tu.make_mlp_default([self.latent_emb_dim] + self.model_args.DECODER_DIMS + [self.dims_proprioceptive['ant']], final_nonlinearity=False, ),
            'walker2d': tu.make_mlp_default([self.latent_emb_dim] + self.model_args.DECODER_DIMS + [self.dims_proprioceptive['walker2d']], final_nonlinearity=False, ),
            'halfcheetah': tu.make_mlp_default([self.latent_emb_dim] + self.model_args.DECODER_DIMS + [self.dims_proprioceptive['halfcheetah']], final_nonlinearity=False, ),
            'swimmer': tu.make_mlp_default([self.latent_emb_dim] + self.model_args.DECODER_DIMS + [self.dims_proprioceptive['swimmer']], final_nonlinearity=False, ),
            'reacher': tu.make_mlp_default([self.latent_emb_dim] + self.model_args.DECODER_DIMS + [self.dims_proprioceptive['reacher']], final_nonlinearity=False, ),
            'hopper': tu.make_mlp_default([self.latent_emb_dim] + self.model_args.DECODER_DIMS + [self.dims_proprioceptive['hopper']], final_nonlinearity=False, ),
            'humanoid': tu.make_mlp_default([self.latent_emb_dim] + self.model_args.DECODER_DIMS + [self.dims_proprioceptive['humanoid']], final_nonlinearity=False, )
        })
        self.morphology_to_pose_projectors = nn.ModuleDict({
            'unimal': tu.make_mlp_default([self.nums_joint['unimal']*self.morph_emb_dim] + self.model_args.DECODER_DIMS + [self.latent_emb_dim], final_nonlinearity=False, ),
            'walker': tu.make_mlp_default([self.nums_joint['walker']*self.morph_emb_dim] + self.model_args.DECODER_DIMS + [self.latent_emb_dim], final_nonlinearity=False, ),
            'ant': tu.make_mlp_default([self.nums_joint['ant']*self.morph_emb_dim] + self.model_args.DECODER_DIMS + [self.latent_emb_dim], final_nonlinearity=False, ),
            'walker2d': tu.make_mlp_default([self.nums_joint['walker2d']*self.morph_emb_dim] + self.model_args.DECODER_DIMS + [self.latent_emb_dim], final_nonlinearity=False, ),
            'halfcheetah': tu.make_mlp_default([self.nums_joint['halfcheetah']*self.morph_emb_dim] + self.model_args.DECODER_DIMS + [self.latent_emb_dim], final_nonlinearity=False, ),
            'swimmer': tu.make_mlp_default([self.nums_joint['swimmer']*self.morph_emb_dim] + self.model_args.DECODER_DIMS + [self.latent_emb_dim], final_nonlinearity=False, ),
            'reacher': tu.make_mlp_default([self.nums_joint['reacher']*self.morph_emb_dim] + self.model_args.DECODER_DIMS + [self.latent_emb_dim], final_nonlinearity=False, ),
            'hopper': tu.make_mlp_default([self.nums_joint['hopper']*self.morph_emb_dim] + self.model_args.DECODER_DIMS + [self.latent_emb_dim], final_nonlinearity=False, ),
            'humanoid': tu.make_mlp_default([self.nums_joint['humanoid']*self.morph_emb_dim] + self.model_args.DECODER_DIMS + [self.latent_emb_dim], final_nonlinearity=False, )
        })

        # shared modules
        self.s_a_cross = True
        if self.s_a_cross:
            self.decoder_layer = TransformerDecoderLayer(self.morph_emb_dim, self.model_args.NHEAD, self.model_args.DIM_FEEDFORWARD, self.model_args.DROPOUT)
            self.morphology_decoder = TransformerDecoder(self.decoder_layer, 1, norm=None, )

        self.ln_w_current_pose = True
        #if self.ln_w_current_pos:
        #    self.seq_t_linear_weighter = nn.Linear(2 * self.morph_len * decoder_out_dim, self.morph_len * decoder_out_dim)  # 52 => 128

        if cfg.MODEL.TIME_SEQ_MODEL == 'OrgEncoder':
            latent_encode_layer = TransformerEncoderLayerResidual(2 * self.latent_emb_dim, self.model_args.NHEAD, self.model_args.DIM_FEEDFORWARD, self.model_args.DROPOUT)
            self.latent_encoder = TransformerEncoder(latent_encode_layer, 1, norm=None, )
        elif cfg.MODEL.TIME_SEQ_MODEL == 'GPT2':
            if transformers.__version__ == '4.5.1':
                GPT2config = transformers.GPT2Config(
                    vocab_size=1,  # doesn't matter -- we don't use the vocab
                    n_embd=2 * self.latent_emb_dim,
                    n_layer=1,
                    n_head=self.model_args.NHEAD,
                    n_inner=4 * self.latent_emb_dim,
                    activation_function='relu',
                    n_positions=1024,
                    resid_pdrop=self.model_args.DROPOUT,
                    attn_pdrop=self.model_args.DROPOUT,
                    # **kwargs
                )
            else:
                GPT2config = transformers.GPT2Config(
                    vocab_size=1,  # doesn't matter -- we don't use the vocab
                    n_embd=2 * self.latent_emb_dim,
                    n_layer=1,
                    n_head=self.model_args.NHEAD,
                    n_inner=4 * self.latent_emb_dim,
                    activation_function='relu',
                    n_ctx=1024,
                    resid_pdrop=self.model_args.DROPOUT,
                    attn_pdrop=self.model_args.DROPOUT,
                    # **kwargs
                )
            self.latent_encoder = GPT2Model(GPT2config)
        else:
            raise NotImplementedError

        if cfg.MODE != 'pretrain' and cfg.MODEL.ACTOR_CRITIC_SHARE:
            self.value_projector = tu.make_mlp_default([self.morph_emb_dim] + self.model_args.DECODER_DIMS + [1], final_nonlinearity=False, )  # [128, J]

        # init weights
        self.init_weights()
        return

    def init_weights(self):
        initrange = self.model_args.EMBED_INIT

        for env in self.envs:
            self.limb_act_embeders[env].weight.data.uniform_(-initrange, initrange)

        #if self.ln_w_current_pose:
        #    self.seq_t_linear_weighter.weight.data.uniform_(-initrange, initrange)

        initrange = self.model_args.DECODER_INIT
        for env in self.envs:
            self.action_projectors[env][-1].weight.data.uniform_(-initrange, initrange)
            self.action_projectors[env][-1].bias.data.zero_()

            self.obs_projectors[env][-1].weight.data.uniform_(-initrange, initrange)
            self.obs_projectors[env][-1].bias.data.zero_()

            # self.morphology_to_pose_projectors[env][-1].weight.data.uniform_(-initrange, initrange)
            # self.morphology_to_pose_projectors[env][-1].bias.data.zero_()

        if cfg.MODE != 'pretrain' and cfg.MODEL.ACTOR_CRITIC_SHARE:
            self.value_projector[-1].weight.data.uniform_(-initrange, initrange)
            self.value_projector[-1].bias.data.zero_()

        return


    def encode_morphology(self, embeder, pos_embedder, morphology_to_pose_projector, num_joint, inputs, mask=None, seq_mask=None):
        '''
        encode the morphology
        :param inputs: shape=[num_samples,num_joints*nvar_per_joint]
        :param masks: shape=[num_samples,num_joints*nvar_per_joint]
        :return: morphology_embed, shape=[num_limbs,batch_size,embed_dim]
        '''
        if mask is not None:
            inputs = (1.0 - mask)*inputs
        morph_inputs = inputs.reshape(inputs.shape[0], num_joint, -1).permute(1, 0, 2)  # shape = [num_sample, num_joint, nvar_per_joint] => [num_joint, num_sample, nvar_per_joint]

        morph_embed = embeder(morph_inputs) * math.sqrt(self.morph_emb_dim)  # [num_joint, num_sample, nvar_per_joint] -> [num_joint, num_sample, emb_dim]
        morph_embed = pos_embedder(morph_embed)  # shape = [num_joint, num_sample, emb_dim]

        if seq_mask is not None:
            seq_mask = seq_mask.bool()
        pre_pose_embed = self.morphology_encoder(morph_embed, src_key_padding_mask=seq_mask).permute(1, 0, 2)  # shape = [num_joint, num_sample, morph_emb_dim] => [num_sample, num_joint, morph_emb_dim]
        pose_embed = pre_pose_embed.reshape(pre_pose_embed.shape[0], -1)  # shape = [num_sample, num_joint*morph_emb_dim]
        pose_embed = morphology_to_pose_projector(pose_embed)  # shape = [num_sample, latent_emb_dim]
        return pose_embed, pre_pose_embed

    def forward(self, proprioceptive, exteroceptive, act_prev, rew_prev, obs_mask=None, act_mask=None, obs_joint_mask=None, act_joint_mask=None, timemask=None, timestep=None, env=None):
        '''

        :param proprioceptive: shape = [batch_size, T, dim_proprioceptive]
        :param exteroceptive: shape = [batch_size, T, dim_exteroceptive]
        :param act_prev: shape = [batch_size, T, dim_action]
        :param rew_prev: shape = [batch_size, T, 1]
        :param obs_mask: shape = [batch_size, T, dim_proprioceptive]
        :param act_mask: shape = [batch_size, T, dim_action]
        :param obs_joint_mask: shape = [batch_size, T, num_joint]
        :param act_joint_mask: shape = [batch_size, T, num_joint]
        :param timemask: shape = [batch_size, T]
        :param timestep: shape = [batch_size, T]
        :return:
        '''
        has_exteroceptive = exteroceptive is not None and exteroceptive.shape[-1] > 0

        for the_env in self.envs:
            if the_env == env:
                self.limb_embeders[the_env].train()
                self.pos_embedders[the_env].train()
                self.limb_act_embeders[the_env].train()
                self.action_projectors[the_env].train()
                self.obs_projectors[the_env].train()
                self.morphology_to_pose_projectors[the_env].train()
                if has_exteroceptive:
                    self.extreo_embeders[the_env].train()
                else:
                    self.extreo_embeders[the_env].eval()
            else:
                self.limb_embeders[the_env].eval()
                self.pos_embedders[the_env].eval()
                self.limb_act_embeders[the_env].eval()
                self.action_projectors[the_env].eval()
                self.obs_projectors[the_env].eval()
                self.morphology_to_pose_projectors[the_env].eval()
                self.extreo_embeders[the_env].eval()

        batch_size = proprioceptive.shape[0]
        window_size = proprioceptive.shape[1]

        proprioceptive_all = proprioceptive.reshape(batch_size * window_size, proprioceptive.shape[-1])
        obs_mask_all = obs_mask.reshape(batch_size * window_size, obs_mask.shape[-1]) if obs_mask is not None else None
        obs_joint_mask_all = obs_joint_mask.reshape(batch_size * window_size, obs_joint_mask.shape[-1]) if obs_joint_mask is not None else None
        pose_embed_proprioceptive_all, pre_pose_embed_proprioceptive_all = self.encode_morphology(self.limb_embeders[env], self.pos_embedders[env], self.morphology_to_pose_projectors[env], self.nums_joint[env], proprioceptive_all, obs_mask_all, obs_joint_mask_all)  # shape = [batch_size*window_size, num_joints, morph_emb_dim]
        pose_embed_proprioceptive = pose_embed_proprioceptive_all.reshape(batch_size, window_size, self.latent_emb_dim)  # shape = [batch_size, window_size, latent_emb_dim]
        pre_pose_embed_proprioceptive = pre_pose_embed_proprioceptive_all.reshape(batch_size, window_size, self.nums_joint[env], self.morph_emb_dim)

        act_prev_all = act_prev.reshape(batch_size * window_size, act_prev.shape[-1])
        act_mask_all = act_mask.reshape(batch_size * window_size, act_mask.shape[-1]) if act_mask is not None else None
        act_joint_mask_all = act_joint_mask.reshape(batch_size * window_size, act_joint_mask.shape[-1]) if act_joint_mask is not None else None
        pose_embed_act_prev_all, _ = self.encode_morphology(self.limb_act_embeders[env], self.pos_embedders[env], self.morphology_to_pose_projectors[env], self.nums_joint[env], act_prev_all, act_mask_all, act_joint_mask_all)  # shape = [batch_size*window_size, num_joints, morph_emb_dim]
        pose_embed_act_prev = pose_embed_act_prev_all.reshape(batch_size, window_size, self.latent_emb_dim)  # shape = [batch_size, window_size, latent_emb_dim]

        z_a = pose_embed_act_prev
        z_s = pose_embed_proprioceptive

        # exteroceptive
        if not cfg.MODEL.EXTERO_BLIND and has_exteroceptive:
            pose_embed_exteroceptive = self.extreo_embeders[env](exteroceptive)
            z_s += pose_embed_exteroceptive

        s_a_joint_dim = -2
        nvar_timestep = 2  # s and a
        z = torch.cat([z_a.unsqueeze(s_a_joint_dim), z_s.unsqueeze(s_a_joint_dim)], dim=s_a_joint_dim)  # shape = [batch_size, window_size, nvar_timestep, latent_emb_dim]

        if cfg.MODEL.USE_TIMESTEPS and self.timestep_embeder is not None:
            time_embeddings = self.timestep_embeder(timestep.int()).unsqueeze(2)  # shape = [batch_size, window_size] => [batch_size, window_size, latent_emb_dim] => [batch_size, window_size, num_joint, 1, latent_emb_dim]
            z = z + time_embeddings  # shape = [window_size, batch_size, num_joints, nvar_timestep, latent_emb_dim]

        decoder_input = z.reshape(batch_size, window_size, -1)  # shape = [ batch_size, window_size, nvar_timestep*latent_emb_dim]

        if cfg.MODEL.TIME_SEQ_MODEL == 'OrgEncoder':
            latent_attn_mask = None  # some issue with latent_attn_mask
            decoder_output = self.latent_encoder(decoder_input.permute(1, 0, 2), mask=latent_attn_mask, src_key_padding_mask=timemask.float())  # [T, 32, 12*2*128] => [T, 32, 12*2*128]
            decoder_output = decoder_output.permute(1, 0, 2)  # [T, 32, 12, 2, 128]
        elif cfg.MODEL.TIME_SEQ_MODEL == 'GPT2':
            decoder_outputs = self.latent_encoder(inputs_embeds=decoder_input, attention_mask=1.0-timemask.float())  # TODO reverse mask definition in GPT2Model!!!
            decoder_output = decoder_outputs['last_hidden_state']  # shape = [batch_size, window_size, nvar_timestep*latent_emb_dim]
        else:
            raise NotImplementedError
        decoder_output = decoder_output.reshape(batch_size, window_size, nvar_timestep, self.latent_emb_dim)  # shape = [batch_size, window_size, nvar_timestep, latent_emb_dim]

        z_a_out = decoder_output[:, :, 0, :]  # shape = [batch_size, window_size, latent_emb_dim]
        z_s_out = decoder_output[:, :, 1, :]  # shape = [batch_size, window_size, latent_emb_dim]

        # from a_prev to s
        obs_preds = self.obs_projectors[env](z_a_out)  # shape = [batch_size, window_size, dim_proprioceptive]
        if obs_mask is not None:
            obs_preds = (1.0 - obs_mask) * obs_preds
        # from s to a
        action_preds = self.action_projectors[env](z_s_out)  # shape = [batch_size, window_size, dim_action]
        if act_mask is not None:
            action_preds = (1.0 - act_mask) * action_preds

        if cfg.MODE != 'pretrain' and cfg.MODEL.ACTOR_CRITIC_SHARE:
            value_preds = self.value_projector(pre_pose_embed_proprioceptive).reshape(batch_size, window_size, -1)  # shape = [batch_size, window_size, num_joints, 1] => [batch_size, window_size, num_joints]
            if act_joint_mask is not None:
                value_preds = (1.0 -act_joint_mask) * value_preds
            return action_preds, obs_preds, value_preds
        return action_preds, obs_preds
