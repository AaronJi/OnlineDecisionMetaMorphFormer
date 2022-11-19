import math
import torch
import torch.nn as nn

import mlagents_dev.trainers.torch.util.model as tu

class MetaMorphFormerActor(nn.Module):

    def __init__(self, morph_len, limb_embeder, morphology_encoder, extreo_embeder):
        super(MetaMorphFormerActor, self).__init__()
        self.morph_len = morph_len
        self.morph_emb_dim = 128
        self.limb_embeder = limb_embeder
        self.morphology_encoder = morphology_encoder
        self.extreo_embeder = extreo_embeder

        self.latent_emb_dim = 512

        #self.action_projectors = nn.ModuleDict({
        #    'walker': tu.make_mlp_default([self.latent_emb_dim] + [64], final_nonlinearity=False, ),
        #})
        #  + self.model_args.DECODER_DIMS

        self.morphology_to_pose_projectors = nn.ModuleDict({
            'walker': tu.make_mlp_default([self.morph_emb_dim] + [self.latent_emb_dim], final_nonlinearity=False, ),
        })
        #  + self.model_args.DECODER_DIMS
        # 16 *

        self.action_projector = tu.make_mlp_default([self.latent_emb_dim] + [] + [4], final_nonlinearity=False, )  # [128, J]
        # self.model_args.DECODER_DIMS

        ## init weight
        self.init_weights()
        return

    def init_weights(self):
        # initrange = cfg.MODEL.TRANSFORMER.EMBED_INIT
        initrange = 0.1
        self.limb_embeder.weight.data.uniform_(-initrange, initrange)

        #initrange = 0.01  # cfg.MODEL.TRANSFORMER.DECODER_INIT
        #self.action_projectors['walker'][-1].weight.data.uniform_(-initrange, initrange)
        #self.action_projectors['walker'][-1].bias.data.zero_()

        initrange = 0.01  # cfg.MODEL.TRANSFORMER.DECODER_INIT
        self.action_projector[-1].weight.data.uniform_(-initrange, initrange)
        self.action_projector[-1].bias.data.zero_()
        return

    def encode_morphology(self, proprioceptive, obs_mask, return_attention=False):
        '''
        encode the morphology at time t
        :param obs_t: shape=[batch_size,num_limbs*52]
        :param obs_mask_t: shape=[batch_size,num_limbs]
        :param return_attention:
        :return: morphology_embed, shape=[num_limbs,batch_size,embed_dim]
        '''

        proprioceptive = proprioceptive.reshape(proprioceptive.shape[0], self.morph_len, -1).permute(1, 0, 2)  # shape = [batch_size, num_limbs, dim_limb_obs] => [num_limbs, batch_size, dim_limb_obs]
        proprioceptive_embed = self.limb_embeder(proprioceptive) * math.sqrt(self.morph_emb_dim)  # (num_limbs, batch_size, limb_obs_size) -> (num_limbs, batch_size, embed_dim)

        #if self.model_args.POS_EMBEDDING in ["learnt", "abs"]:
        #    obs_embed = self.pos_embedding(obs_embed)  # shape = [12, 32, 128]

        attention_maps = None
        if return_attention:
            morphology_embed, attention_maps = self.morphology_encoder.get_attention_maps(proprioceptive_embed, src_key_padding_mask=None)  # obs_mask_t: [batch_size, num_limbs]  TODO
        else:
            # (num_limbs, batch_size, d_model)
            morphology_embed = self.morphology_encoder(proprioceptive_embed, src_key_padding_mask=None)  # shape = [num_limbs, batch_size, emb_size]

        pre_pose_embed = morphology_embed.permute(1, 0, 2)
        #pose_embed = pre_pose_embed.reshape(pre_pose_embed.shape[0], -1)
        pose_embed_proprioceptive = self.morphology_to_pose_projectors['walker'](pre_pose_embed)  # shape = [num_sample, latent_emb_dim]

        return pose_embed_proprioceptive

    def forward(self, proprioceptive, exteroceptive, obs_mask):
        pose_embed_proprioceptive = self.encode_morphology(proprioceptive, obs_mask)

        pose_embed_exteroceptive = self.extreo_embeder(exteroceptive).unsqueeze(1).repeat(1, self.morph_len, 1)
        #pose_embed_exteroceptive = pose_embed_exteroceptive.unsqueeze(1)
        #pose_embed_exteroceptive = pose_embed_exteroceptive.repeat(1, self.morph_len, 1)
        pose_embed = pose_embed_proprioceptive + pose_embed_exteroceptive

        #pose_embed = pose_embed_proprioceptive

        #z_s = pose_embed_proprioceptive
        #z_s += pose_embed_exteroceptive

        #decoder_input = self.action_projectors['walker'](z_s)

        #pose_embed = morphology_embed.permute(1, 0, 2)
        #pose_embed = self.action_projector(pose_embed)
        #pose_embed = pose_embed.reshape(pose_embed.shape[0], -1)
        #return decoder_input
        # hfield_obs_emb.repeat(self.morph_len, 1)
        output = self.action_projector(pose_embed)
        output = output.reshape(output.shape[0], -1)

        return output
