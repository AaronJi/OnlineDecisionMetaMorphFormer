from typing import Callable, List, Dict, Tuple, Optional, Union

from mlagents_dev.torch_utils import torch, nn
from mlagents_dev.trainers.torch.networks import Actor
from mlagents_dev.trainers.torch.networks import ObservationEncoder

from mlagents_envs.base_env import ActionSpec, ObservationSpec, ObservationType
#from mlagents_dev.trainers.torch.action_model import ActionModel
from mlagents_dev.trainers.torch.agent_action import AgentAction
from mlagents_dev.trainers.torch.action_log_probs import ActionLogProbs
from mlagents_dev.trainers.settings import NetworkSettings, EncoderType, ConditioningType
from mlagents_dev.trainers.buffer import AgentBuffer
from mlagents_dev.trainers.torch.layers import LSTM, LinearEncoder
from mlagents_dev.trainers.torch.conditioning import ConditionalEncoder

from mlagents_dev.trainers.torch.task_specific.ragdoll_space_converter import convert_observation
#from mlagents_dev.trainers.torch.metamorph.metamorph_actor import MetaMorphActor
from mlagents_dev.trainers.torch.metamorphformer.metamorphformer_actor import MetaMorphFormerActor
#from mlagents_dev.trainers.torch.metamorph.action_model import ActionModel
from mlagents_dev.trainers.torch.metamorphformer.action_model import ActionModel
from mlagents_dev.trainers.torch.transformer.transformer_layer import TransformerEncoderLayerResidual
from mlagents_dev.trainers.torch.transformer.encoder import TransformerEncoder
import mlagents_dev.trainers.torch.util.model as tu

class NetworkBody(nn.Module):
    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        encoded_act_size: int = 0,
    ):
        super().__init__()
        self.normalize = network_settings.normalize  # True
        self.use_lstm = network_settings.memory is not None  # False
        self.h_size = network_settings.hidden_units  # 512
        self.m_size = (network_settings.memory.memory_size if network_settings.memory is not None else 0)  # 0

        self.observation_encoder = ObservationEncoder(
            observation_specs,
            self.h_size,
            network_settings.vis_encode_type,  # simple
            self.normalize,  #
        )
        #self.processors = self.observation_encoder.processors
        total_enc_size = self.observation_encoder.total_enc_size  # 243
        total_enc_size += encoded_act_size  # 243+0

        '''
        if (self.observation_encoder.total_goal_enc_size > 0 and network_settings.goal_conditioning_type == ConditioningType.HYPER):  # 0; HYPER
            self._body_endoder = ConditionalEncoder(total_enc_size, self.observation_encoder.total_goal_enc_size, self.h_size, network_settings.num_layers, 1,)
        else:
            self._body_endoder = LinearEncoder(total_enc_size, network_settings.num_layers, self.h_size)  # 243, 3, 512        
        '''

        if self.use_lstm:
            self.lstm = LSTM(self.h_size, self.m_size)
        else:
            self.lstm = None  # type: ignore

        ## new
        self.morph_emb_dim = 128
        self.latent_emb_dim = 512
        morph_len = 16
        dim_limb_obs = 15
        #self.limb_embeder = nn.Linear(dim_limb_obs, self.h_size)

        limb_embeders = nn.ModuleDict({
            'walker': nn.Linear(dim_limb_obs, self.morph_emb_dim),
        })

        # Transformer Encoder
        encoder_layers = TransformerEncoderLayerResidual(
            self.morph_emb_dim, #cfg.MODEL.LIMB_EMBED_SIZE,  # 128
            2, #self.model_args.NHEAD,  # 2
            1024, #self.model_args.DIM_FEEDFORWARD,  # 1024
            0.0, #self.model_args.DROPOUT,  # 0.0
        )
        morphology_encoder = TransformerEncoder(encoder_layers, 5, norm=None, ) # self.model_args.NLAYERS

        extreo_embeders = nn.ModuleDict({
            'walker': MLPObsEncoder(18, out_dim=self.latent_emb_dim),
        })



        self.metamorph_actor = MetaMorphFormerActor(morph_len, limb_embeders['walker'], morphology_encoder, extreo_embeders['walker'])


        return

    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.observation_encoder.update_normalization(buffer)

    def copy_normalization(self, other_network: "NetworkBody") -> None:
        self.observation_encoder.copy_normalization(other_network.observation_encoder)

    @property
    def memory_size(self) -> int:
        return self.lstm.memory_size if self.use_lstm else 0

    def forward(
        self,
        inputs: List[torch.Tensor],  # len = 1; shape = [1, 243]
        actions: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_self = self.observation_encoder(inputs)  # shape = [1, 243]

        if actions is not None:  # does not happen
            encoded_self = torch.cat([encoded_self, actions], dim=1)

        '''
        if isinstance(self._body_endoder, ConditionalEncoder):
            goal = self.observation_encoder.get_goal_encoding(inputs)
            encoding = self._body_endoder(encoded_self, goal)
        else:
            encoding = self._body_endoder(encoded_self)  # shape = [1, 512]        
        '''

        obs_new = convert_observation(encoded_self)

        encoding = self.metamorph_actor(obs_new['proprioceptive'], obs_new['exteroceptive'], obs_new['obs_mask'])

        if self.use_lstm:
            # Resize to (batch, sequence length, encoding size)
            encoding = encoding.reshape([-1, sequence_length, self.h_size])
            encoding, memories = self.lstm(encoding, memories)
            encoding = encoding.reshape([-1, self.m_size // 2])

        return encoding, memories

class SimpleActor(nn.Module, Actor):
    MODEL_EXPORT_VERSION = 3  # Corresponds to ModelApiVersion.MLAgents2_0

    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
    ):
        super().__init__()
        self.action_spec = action_spec  # Continuous: 39, Discrete: ()
        self.version_number = torch.nn.Parameter(torch.Tensor([self.MODEL_EXPORT_VERSION]), requires_grad=False)  # 3?
        self.is_continuous_int_deprecated = torch.nn.Parameter(torch.Tensor([int(self.action_spec.is_continuous())]), requires_grad=False)  # True
        self.continuous_act_size_vector = torch.nn.Parameter(torch.Tensor([int(self.action_spec.continuous_size)]), requires_grad=False)  # 39
        self.discrete_act_size_vector = torch.nn.Parameter(torch.Tensor([self.action_spec.discrete_branches]), requires_grad=False)  # ()
        self.act_size_vector_deprecated = torch.nn.Parameter(torch.Tensor([self.action_spec.continuous_size + sum(self.action_spec.discrete_branches)]), requires_grad=False,)  # 39
        self.network_body = NetworkBody(observation_specs, network_settings)
        if network_settings.memory is not None:
            self.encoding_size = network_settings.memory.memory_size // 2
        else:
            self.encoding_size = network_settings.hidden_units  # 512
        self.memory_size_vector = torch.nn.Parameter(torch.Tensor([int(self.network_body.memory_size)]), requires_grad=False)  # 0

        self.encoding_size = 128
        self.action_model = ActionModel(
            self.encoding_size,
            action_spec,
            conditional_sigma=conditional_sigma,
            tanh_squash=tanh_squash,
            deterministic=network_settings.deterministic,
        )

        return

    @property
    def memory_size(self) -> int:
        return self.network_body.memory_size

    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.network_body.update_normalization(buffer)

    def get_action_and_stats(
        self,
        inputs: List[torch.Tensor],  # [shape=[bsz, 243]]
        masks: Optional[torch.Tensor] = None,  # None
        memories: Optional[torch.Tensor] = None,  # []
        sequence_length: int = 1,  # 1
    ) -> Tuple[AgentAction, ActionLogProbs, torch.Tensor, torch.Tensor]:

        encoding, memories = self.network_body(inputs, memories=memories, sequence_length=sequence_length)
        action, log_probs, entropies = self.action_model(encoding, masks)

        # action: AgentAction.continuous_tensor, shape = [bsz, 39]
        # log_probs: ActionLogProbs.continuous_tensor, shape = [bsz, 39]
        # entropies: shape = [1]
        return action, log_probs, entropies, memories

    def get_stats(
        self,
        inputs: List[torch.Tensor],
        actions: AgentAction,
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[ActionLogProbs, torch.Tensor]:
        encoding, actor_mem_outs = self.network_body(inputs, memories=memories, sequence_length=sequence_length)
        log_probs, entropies = self.action_model.evaluate(encoding, masks, actions)

        return log_probs, entropies

    def forward(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[int, torch.Tensor], ...]:
        """
        Note: This forward() method is required for exporting to ONNX. Don't modify the inputs and outputs.

        At this moment, torch.onnx.export() doesn't accept None as tensor to be exported,
        so the size of return tuple varies with action spec.
        """
        encoding, memories_out = self.network_body(inputs, memories=memories, sequence_length=1)

        (
            cont_action_out,
            disc_action_out,
            action_out_deprecated,
            deterministic_cont_action_out,
            deterministic_disc_action_out,
        ) = self.action_model.get_action_out(encoding, masks)

        export_out = [self.version_number, self.memory_size_vector]
        if self.action_spec.continuous_size > 0:
            export_out += [
                cont_action_out,
                self.continuous_act_size_vector,
                deterministic_cont_action_out,
            ]
        if self.action_spec.discrete_size > 0:
            export_out += [
                disc_action_out,
                self.discrete_act_size_vector,
                deterministic_disc_action_out,
            ]
        if self.network_body.memory_size > 0:
            export_out += [memories_out]

        return tuple(export_out)




class MLPObsEncoder(nn.Module):
    """Encoder for env obs like hfield."""

    def __init__(self, obs_dim, out_dim=None):
        super(MLPObsEncoder, self).__init__()
        mlp_dims = [obs_dim]# + [cfg.MODEL.TRANSFORMER.EXT_HIDDEN_DIMS]
        if out_dim is not None:
            mlp_dims += [out_dim]
        self.encoder = tu.make_mlp_default(mlp_dims)
        self.obs_feat_dim = mlp_dims[-1]

    def forward(self, obs):
        return self.encoder(obs)