from typing import List, Tuple, NamedTuple, Optional
from mlagents_dev.torch_utils import torch, nn
from mlagents_dev.trainers.torch.distributions import (
    DistInstance,
    DiscreteDistInstance,
    GaussianDistribution,
    GaussianDistribution0,
    MultiCategoricalDistribution,
)
from mlagents_dev.trainers.torch.agent_action import AgentAction
from mlagents_dev.trainers.torch.action_log_probs import ActionLogProbs
from mlagents_envs.base_env import ActionSpec

from mlagents_dev.trainers.torch.util import model as tu
from mlagents_dev.trainers.torch.task_specific.ragdoll_space_converter import convert_action, convert_action_back, act_mask

EPSILON = 1e-7  # Small value to avoid divide by zero


class DistInstances(NamedTuple):
    """
    A NamedTuple with fields corresponding the the DistInstance objects
    output by continuous and discrete distributions, respectively. Discrete distributions
    output a list of DistInstance objects whereas continuous distributions output a single
    DistInstance object.
    """

    continuous: Optional[DistInstance]
    discrete: Optional[List[DiscreteDistInstance]]


class ActionModel(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        action_spec: ActionSpec,
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
        deterministic: bool = False,
    ):
        """
        A torch module that represents the action space of a policy. The ActionModel may contain
        a continuous distribution, a discrete distribution or both where construction depends on
        the action_spec.  The ActionModel uses the encoded input of the network body to parameterize
        these distributions. The forward method of this module outputs the action, log probs,
        and entropies given the encoding from the network body.
        :params hidden_size: Size of the input to the ActionModel.
        :params action_spec: The ActionSpec defining the action space dimensions and distributions.
        :params conditional_sigma: Whether or not the std of a Gaussian is conditioned on state.
        :params tanh_squash: Whether to squash the output of a Gaussian with the tanh function.
        :params deterministic: Whether to select actions deterministically in policy.
        """
        super().__init__()
        self.encoding_size = hidden_size
        self.action_spec = action_spec
        self._continuous_distribution = None
        self._discrete_distribution = None

        if self.action_spec.continuous_size > 0:
            self._continuous_distribution = GaussianDistribution0(
                16*4, #self.action_spec.continuous_size,
                conditional_sigma=conditional_sigma,  # False
                tanh_squash=tanh_squash,  # False
            )
            #self._continuous_distribution = GaussianDistribution(
            #    self.encoding_size,
            #    64, #self.action_spec.continuous_size,
            #    conditional_sigma=conditional_sigma,
            #    tanh_squash=tanh_squash,
            #)


        if self.action_spec.discrete_size > 0:
            self._discrete_distribution = MultiCategoricalDistribution(self.encoding_size, self.action_spec.discrete_branches)

        # During training, clipping is done in TorchPolicy, but we need to clip before ONNX
        # export as well.
        self._clip_action_on_export = not tanh_squash  # True
        self._deterministic = deterministic  # False

        #self.action_projector = tu.make_mlp_default([hidden_size] + [] + [4], final_nonlinearity=False, )  # [128, J]
        # self.model_args.DECODER_DIMS

        #self.init_weights()
        return

    def init_weights(self):
        initrange = 0.01  # cfg.MODEL.TRANSFORMER.DECODER_INIT
        self.action_projector[-1].weight.data.uniform_(-initrange, initrange)
        self.action_projector[-1].bias.data.zero_()
        return

    def _sample_action(self, dists: DistInstances) -> AgentAction:
        """
        Samples actions from a DistInstances tuple
        :params dists: The DistInstances tuple
        :return: An AgentAction corresponding to the actions sampled from the DistInstances
        """

        continuous_action: Optional[torch.Tensor] = None
        discrete_action: Optional[List[torch.Tensor]] = None
        # This checks None because mypy complains otherwise
        if dists.continuous is not None:
            if self._deterministic:
                continuous_action = dists.continuous.deterministic_sample()
            else:
                continuous_action = dists.continuous.sample()

            # TODO mask
            act_mask_batch = continuous_action * 0 + act_mask
            continuous_action = (1.0 - act_mask_batch) * continuous_action

        if dists.discrete is not None:
            discrete_action = []
            if self._deterministic:
                for discrete_dist in dists.discrete:
                    discrete_action.append(discrete_dist.deterministic_sample())
            else:
                for discrete_dist in dists.discrete:
                    discrete_action.append(discrete_dist.sample())
        return AgentAction(continuous_action, discrete_action)

    def _get_dists(self, inputs: torch.Tensor, masks: torch.Tensor) -> DistInstances:
        """
        Creates a DistInstances tuple using the continuous and discrete distributions
        :params inputs: The encoding from the network body
        :params masks: Action masks for discrete actions
        :return: A DistInstances tuple
        """
        continuous_dist: Optional[DistInstance] = None
        discrete_dist: Optional[List[DiscreteDistInstance]] = None
        # This checks None because mypy complains otherwise
        if self._continuous_distribution is not None:
            continuous_dist = self._continuous_distribution(inputs)
        if self._discrete_distribution is not None:
            discrete_dist = self._discrete_distribution(inputs, masks)
        return DistInstances(continuous_dist, discrete_dist)

    def _get_probs_and_entropy(self, actions: AgentAction, dists: DistInstances) -> Tuple[ActionLogProbs, torch.Tensor]:
        """
        Computes the log probabilites of the actions given distributions and entropies of
        the given distributions.
        :params actions: The AgentAction
        :params dists: The DistInstances tuple
        :return: An ActionLogProbs tuple and a torch tensor of the distribution entropies.
        """
        entropies_list: List[torch.Tensor] = []
        continuous_log_prob: Optional[torch.Tensor] = None
        discrete_log_probs: Optional[List[torch.Tensor]] = None
        all_discrete_log_probs: Optional[List[torch.Tensor]] = None
        # This checks None because mypy complains otherwise

        if dists.continuous is not None:
            continuous_log_prob = dists.continuous.log_prob(actions.continuous_tensor)
            entropies_list.append(dists.continuous.entropy())

        if dists.discrete is not None:
            discrete_log_probs = []
            all_discrete_log_probs = []
            for discrete_action, discrete_dist in zip(actions.discrete_list, dists.discrete):  # type: ignore
                discrete_log_prob = discrete_dist.log_prob(discrete_action)
                entropies_list.append(discrete_dist.entropy())
                discrete_log_probs.append(discrete_log_prob)
                all_discrete_log_probs.append(discrete_dist.all_log_prob())
        action_log_probs = ActionLogProbs(continuous_log_prob, discrete_log_probs, all_discrete_log_probs)
        entropies = torch.cat(entropies_list, dim=1)
        return action_log_probs, entropies

    def evaluate(self, inputs: torch.Tensor, masks: torch.Tensor, actions: AgentAction) -> Tuple[ActionLogProbs, torch.Tensor]:
        """
        Given actions and encoding from the network body, gets the distributions and
        computes the log probabilites and entropies.
        :params inputs: The encoding from the network body
        :params masks: Action masks for discrete actions
        :params actions: The AgentAction
        :return: An ActionLogProbs tuple and a torch tensor of the distribution entropies.
        """
        #inputs = self.action_projector(inputs)
        #inputs = inputs.permute(1, 0, 2)
        #inputs = inputs.reshape(inputs.shape[0], -1)

        continuous_action_new = convert_action(actions.continuous_tensor)
        actions_new = AgentAction(continuous_action_new, None)
        dists = self._get_dists(inputs, masks)

        log_probs, entropies = self._get_probs_and_entropy(actions_new, dists)

        # Use the sum of entropy across actions, not the mean
        entropy_sum = torch.sum(entropies, dim=1)

        continuous_log_probs_new = convert_action_back(log_probs.continuous_tensor)
        log_probs_new = ActionLogProbs(continuous_log_probs_new, None, None)
        #print('^^^^0')
        #print(inputs.shape)
        #print(continuous_action_new.shape)
        #print(continuous_log_probs_new.shape)
        #exit(3)
        return log_probs_new, entropy_sum

    def get_action_out(self, inputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Gets the tensors corresponding to the output of the policy network to be used for
        inference. Called by the Actor's forward call.
        :params inputs: The encoding from the network body
        :params masks: Action masks for discrete actions
        :return: A tuple of torch tensors corresponding to the inference output
        """
        #inputs = self.action_projector(inputs)
        #inputs = inputs.permute(1, 0, 2)
        #inputs = inputs.reshape(inputs.shape[0], -1)

        dists = self._get_dists(inputs, masks)
        continuous_out, discrete_out, action_out_deprecated = None, None, None
        deterministic_continuous_out, deterministic_discrete_out = (None, None,)  # deterministic actions
        if self.action_spec.continuous_size > 0 and dists.continuous is not None:
            continuous_out = dists.continuous.exported_model_output()
            action_out_deprecated = continuous_out
            deterministic_continuous_out = dists.continuous.deterministic_sample()

            continuous_out = convert_action_back(continuous_out)
            action_out_deprecated = convert_action_back(action_out_deprecated)
            deterministic_continuous_out = convert_action_back(deterministic_continuous_out)

            if self._clip_action_on_export:
                continuous_out = torch.clamp(continuous_out, -3, 3) / 3
                action_out_deprecated = continuous_out
                deterministic_continuous_out = (torch.clamp(deterministic_continuous_out, -3, 3) / 3)
        if self.action_spec.discrete_size > 0 and dists.discrete is not None:
            discrete_out_list = [discrete_dist.exported_model_output() for discrete_dist in dists.discrete]
            discrete_out = torch.cat(discrete_out_list, dim=1)
            action_out_deprecated = torch.cat(discrete_out_list, dim=1)
            deterministic_discrete_out_list = [discrete_dist.deterministic_sample() for discrete_dist in dists.discrete]
            deterministic_discrete_out = torch.cat(deterministic_discrete_out_list, dim=1)

        # deprecated action field does not support hybrid action
        if self.action_spec.continuous_size > 0 and self.action_spec.discrete_size > 0:
            action_out_deprecated = None

        #print('^^^^1')
        #print(inputs.shape)
        #print(continuous_out.shape)
        #print(deterministic_continuous_out.shape)
        #exit(3)
        return (
            continuous_out,
            discrete_out,
            action_out_deprecated,
            deterministic_continuous_out,
            deterministic_discrete_out,
        )

    def forward(self, inputs: torch.Tensor, masks: torch.Tensor) -> Tuple[AgentAction, ActionLogProbs, torch.Tensor]:
        """
        The forward method of this module. Outputs the action, log probs,
        and entropies given the encoding from the network body.
        :params inputs: The encoding from the network body
        :params masks: Action masks for discrete actions
        :return: Given the input, an AgentAction of the actions generated by the policy and the corresponding
        ActionLogProbs and entropies.
        """

        #inputs = self.action_projector(inputs)
        #inputs = inputs.permute(1, 0, 2)
        #inputs = inputs.reshape(inputs.shape[0], -1)

        dists = self._get_dists(inputs, masks)
        actions = self._sample_action(dists)
        log_probs, entropies = self._get_probs_and_entropy(actions, dists)
        # Use the sum of entropy across actions, not the mean
        entropy_sum = torch.sum(entropies, dim=1)

        continuous_action_new = convert_action_back(actions.continuous_tensor)
        continuous_log_probs_new = convert_action_back(log_probs.continuous_tensor)

        actions_new = AgentAction(continuous_action_new, None)
        log_probs_new = ActionLogProbs(continuous_log_probs_new, None, None)
        #print('^^^^2')
        #print(inputs.shape)
        #print(continuous_action_new.shape)
        #print(continuous_log_probs_new.shape)
        #exit(3)
        return (actions_new, log_probs_new, entropy_sum)

'''
def conv_act_back(continuous_action):

    continuous_action = continuous_action.permute(1, 0, 2)
    continuous_action = continuous_action.reshape(continuous_action.shape[0], -1)
    act_mask_batch = act_mask.repeat(continuous_action.shape[0], 1)
    continuous_action = (1.0 - act_mask_batch) * continuous_action
    continuous_action = convert_action_back(continuous_action)
    return continuous_action

def conv_act(continuous_action):

    continuous_action = convert_action(continuous_action)
    act_mask_batch = act_mask.repeat(continuous_action.shape[0], 1)
    continuous_action = (1.0 - act_mask_batch) * continuous_action
    continuous_action = continuous_action.reshape(continuous_action.shape[0], 16, -1)
    continuous_action = continuous_action.permute(1, 0, 2)
    return continuous_action
'''
