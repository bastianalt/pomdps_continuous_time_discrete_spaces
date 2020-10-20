import sys
import torch.distributions

sys.path.append('..')

from packages.models.components.filters import ContinuousRandomDiscreteFilter
from packages.models.pomdps.simulator import POMDPSimulator
from examples.models.slotted_aloha import SlottedAloha
from packages.models.pomdps.policys import AdvantagePolicy, AdvantageValuePolicy
from packages.models.pomdps.learners.collocation import ExactAdvantageFunction
from packages.models.components.advantages import AdvantageNet
from packages.models.components.values import ValueNet

# === Settings ===

# evaluate exact advantage function
exact_advantage_values = True

# checkpoint file that contains the model
checkpoint_file = 'model_col_sa'

# length of simulation
sim_len = 100

# seed for simulation
seed = 0


# === Processing ===

# load checkpoint
checkpoint = torch.load("checkpoints/" + checkpoint_file, map_location=torch.device('cpu'))

# construct pomdp
pomdp = SlottedAloha(**checkpoint['pomdp_params'])

# initialize value and advantage nets
num_states = pomdp.SSpace.nElements
value_net = ValueNet(num_states)
value_net.load_state_dict(checkpoint['value_net_state_dict'])

if not exact_advantage_values:
    advantage_fun = AdvantageNet(num_states, pomdp.ASpace.nElements)
    advantage_fun.load_state_dict(checkpoint['advantage_net_state_dict'])
    policy = AdvantagePolicy(pomdp.SSpace, pomdp.ASpace, advantage_fun)
else:
    advantage_fun = ExactAdvantageFunction(pomdp, value_net)
    policy = AdvantageValuePolicy(pomdp, value_net)

# initialize filter
filter = ContinuousRandomDiscreteFilter(pomdp.SSpace,
                                        pomdp.ASpace,
                                        pomdp.TModel,
                                        pomdp.OModel,
                                        policy)

t_grid = torch.linspace(0, sim_len, 200)
num_transition_states = pomdp.n_transition_states
num_packages = pomdp.n_packages
initial_belief = torch.zeros(num_transition_states, num_packages)
initial_belief[0, 0] = 1.
initial_belief = initial_belief.flatten()
start_state = torch.as_tensor(0).reshape(1, -1)

ode_options = {'method': 'rk4',
               'options': {'step_size': .01}}

# set seed
torch.manual_seed(seed)

# sample a trajectory
sim = POMDPSimulator(pomdp)
latent_traj, observed_traj, observation_traj = sim.sampleTraj(t_grid=t_grid,
                                                              pi=policy,
                                                              filter=filter,
                                                              initial_belief=initial_belief,
                                                              start_state=start_state,
                                                              ode_options=ode_options)

# compute advantage values for all beliefs of the trajectory
belief_time = torch.stack(list(map(lambda x: x.t, observed_traj)))
belief_beliefs = torch.stack(list(map(lambda x: x.belief, observed_traj)))
advantage_values = advantage_fun(belief_beliefs).detach()
value_values = value_net(belief_beliefs, compute_grad=False)[:, 0].detach()

torch.save({
    'latent_traj': latent_traj,
    'observed_traj': observed_traj,
    'observation_traj': observation_traj,
    'advantage_values': advantage_values,
    'value_values': value_values,
    'pomdp_params': checkpoint['pomdp_params'],
    'learner_params': checkpoint['learner_params'],
}, "checkpoints/sim_aloha_" + str(seed) + "_" + checkpoint_file)

print('success.')
