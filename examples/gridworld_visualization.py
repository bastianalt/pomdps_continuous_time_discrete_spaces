import sys
import torch
import matplotlib.pyplot as plt
import torch.distributions
from matplotlib import ticker

sys.path.append('..')
from packages.models.pomdps.learners.collocation import CollocationLearner, ExactAdvantageFunction
from examples.models.gridworld import Gridworld, GridWorldVisualization
from packages.models.pomdps.policys import AdvantagePolicy, AdvantageValuePolicy
from packages.models.components.advantages import AdvantageNet
from packages.models.components.values import ValueNet

advantage_updating = True
save = True
simulation_file = ''
# settings
exact_advantage_values = False
checkpoint_file = 'model_col_gw'
simulation_file = ''

# if you want to compute error of the loaded models, train for some steps again
evaluate_errors = False

# load checkpoint
checkpoint = torch.load("checkpoints/"+checkpoint_file, map_location=torch.device('cpu'))

# construct pomdp
pomdp = Gridworld(**checkpoint['pomdp_params'])

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

# make one hot beliefs and compute value/advantage values
beliefs = torch.eye(num_states)

valvals = value_net(beliefs, compute_grad=False).detach()[:, 0]
advals = advantage_fun(beliefs).detach()

if evaluate_errors:
    # sampling function that samples exactly the beliefs that will be tested for
    def belief_sample_fun(num):
        assert(num == pomdp.n_states)
        batch = torch.eye(num)
        return batch

    belief_prior = torch.ones(pomdp.n_states)

    # Init learner that uses the collocation method
    learner = CollocationLearner(pomdp, num_episodes=5, name='test',
                                 checkpoint_interval=-1, value_net=value_net, advantage_net=advantage_fun,
                                 batch_size=pomdp.n_states, num_iter_v=5, num_iter_a=5,
                                 belief_sample_fn=belief_sample_fun)

    learner.learn()

# normalize values for plotting
valvals_norm = valvals - torch.min(valvals)
valvals_norm /= torch.max(valvals_norm).abs()
valvals_norm = valvals

vis = GridWorldVisualization(pomdp)
vp = vis.plot_values(valvals_norm, walls_fill=valvals_norm.min())
vis.plot_walls()

# Add simulation
# settings
if simulation_file:
    # load checkpoint
    checkpoint = torch.load("checkpoints/"+simulation_file, map_location=torch.device('cpu'))

    # construct pomdp
    pomdp = Gridworld(**checkpoint['pomdp_params'])

    # get latent states
    states_time = torch.stack([torch.as_tensor(0.)] + list(map(lambda x: x.t_end, checkpoint['latent_traj'])))
    states_list = list(map(lambda x: x.state, checkpoint['latent_traj']))
    states_full = torch.stack(states_list[:1] + states_list)[:, 0]

    # get beliefs, actions, value and advantage values
    belief_time = torch.stack(list(map(lambda x: x.t, checkpoint['observed_traj'])))
    belief_actions = torch.stack(list(map(lambda x: x.action, checkpoint['observed_traj'])))
    belief_beliefs = torch.stack(list(map(lambda x: x.belief, checkpoint['observed_traj'])))

    adv_vals = checkpoint['advantage_values']
    min_advvals, _ = torch.min(adv_vals, dim=1, keepdim=True)
    adv_vals -= min_advvals
    adv_vals /= torch.norm(adv_vals, dim=1, keepdim=True)
    val_vals = checkpoint['value_values']

    # get observations
    obs_time = torch.stack(list(map(lambda x: x.time, checkpoint['observation_traj'])))
    obs_obs = torch.stack(list(map(lambda x: x.observation, checkpoint['observation_traj'])))

    t_max = 8.5
    idx_too_much = torch.where(states_time >= t_max)[0]
    if list(idx_too_much.size())[0] > t_max:
        endidx = idx_too_much[0]
    else:
        endidx = None

    vis.plot_trajectories(states_full[None, :endidx])

if advantage_updating:
    labelstr = 'au'
else:
    labelstr = 'col'

vis.plot_advantages(advals)
if save:
    plt.savefig('gridworld_'+labelstr)
plt.show()

# extra plot for heatmap
plt.figure()
heatmap = plt.pcolor(vp)
cb = plt.colorbar(heatmap, orientation="horizontal", pad=0.2)
cb.set_label('Value function')
tick_locator = ticker.MaxNLocator(nbins=4)
cb.locator = tick_locator
cb.update_ticks()
if save:
    plt.savefig('gridworld_'+labelstr+'_legend')
plt.show()
