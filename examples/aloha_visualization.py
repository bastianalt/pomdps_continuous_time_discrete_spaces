import sys
import matplotlib.pyplot as plt
import torch
from matplotlib import ticker

sys.path.append('..')
from examples.models.slotted_aloha import SlottedAloha

# settings
simulation_file = 'sim_aloha_col'

abs_time = False
colorbar = False
colorbar_aspect = 11.
n_bins = 3

t_min = 0.0
t_max = 100.0

# load checkpoint
checkpoint = torch.load("checkpoints/"+simulation_file, map_location=torch.device('cpu'))

# construct pomdp
pomdp = SlottedAloha(**checkpoint['pomdp_params'])


def constrain_time(time, t_min, t_max):
    t_min = float(t_min)
    t_max = float(t_max)

    states_valid = torch.where((time >= t_min) * (time <= t_max))[0]

    if states_valid[0] == 0:
        start_idx = 0
        start_time = torch.empty(0)
    else:
        start_idx = states_valid[0] - 1
        start_time = torch.tensor(t_min).reshape(1)  # states_time[states_valid[0] - 1]

    res_time = torch.cat((start_time, time[states_valid[0]:(states_valid[-1] + 1)]))
    i_min = start_idx
    i_max = states_valid[-1] + 1
    return res_time, i_min, i_max


# get latent states
states_time = torch.stack([torch.as_tensor(0.)] + list(map(lambda x: x.t_end, checkpoint['latent_traj'])))
states_full = torch.stack(list(map(lambda x: x.state, checkpoint['latent_traj'])))
states_time, i_min, i_max = constrain_time(states_time, t_min, t_max)
states_values = states_full[i_min:i_max]
states_pack = states_values % pomdp.n_packages
states_td = states_values // pomdp.n_packages
states_time -= t_min

# get beliefs, actions, value and advantage values
belief_time = torch.stack(list(map(lambda x: x.t, checkpoint['observed_traj'])))
belief_actions = torch.stack(list(map(lambda x: x.action, checkpoint['observed_traj'])))
belief_beliefs = torch.stack(list(map(lambda x: x.belief, checkpoint['observed_traj']))).reshape(-1, pomdp.n_transition_states, pomdp.n_packages)
belief_time, i_min, i_max = constrain_time(belief_time, t_min, t_max)
belief_actions = belief_actions[i_min:i_max]
belief_beliefs = belief_beliefs[i_min:i_max]
belief_time -= t_min

belief_ts = torch.sum(belief_beliefs, dim=-1)
belief_pack = torch.sum(belief_beliefs, dim=1)
adv_vals = checkpoint['advantage_values'][i_min:i_max]
min_advvals, _ = torch.min(adv_vals, dim=1, keepdim=True)
adv_vals -= min_advvals
adv_vals /= torch.norm(adv_vals, dim=1, keepdim=True)
val_vals = checkpoint['value_values'][i_min:i_max]

# get observations
obs_time = torch.stack(list(map(lambda x: x.time, checkpoint['observation_traj'])))
obs_obs = torch.stack(list(map(lambda x: x.observation, checkpoint['observation_traj'])))
obs_valid = torch.where((obs_time >= t_min) * (obs_time <= t_max))[0]
obs_time = obs_time[obs_valid]
obs_obs = obs_obs[obs_valid]
obs_time -= t_min

# plotting
fig = plt.figure()
ax = plt.subplot(311)
ax.set_yticks([0., 3., 6., 9.])
ax.xaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(left=.125, bottom=0.1, right=0.9, top=0.9, wspace=.5, hspace=.15)
cnt = plt.contourf(belief_time, torch.arange(pomdp.n_packages), belief_pack.T, 40, cmap='viridis', vmin=-.07)
for c in cnt.collections:
    c.set_edgecolor("face")
plt.step(states_time, states_pack, color='k')
ax.set_ylabel('Packages')

ax = plt.subplot(312)
for i in range(pomdp.n_transition_states):
    offset = [0, 1, 2]
    cnt = plt.contourf(belief_time, torch.arange(2) + offset[i], belief_ts.T[i:i+1].repeat(2, 1), 40, vmin=-.07)
    for c in cnt.collections:
        c.set_edgecolor("face")
plt.scatter(obs_time, obs_obs + .5, marker='x', color='rebeccapurple')
plt.step(states_time, states_td + .5, color='k')
plt.yticks([.5, 1.5, 2.5])
ax.set_ylabel('Transmission')
ax.set_yticklabels(['i', 't', 'c'])
ax.xaxis.set_major_locator(plt.NullLocator())

belief_pack[0, 0] = 1.0

ax = plt.subplot(313)
ax.set_yticks([0., 4., 8.])
cnt = plt.contourf(belief_time, torch.arange(pomdp.n_actions), adv_vals.T, 40)
for c in cnt.collections:
    c.set_edgecolor("face")
plt.step(belief_time, belief_actions, color='k')
if colorbar:
    heatmap = plt.pcolor(belief_pack)
    cb = plt.colorbar(heatmap, aspect=colorbar_aspect)
    cb.set_label('Advantage values')
    tick_locator = ticker.FixedLocator([0.0, 0.5, 1.0])
    cb.locator = tick_locator
    cb.update_ticks()
ax.set_ylabel('Action')
ax.set_xlabel('Time')
plt.savefig('aloha_' + str(int(colorbar)))
plt.show()

# extra plot for heatmap
plt.figure()

heatmap = plt.pcolor(belief_pack)
plt.gca().set_xlabel('time')

cb = plt.colorbar(heatmap, aspect=25)
tick_locator = ticker.FixedLocator([0.0, 0.5, 1.0])
cb.locator = tick_locator
cb.update_ticks()
plt.savefig('aloha_legend')
plt.show()
