import torch
import matplotlib.pyplot as plt
import sys
import pylab

sys.path.append('..')
from packages.models.components.advantages import AdvantageNet
from packages.models.components.values import ValueNet
from examples.models.tiger import TigerProblem

# settings
cp_file_col = 'model_col_tiger'
cp_file_adv = 'model_adv_tiger'

# load checkpoint
cp_col = torch.load("checkpoints/" + cp_file_col, map_location=torch.device('cpu'))
cp_adv = torch.load("checkpoints/" + cp_file_adv, map_location=torch.device('cpu'))

# create pomdp
pomdp = TigerProblem()
ns = pomdp.SSpace.nElements
na = pomdp.ASpace.nElements

value_net_col = ValueNet(ns)
value_net_col.load_state_dict(cp_col['value_net_state_dict'])
value_net_adv = ValueNet(ns)
value_net_adv.load_state_dict(cp_adv['value_net_state_dict'])
advantage_fun_col = AdvantageNet(ns, na)
advantage_fun_col.load_state_dict(cp_col['advantage_net_state_dict'])
advantage_fun_adv = AdvantageNet(ns, na)
advantage_fun_adv.load_state_dict(cp_adv['advantage_net_state_dict'])

# create value and advantage plots
with torch.no_grad():
    num_states = 100
    s1 = torch.linspace(0, 1, num_states)
    s0 = 1 - s1
    beliefs = torch.stack((s0, s1)).T

    vv_col = value_net_col(beliefs, compute_grad=False)[:, 0].detach()
    av_col = advantage_fun_col(beliefs).detach()

    vv_adv = value_net_adv(beliefs, compute_grad=False)[:, 0].detach()
    av_adv = advantage_fun_adv(beliefs).detach()

    fig = plt.figure()

    ax = plt.subplot(221)
    plt.subplots_adjust(left=0., bottom=0., right=1., top=1., wspace=.3, hspace=0.15)
    plt.plot(s1, vv_col)

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    ax = plt.subplot(222)
    linestyles = ['solid', 'dashed', 'dotted']
    for a in range(na):
        plt.plot(s1, av_col[:, a], label=pomdp.a_labels[a], linestyle=linestyles[a])
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    ax1 = plt.subplot(223)
    plt.plot(s1, vv_adv)
    plt.xlabel("Belief tiger right")
    plt.ylabel('Value function')
    ax2 = ax1.twinx()
    ax2.yaxis.set_major_locator(plt.NullLocator())
    beliefs = torch.stack(list(map(lambda x: x.belief, cp_adv['data'])))

    plt.hist(beliefs[:, 0], 50, color='darkorange')

    ax_advadv = plt.subplot(224)
    lines =[]
    for a in range(na):
        lines += [plt.plot(s1, av_adv[:, a], label=pomdp.a_labels[a], linestyle=linestyles[a])]
    plt.ylabel('Advantage function')
    plt.xlabel("Belief tiger right")

    plt.savefig('tiger6')

    legend_fig = pylab.figure()
    legend = pylab.figlegend(*ax_advadv.get_legend_handles_labels(), loc='center', ncol=3)
    legend_fig.canvas.draw()
    legend_fig.savefig('tiger_legend',
        bbox_inches=legend.get_window_extent().transformed(legend_fig.dpi_scale_trans.inverted()))
    plt.show()
