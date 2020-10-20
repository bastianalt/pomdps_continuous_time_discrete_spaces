import logging
import sys
import torch
import torch.distributions
from datetime import datetime
from torch.optim import Adam

sys.path.append('..')
from packages.models.pomdps.learners.collocation import CollocationLearner
from examples.models.gridworld import Gridworld

# get time stamp
timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

torch.manual_seed(0)

name = 'gw_col'
device_str = 'cpu'
checkpoint_interval = -1
write_logs_to_file = False

# only write logs and save checkpoints if cuda available = running seriously
if torch.cuda.is_available():
    device_str = 'cuda:0'
    write_logs_to_file = True
    checkpoint_interval = 1000

# logging settings
format = "%(asctime)s %(levelname)s %(message)s"  # %(name)s
if write_logs_to_file:
    log_file = 'log/{}_{}.txt'.format(name, timestamp)
    logging.basicConfig(filename=log_file, level=logging.INFO, format=format)
    logger = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
else:
    logging.basicConfig(level=logging.INFO, format=format)
    logger = logging.getLogger()

device = torch.device(device_str)

# init pomdp
pomdp = Gridworld().to(device)


# sample beliefs from gaussians
def belief_sample_fn(num):
    # sample one hot states on grid
    states = torch.distributions.Categorical(torch.ones(pomdp.n_states)).sample([num])
    states_grid_pos = pomdp.grid_position(states)
    states_grid = torch.zeros(num, pomdp.height, pomdp.width)
    states_grid[torch.arange(num), states_grid_pos[:, 0], states_grid_pos[:, 1]] = 1

    # sample variance for Gaussian from inverse gamma
    sigmas = 1 / torch.distributions.gamma.Gamma(2, 1).sample([num])
    sigmas[0] = 1

    # create gaussian maps
    filter_half_width = 2
    gf = pomdp.generate_Log_Normal_2d(2 * filter_half_width + 1).exp()

    # convolute observation patterns with map
    gmap = torch.conv2d(input=states_grid[:, None], weight=gf[:, None], padding=[filter_half_width, filter_half_width])[:, 0]

    # sample
    # create filters and convolve maps please please
    gmap_log = gmap.log()
    gmap_sigma = (gmap_log / (2 * sigmas[:, None, None] ** 2)).exp()
    gmap_sigma /= torch.sum(gmap_sigma, axis=[1, 2], keepdim=True)
    batch_beliefs = gmap_sigma.reshape(gmap_sigma.shape[0], -1)[:, ~pomdp.walls.flatten()]

    return batch_beliefs

opt_constructor = Adam
# Init learner that uses the collocation method
learner = CollocationLearner(pomdp, name="{}_{}".format(name, timestamp), num_episodes=10000,
                             checkpoint_interval=checkpoint_interval,
                             batch_size=256, num_iter_v=25, num_iter_a=25, belief_sample_fn=belief_sample_fn,
                             opt_constructor=opt_constructor, discount_decay=500)

logger.info("Apply collocation learning to grid world problem.")
learner.learn()
learner.learn_advantage()
