import logging
import sys
import torch

sys.path.append('..')
from packages.models.pomdps.learners.advantage_updating import AdvantageUpdateLearner
from examples.models.gridworld import Gridworld
from datetime import datetime

# get time stamp
timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

name = 'gw_col'
device_str = 'cpu'
checkpoint_interval = -1
write_logs_to_file = False

# only write logs and save checkpoints if cuda available = running seriously
if torch.cuda.is_available():
    device_str = 'cuda:0'
    write_logs_to_file = True
    checkpoint_interval = 100

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

sample_initial_belief = torch.ones(pomdp.SSpace.nElements) * .1

ode_options = {'method': 'rk4',
               'options': {'step_size': .01}}

# Init learner that uses the collocation method
learner = AdvantageUpdateLearner(pomdp,name="{}_{}".format(name, timestamp), checkpoint_interval=checkpoint_interval,
                                 episode_length=5, reset_episode=True, num_episode_samples=100, batch_size=256,
                                 num_optim_iter=20, num_episodes=1000, memory_capacity=50000, ode_options=ode_options,
                                 sample_initial_belief=sample_initial_belief)

logger.info("Apply advantage updating to slotted aloha problem.")
learner.learn()
