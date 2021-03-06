import logging
import sys
import torch

sys.path.append('..')
from packages.models.pomdps.learners.advantage_updating import AdvantageUpdateLearner
from examples.models.slotted_aloha import SlottedAloha
from datetime import datetime


seed = 0
# set seed
torch.manual_seed(seed)

# get time stamp
timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

name = 'aloha_au'
device_str = 'cpu'
checkpoint_interval = -1
write_logs_to_file = False

# only write logs and save checkpoints if cuda available = running seriously
if torch.cuda.is_available():
    device_str = 'cuda:0'
    write_logs_to_file = True
    checkpoint_interval = 100
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

# Example for a simple two state problem
pomdp = SlottedAloha().to(device)

belief_prior = torch.ones(pomdp.SSpace.nElements) * .1

ode_options = {'method': 'rk4',
               'options': {'step_size': .01}}

# Init learner that uses the collocation method
learner = AdvantageUpdateLearner(pomdp, name=name+"_{}".format(timestamp), checkpoint_interval=checkpoint_interval,
                                 episode_length=20, num_episode_samples=1000, batch_size=256, num_optim_iter=20,
                                 num_episodes=1000, memory_capacity=50000, ode_options=ode_options,
                                 sample_initial_belief=belief_prior)
logger.info("Apply advantage updating to slotted aloha problem.")
learner.learn()
