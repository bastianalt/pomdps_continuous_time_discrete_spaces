import logging
import sys
import torch

sys.path.append('..')
from packages.models.pomdps.learners.collocation import CollocationLearner
from examples.models.slotted_aloha import SlottedAloha
from datetime import datetime
from torch.optim import Adam

# settings
name = 'aloha_col'
seed = 0

# get time stamp
timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

# set seed
torch.manual_seed(seed)

write_logs_to_file = False
device_str = 'cpu'
checkpoint_interval = -1

# only write logs and save checkpoints if cuda available = running seriously
if torch.cuda.is_available():
    device_str = 'cuda:0'
    write_logs_to_file = True
    checkpoint_interval = 500

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
pomdp = SlottedAloha().to(device)

belief_prior = torch.ones(pomdp.n_states) * .1

opt_constructor = Adam
# Init learner that uses the collocation method
learner = CollocationLearner(pomdp, name="{}_{}".format(name, timestamp), num_episodes=10000,
                             checkpoint_interval=checkpoint_interval,
                             batch_size=256, num_iter_v=25, num_iter_a=25, belief_prior=belief_prior,
                             discount_decay=500, opt_constructor=opt_constructor)

logger.info("Apply collocation learning to slotted aloha problem.")
learner.learn()
learner.learn_advantage()
