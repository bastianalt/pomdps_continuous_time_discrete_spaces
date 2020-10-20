import logging
import sys
from datetime import datetime
from torch.optim import Adam
import torch.optim

sys.path.append('..')
from packages.models.pomdps.learners.collocation import CollocationLearner
from examples.models.tiger import TigerProblem

name = "tiger_col"

# set seed
torch.manual_seed(0)

# get time stamp
timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

device_str = 'cpu'
checkpoint_interval = -1
write_logs_to_file = False

# only write logs and save checkpoints if cuda available = running seriously
if torch.cuda.is_available():
    device_str = 'cuda:0'
    write_logs_to_file = True
    checkpoint_interval = 1000

# logging settings
format = "%(asctime)s %(levelname)s %(name)s %(message)s"
logging.basicConfig(filename="log/tiger_col_{}.txt".format(timestamp), level=logging.INFO, format=format)
logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# init pomdp
pomdp = TigerProblem()

belief_prior = torch.ones(pomdp.SSpace.nElements) * .1

opt_constructor = Adam

# init learner that uses the collocation method
learner = CollocationLearner(pomdp, name="{}_{}".format(name, timestamp), num_episodes=10000,
                             checkpoint_interval=checkpoint_interval,
                             batch_size=256, num_iter_v=10, num_iter_a=10, belief_prior=belief_prior,
                             opt_constructor=opt_constructor, discount_decay=500)

logger.info("Apply collocation learning to tiger problem.")
learner.learn()
learner.learn_advantage()
