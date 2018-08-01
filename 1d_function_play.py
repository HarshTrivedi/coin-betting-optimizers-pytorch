import math
import argparse

import torch
from torch.autograd import Variable

from lib import manual_functions, draw
import cocob


parser = argparse.ArgumentParser(description='See any optimizer in action on any 1d function .'
                                 'Default: observed internals states of CocobBackprop on '
                                 'f(x) = |x-1| as it optimizes.')
parser.add_argument('--play', type=bool,
                         help='Pass --play flag to start optimization non-interactively',
                         default=False)
args = parser.parse_args()

################ 1-D Function of Choice  ###########################
# (Any 1d func is easily implementable in lib/manual_functions.py) #
function = lambda : manual_functions.FunctionCocobExample()
initial_point = 0.0     # x_0
function_name = 'f(x) = |x-1|'
plot_limits   = [0, 2]
####################################################################

x_t = torch.tensor(initial_point, requires_grad=True)

optimizer = cocob.CocobBackprop([x_t], alpha=100.0)
# optimizer = cocob.CocobOns([x_t]) Experimental version, do not use yet.
# optimizer = torch.optim.Adam([x_t])  # Adam with default lr extremely slow (try 0.1)
# optimizer = torch.optim.Adagrad([x_t]) # Adagrad with default lr extremely slow (try 0.1)

bet_fraction = 0.0
iterates, additionals = [], []
iterates.append(x_t.detach().item())
for t in range(30):
    y = function()(x_t)

    optimizer.zero_grad()
    y.backward()
    optimizer.step()

    iterates.append(x_t.detach().item())

    opt_state = optimizer.state[optimizer.param_groups[0]['params'][0]]
    # opt_state is the optimizer state dict, internal optimizer state can be logged.
    # eg. print float(opt_state['wealth'].detach().item())

    attributes = ["wealth", "reward", "bet"]
    additional = {}
    for attribute in attributes:
        if opt_state.get(attribute, None):
            additional[attribute] = opt_state[attribute].detach().item()
    if opt_state.get("bet_fraction", None):
        additional["x_t"] = x_t.detach().item()
        additional["prev_bet_fraction"] = bet_fraction
        additional["current_bet_fraction"] = opt_state["bet_fraction"].detach().item()
        bet_fraction = additional["current_bet_fraction"]

    additionals.append(additional)

optimizer_name = optimizer.__class__.__name__
func = lambda x: float(function().forward(torch.tensor(x)).detach().item())

# Pass play = False, to turn-off autoplay and keep it 'press Enter' based
draw.plot_iterates_on(iterates,
                      plot_limits,
                      func,
                      function_name,
                      optimizer_name,
                      additionals=additionals,
                      play=True)
