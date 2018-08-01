import math
import torch
from lib import manual_functions
from tqdm import tqdm
import cocob

################ 1-D Function of Choice  ##################
# (Any 1d is easily implementable in manual_functions.py) #
function = lambda : manual_functions.FunctionCocobExample()
initial_point = 0.0     # x_0
optimal_point = 1.0    # x*
function_name = 'f(x) = |x-1|'
###########################################################

x_t = torch.tensor(initial_point, requires_grad=True)
x_opt = torch.tensor(optimal_point, requires_grad=False)
x_avg = torch.tensor(0.0, requires_grad=False)

name_optimizers = [
        ('cocob_backprop', cocob.CocobBackprop([x_t])),
        ('cocob_ons', cocob.CocobOns([x_t])),
        ('adam', torch.optim.Adam([x_t])),
        ('adagrad', torch.optim.Adagrad([x_t]))
    ]

suboptimality_dict = {}
iterations_count   = 5000
for opt_name, optimizer in name_optimizers:
    print('Using Optimizer {}'.format(opt_name))
    x_t.data   = torch.tensor(initial_point)
    y_opt = function()(x_opt.unsqueeze(0))

    suboptimalities = []
    iterates = []
    for t in tqdm(range(iterations_count)):

        current = x_t.detach().item()
        iterates.append(current)
        average_iterate = sum(iterates) / float(len(iterates))

        y_t = function()(x_t)

        x_avg.data = torch.tensor(average_iterate)
        y_at_x_avg = function()(x_avg)
        suboptimality = math.log10(y_at_x_avg.detach().item() - y_opt.detach().item())
        suboptimalities.append(suboptimality)

        optimizer.zero_grad()
        y_t.backward()
        optimizer.step()

    suboptimality_dict[opt_name] = suboptimalities


### This section plots matplotlib plots                       ####
### of log-suboptimalities and saves in appropriate directory ####

import matplotlib.pyplot as plt
for opt_name, optimizer in name_optimizers:
    plt.plot(range(len(suboptimality_dict[opt_name])), suboptimality_dict[opt_name], '-', label=opt_name)

plt.legend(loc='upper right')
plt.xlabel('iterates')
plt.ylabel('log-suboptimalities')
plt.title( 'optimize: {}'.format(function_name) )
plt.grid()
# plt.show()
plt.savefig('log-suboptimalities-1d-function.png')

############################################
