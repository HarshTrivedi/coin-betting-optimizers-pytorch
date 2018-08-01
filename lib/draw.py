import matplotlib.pyplot as plt
import torch
import numpy as np
import time


def plot_mark(inputs, function, mark, color, name):

    min_difference = min([ np.abs(input-mark) for input in inputs ])
    mark_index =  [ i for i, input in enumerate(inputs) if min_difference == np.abs(input-mark) ][0]
    _in  = inputs[mark_index]
    _out = function(_in)
    plt.plot([_in],[_out], ls="", marker="o", color=color, label=name)


def plot_iterates_on( iterates, limits, function, function_name, optimizer_name, additionals=None, play=False):
    lower_limit, upper_limit = limits
    inputs  = np.linspace(lower_limit,upper_limit, 1001)
    outputs = [function(x) for x in inputs]

    consecutive_iterates = [[iterates[0], iterates[0]] ]+ [iterates[i:i+2] for i in range(len(iterates)-2+1)]

    plt.ion()
    plt.show()

    _sum = 0.0; average = 0.0
    for index, (prev, current) in enumerate(consecutive_iterates):

        _sum += current
        average = _sum / float(index+1)
        # exp_average = (0.95*average) + (0.05)*(current)

        plt.plot(inputs, outputs , label=function_name)
        plot_mark( inputs, function, prev, 'blue', 'previous')
        plot_mark( inputs, function, current, 'red', 'current')
        plot_mark( inputs, function, average, 'black', 'average')
        # plot_mark( inputs, function, exp_average, 'brown')

        if additionals:
            sorted_keys = sorted(additionals[index].keys())
            for i, key in enumerate(sorted_keys):
                note = '{} : {}'.format(key, round(additionals[index][key], 3))
                x_pos = 0.15
                y_pos = 0.15 + (0.03*i)
                plt.figtext(x_pos, y_pos, note, fontsize=9)

        plt.title( '{} Iterate: {}'.format(optimizer_name, index+1) )
        plt.legend(loc='lower right')
        # plt.savefig('images/1d_play_{}.png'.format(index))
        if play:
            plt.pause(0.5)
        else:
            input("Press Enter to continue. Iterate: {}".format(index+1))
        plt.clf()
