# Coin Betting Optimization Algorithms in Pytorch

Original Cocob implementation in tensorflow can be found [here](https://github.com/bremen79/cocob). This is a minified repository from [this](https://github.com/HarshTrivedi/coin-betting-optimizers-pytorch/blob/master/slides.pdf) work, which is unpublished yet.

It contains two published coin betting optimization algorithms:

1. Cocob Backprop: [Training Deep Networks without Learning Rates Through Coin Betting](https://arxiv.org/pdf/1705.07795.pdf)
2. Cocob through Ons: [
Black-Box Reductions for Parameter-free Online Learning in Banach Spaces](https://arxiv.org/pdf/1705.07795.pdf)

both of which do not require any learning rates and yet have optimal convergence gauarantees for non-smooth convex functions. **Cocob-Ons** is an experimental variation from the paper and is WIP, do not use it yet.

To understand betting game and the duality between coin betting and convex optimization please check following: [Slides](http://francesco.orabona.com/papers/slides_cocob.pdf), [Video](https://www.youtube.com/watch?v=61o-TMEcDMM)

### Code overview:

1. `cocob.py` has pytorch implementations for coin betting based optimization.
2. Mnist and Cifar can be trained and `mnist_optimize.py` and `cifar_optimize.py`. It will save the analysable log-losses after training finishes.
3. Use `1d_function_optimize.py` to run cocob on any 1d function and check log-suboptimalities. Default function is `f(x)=|x-10|`.
4. Run `1d_function_play.py` to see cocob live in action on a 1d function. You can see the internal states of the betting game on matloblib interactively as the optimization goes on!

### Some plots

<img src="https://raw.githubusercontent.com/HarshTrivedi/coin-betting-optimizers-pytorch/master/log-losses-mnist.png" width="300">  | <img src="https://raw.githubusercontent.com/HarshTrivedi/coin-betting-optimizers-pytorch/master/log-losses-cifar.png" width="300"> | <img src="https://raw.githubusercontent.com/HarshTrivedi/coin-betting-optimizers-pytorch/master/images/1d_play.gif" width="300">
:------------------------------------------:|:-----------------------------------------:|:-------------------------------:


Please cite the following papers if you use this in your work.

```
@inproceedings{orabona2017training,
  title={Training Deep Networks without Learning Rates Through Coin Betting},
  author={Orabona, Francesco and Tommasi, Tatiana},
  booktitle={Advances in Neural Information Processing Systems},
  pages={2160--2170},
  year={2017}
}
```