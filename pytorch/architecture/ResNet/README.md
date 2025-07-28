# Deep Residual Learning


## Introduction


In deeper network , there is a problem of vanishing gradient , which hamper convergence from the beginning. When deeper network is able to start converging , a degradation problem has been exposed , with the network depth increasing , accuracy gets saturated and then degrads rapidly . Such degradion is not caused by overfitting and adding
more layers to a suitable deep model leads to higher training error .


[Resnet Visualization](https://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006)

**H(X) = F(X) + X**

Here ,

- **H(X)** : The original desired mapping .
- **F(X)** : The residual mapping the network learns .
- **X**    : The input to the residual block , added back to the output .


## Problem With Deep Learning


**Problem 1** : Adding more layers to a suitably deep model leads to higher training error .
the process is described in a rather inaccessible way, but the concept is actually very simple: start with a 20-layer neural network that is trained well, and add another 36 layers that do nothing at all (for instance, they could be linear layers). The result will be a 56-layer network that does exactly the same thing as the 20-layer network, proving that there are always deep networks that should be at least as good as any shallow network. But for some reason, stochastic gradient descent (SGD) does not seem able to find them. This degradation problem wasn't due to overfitting but rather the difficulty of optimizing very deep networks. This is referred to as the vanishing gradient problem.


**Problem 2** :

When training neural networks with gradient-based learning methods and backpropagation, each of the neural network's weights receives an update proportional to the gradient of the error function with respect to the current weight in each iteration of training. The problem is that, in some cases, the gradient will be vanishingly small, effectively preventing the weight from changing its value. In the worst case, this may completely stop the neural network from further training.

## ResNets

ResNets introduced  a novel architecture to combat the vanishing gradient problem . The core idea of of ResNets is introducing a so called **identity shortcut connection** that skips one or more layers .  The idea of shortcut connections(skip connections) allows the model to skip layers and help to mitigate the problem of vanishing gradients. These connections allow the gradient to be directly backpropagated to earlier layers , which makes the network easier to optimize . The key  thing is that extra 36 layers , and the have parameters so they are trainable . So , the network can skipt those 36 layers and thus the new is  at least  as good as 20 layers . Those 36 layers can then learn which can make them useful .
