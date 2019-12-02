![DEN Logo](https://github.com/johli/genesis/blob/master/den_github_logo.png?raw=true)

# Deep Exploration in Sequence Generation
Code for training deep generative DNA sequence models in Keras. Implements deep convolutional generative networks which are trained to maximize objective biomolecule targets given a sequence-predictive model.

The generators are trained to jointly maximize sequence diversity and fitness. The training framework is described in a MLCB 2019* conference paper, "[Deep exploration networks for rapid engineering of functional DNA sequences](https://github.com/johli/genesis/blob/master/mlcb_exploration_nets.pdf?raw=true)".

*1st Conference on Machine Learning in Computational Biology, (MLCB 2019), Vancouver, Canada.

#### Features
- Implements a Sequence PWM Generator as a Keras Model, outputting PWMs, Logits, or random discrete samples from the PWM. These representations can be fed into any downstream Keras model for reinforcement learning.
- Implements a Predictor Keras Model wrapper, allowing easy loading of pre-trained sequence models and connecting them to the upstream PWM generator.
- Implements a Loss model with various useful cost and objectives, including regularizing PWM losses (e.g., soft sequence constraints, PWM entropy costs, etc.)
- Includes visualization code for plotting PWMs and cost functions during optimization (as Keras Callbacks).

### Installation
Install by cloning or forking the [github repository](https://github.com/johli/genesis.git):
```sh
git clone https://github.com/johli/genesis.git
cd genesis
python setup.py install
```

#### Required packages
- Tensorflow >= 1.13.1
- Keras >= 2.2.4
- Scipy >= 1.2.1
- Numpy >= 1.16.2
- Isolearn >= 0.2.0 ([github](https://github.com/johli/isolearn.git))

### Analysis Notebooks (Alternative Polyadenylation)
The following Jupyter Notebooks contain all of the training code & analyses that were part of the paper.

[Notebook 1a: Title (Extra title)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/example.ipynb)<br/>
