![DEN Logo](https://github.com/johli/genesis/blob/master/den_github_logo.png?raw=true)

# Deep Exploration in Sequence Generation
Code for training deep generative models for DNA sequences in Keras. Implements generative neural networks which are optimized w.r.t. a downstream deep learning predictor to maximize predicted fitness.

The generators are trained to jointly maximize sequence diversity and predicted fitness. The framework is described in a MLCB 2019* conference paper, "[Deep exploration networks for rapid engineering of functional DNA sequences](https://github.com/johli/genesis/blob/master/mlcb_exploration_nets.pdf?raw=true)".

*1st Conference on Machine Learning in Computational Biology, (MLCB 2019), Vancouver, Canada.

#### Highlights
- Deep generative neural networks for DNA, RNA & Protein sequences.
- Train the generative model to maximize both sequence diversity and functional fitness (Deep Exploration).
- Fitness is evaluated by a user-supplied sequence-predictive model and a cost function.

#### Features
- Implements deep convolutional- and residual generative neural networks.
- Supports vanilla, class-conditional and inverse-regression generators.
- Generators support 1-hot sequence sampling, enabling end-to-end training via straight-through gradients.

### Installation
Install by cloning or forking the [github repository](https://github.com/johli/genesis.git):
```sh
git clone https://github.com/johli/genesis.git
cd genesis
python setup.py install
```

#### Required Packages
- Tensorflow >= 1.13.1
- Keras >= 2.2.4
- Scipy >= 1.2.1
- Numpy >= 1.16.2
- Isolearn >= 0.2.0 ([github](https://github.com/johli/isolearn.git))

#### Saved Models
To aid reproducibility, we provide access to all trained models via the google drive link below:

[Model Repository](https://drive.google.com/open?id=11_wlrjrb0ee_UyaT9agMigpIsmGlFZzU)

**apa/saved_models/apa_models.tar.gz**
> Deep Exploration Networks (DENs) for generating sequences with target APA isoform proportions and cleavage.

**splicing/saved_models/splicing_models.tar.gz**
> DENs for generating sequences with target (differential) 5' splice donor usage proportions.

### Training & Analysis Notebooks 
The following jupyter notebooks contain all of the training code & analyses that were part of the paper.
We also include additional analyses and models which users may find useful.

#### Alternative Polyadenylation
Training and evaluation of Exploration networks for engineering Alternative Polyadenylation signals.

[Notebook 1a: Engineering APA Isoforms (ALIEN1 Library)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_target_isoform_genesis_simple.ipynb)<br/>
[Notebook 1b: Engineering APA Isoforms (ALIEN2 Library)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_target_isoform_genesis_doubledope.ipynb)<br/>
[Notebook 1c: Engineering APA Isoforms (TOMM5 Library)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_target_isoform_genesis_tomm5.ipynb)<br/>
[Notebook 2a: Significance of Similarity Loss (ALIEN1 Library)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_max_isoform_genesis_simple_eval_repelweight.ipynb)<br/>
[Notebook 2b: Significance of Similarity Loss (ALIEN2 Library)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_max_isoform_genesis_doubledope_eval_repelweight.ipynb)<br/>
[Notebook 3: PWMs or Discrete One-Hot Samples? (ALIEN1 Library)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_target_isoform_genesis_eval_sampling_modes.ipynb)<br/>
[Notebook 4: Engineering Cleavage Position (ALIEN1 Library)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_cleavage_genesis.ipynb)<br/>
[Notebook 5: Inverse APA Isoform Regression (ALIEN1 Library)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_isoform_regression_genesis.ipynb)<br/>
<br/>
[Bonus 1: Human pA Sequence GAN (APADB)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/gan/train_sequence_sngan_new_resnet_multisample_batchnorm_normal_apadb.ipynb)<br/>
[Bonus 2: Human pA Sequence VAE (APADB)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/vae/train_sequence_vae_all_code_partials_apadb_new_resnet_len_160.ipynb)<br/>
[Bonus 3: Max APA Isoform GANception (ALIEN1)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/ganception/apa_max_isoform_genesis_wgan_simple_singlesample_descent_normal_latent_similarity.ipynb)<br/>

#### Alternative Splicing
Training and evaluation of Exploration networks for engineering (differential) Alternative Splicing.

[Notebook 1: Engineering Splicing Isoforms (HEK)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/splicing/splicing_target_isoform_genesis_hek.ipynb)<br/>
[Notebook 2: Engineering De-Novo Splice Sites (HEK)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/splicing/splicing_cleavage_genesis_hek.ipynb)<br/>
[Notebook 3a: Differential - CHO vs. MCF7 (CNN Predictor)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/splicing/splicing_differential_genesis_cnn_cho_vs_mcf7.ipynb)<br/>
[Notebook 3b: Differential - CHO vs. MCF7 (Hexamer Regressor)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/splicing/splicing_differential_genesis_logistic_regression_both_regions_cho_vs_mcf7.ipynb)<br/>
[Notebook 3c: Differential - CHO vs. MCF7 (Both Predictors)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/splicing/splicing_differential_genesis_cnn_and_logistic_regression_both_regions_cho_vs_mcf7.ipynb)<br/>

### DEN Training GIFs
The following GIFs illustrate how the Deep Exploration Networks converge on generating maximally fit functional sequences while retaining sequence diversity. Throughout training, we track a set of randomly chosen input seeds and animate the corresponding generated sequences (with their fitness costs).

**WARNING:** The following GIFs contain flickering pixels/colors. Do not look at them if you are sensitive to such images.

#### Alternative Polyadenylation
The following GIF depicts a generator trained to produce maximally strong polyadenylation signals.

![APA Max Isoform GIF](https://github.com/johli/genesis/blob/master/analysis/apa/genesis_max_isoform_simple_fixed_sequences_with_seeds_and_pwms_all_small_32_colors.gif?raw=true)

The next GIF illustrates a class-conditional generator trained to produce polya sequences with target cleavage positions.

![APA Max Cleavage GIF](https://github.com/johli/genesis/blob/master/analysis/apa/genesis_max_cleavage_simple_fixed_sequences_with_seeds_and_pwms_all_small_32_colors.gif?raw=true)

#### Alternative Splicing
This GIF depicts a generator trained to maximize splicing at 5 distinct splice junctions.

![De-Novo Splicing GIF](https://github.com/johli/genesis/blob/master/analysis/splicing/genesis_cleavage_multiclass_fixed_sequences_with_seeds_and_pwms_all_speedup_small.gif?raw=true)
