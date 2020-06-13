![DEN Logo](https://github.com/johli/genesis/blob/master/den_github_logo.png?raw=true)

# Deep Exploration in Sequence Generation
Code for training deep generative models of DNA, RNA and protein sequences in Keras. Implements a class of activation-maximizing generative neural networks (Deep Exploration Networks, or DENs) which are optimized with respect to a downstream deep learning predictor. DENs explicitly maximize sequence diversity by sampling *two* independent patterns at each forward pass of backpropagation and imposing a similarity penalty on those samples. DENs optionally maintain the confidence in generated sequences during backpropagation by incorporating variational autoencoders (VAEs) to estimate their likelihood. Likelihood is approximated by importance sampling and gradients are backpropagated from the VAE to the DEN using straight-through (ST) gradients.

The generators are trained in a closed loop of backpropagation to jointly maximize sequence diversity and predicted fitness. The framework is described in a MLCB 2019* conference paper, "[Deep exploration networks for rapid engineering of functional DNA sequences](https://github.com/johli/genesis/blob/master/mlcb_exploration_nets.pdf?raw=true)".

*1st Conference on Machine Learning in Computational Biology, (MLCB 2019), Vancouver, Canada.

#### Highlights
- Deep generative neural networks for DNA, RNA & protein sequences.
- Train the generator to maximize both diversity and fitness.
- Fitness is evaluated by a user-supplied sequence-predictive model and cost function.

#### Features
- Implements deep convolutional- and residual generative neural networks.
- Supports vanilla, class-conditional and inverse-regression generators.
- Generators support one-hot sampling, enabling end-to-end training via straight-through gradients.
- Optionally maintain likelihood in the generated sequences during training by using importance sampling of a pre-trained variational autoencoder.

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
We also include additional analyses and models which users may find useful. We used the following fitness predictors in our analyses: APARENT [(Bogard et. al., 2019)](https://doi.org/10.1016/j.cell.2019.04.046), DragoNN [(Kundaje Lab)](https://github.com/kundajelab/dragonn), MPRA-DragoNN [(Movva et. al., 2019)](https://doi.org/10.1371/journal.pone.0218073) and our own [(Cell line-specific splicing predictor)](https://github.com/johli/splirent). For some of the benchmarks, we use the Feedback-GAN code ([Gupta et. al., 2019](https://doi.org/10.1038/s42256-019-0017-4); [Github](https://github.com/av1659/fbgan)) and CbAS code ([Brookes et. al., 2019](https://arxiv.org/abs/1901.10060); [Github](https://github.com/dhbrookes/CbAS)).

#### Alternative Polyadenylation
Training and evaluation of Exploration networks for engineering Alternative Polyadenylation signals.

Notebook 0a: APA VAE Training Script (beta = 0.15) ([Not Annealed](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/vae/train_sequence_vae_all_code_partials_simple_old_resnet_len_256_low_kl.ipynb))<br/>
Notebook 0b: APA VAE Training Script (beta = 0.65) ([Not Annealed](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/vae/train_sequence_vae_all_code_partials_simple_old_resnet_len_256_medium_kl.ipynb) | [Annealed](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/vae/train_sequence_vae_all_code_partials_simple_old_resnet_len_256_medium_kl_annealed.ipynb))<br/>
Notebook 0c: APA VAE Training Script (beta = 0.85) ([Annealed](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/vae/train_sequence_vae_all_code_partials_simple_old_resnet_len_256_high_kl_annealed.ipynb))<br/>
(*Note*: The non-annealed version with beta = 0.65 is used in Notebook 6c below.)<br/>

[Notebook 1a: Engineering APA Isoforms (ALIEN1 Library)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_target_isoform_genesis_simple.ipynb)<br/>
[Notebook 1b: Engineering APA Isoforms (ALIEN2 Library)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_target_isoform_genesis_doubledope.ipynb)<br/>
[Notebook 1c: Engineering APA Isoforms (TOMM5 Library)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_target_isoform_genesis_tomm5.ipynb)<br/>
[Notebook 2a: Significance of Diversity Cost (ALIEN1 Library)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_max_isoform_genesis_simple_eval_repelweight.ipynb)<br/>
[Notebook 2b: Significance of Diversity Cost (ALIEN2 Library)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_max_isoform_genesis_doubledope_eval_repelweight.ipynb)<br/>
[Notebook 3a: PWMs or Straight-Through Approximation?](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_max_isoform_genesis_eval_sampling_modes_no_entropy_with_edit_distances.ipynb)<br/>
[Notebook 3b: PWMs or Straight-Through Approximation? (Entropy Penalty)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_max_isoform_genesis_eval_sampling_modes_with_edit_distances.ipynb)<br/>
[Notebook 4: Engineering Cleavage Position (ALIEN1 Library)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_cleavage_genesis.ipynb)<br/>
[Notebook 5: Inverse APA Isoform Regression (ALIEN1 Library)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_isoform_regression_genesis.ipynb)<br/>
[Notebook 6a: Maximal APA Isoform (Sequence Diversity)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_max_isoform_genesis_simple.ipynb) [(Earthmover)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_max_isoform_genesis_simple_earthmover.ipynb)<br/>
[Notebook 6b: Maximal APA Isoform (Latent Diversity)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_max_isoform_genesis_simple_predictor_latent_similarity_cosine.ipynb) [(Earthmover)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_max_isoform_genesis_simple_predictor_latent_similarity_cosine_earthmover.ipynb)<br/>
[Notebook 6c: Evaluate Diversity Costs (Sequence & Latent)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_max_isoform_genesis_simple_eval_models_kl_loss.ipynb)<br/>
[Notebook 7a: Benchmark Comparison](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_max_isoform_genesis_simple_eval_models.ipynb)<br/>
[Notebook 7b: Benchmark Comparison (Computational Cost)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_max_isoform_genesis_simple_eval_basinhopping_trajs.ipynb)<br/>
<br/>
[Extra 1: Native Human pA Sequence GAN (APADB)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/gan/train_sequence_sngan_new_resnet_multisample_batchnorm_normal_apadb.ipynb)<br/>
[Extra 2: Native Human pA Sequence VAE (APADB)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/vae/train_sequence_vae_all_code_partials_apadb_new_resnet_len_160_low_kl.ipynb)<br/>
[Extra 3: Max APA Isoform GANception (ALIEN1)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/ganception/apa_max_isoform_genesis_wgan_simple_singlesample_descent_normal_latent_similarity.ipynb)<br/>

#### Alternative Polyadenylation (Likelihood-bounded)
Addtional examples of engineering Alternative Polyadenylation signals using *Likelihood-bounded* Exploration networks. We combine importance sampling of a variational autoencoder (VAE) and straight-through approximation to propagate likelihood gradients to the generator.

[Notebook 0: Evaluate Variational Autoencoders (Not Annealed)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_max_isoform_genesis_doubledope_eval_vaes.ipynb)<br/>
[Notebook 0a: VAE Training Script (Weak APA - Not Annealed, beta = 1.0)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/vae/train_sequence_vae_all_code_partials_doubledope_old_resnet_len_128_weak_kl_factor_1.ipynb)<br/>
[Notebook 0b: VAE Training Script (Strong APA - Not Annealed, beta = 1.0)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/vae/train_sequence_vae_all_code_partials_doubledope_old_resnet_len_128_strong_kl_factor_1.ipynb)<br/>
(*Note*: These non-annealed versions with beta = 1.0 are used in Notebooks 1 and 2 below.)<br/>

[Notebook 0*: Evaluate Variational Autoencoders (Annealed)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_max_isoform_genesis_doubledope_eval_vaes.ipynb)<br/>
Notebook 0a*: VAE Training Script (Weak APA - Annealed) ([beta = 1.0](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/vae/train_sequence_vae_all_code_partials_doubledope_old_resnet_len_128_weak_kl_factor_1_annealed.ipynb) | [beta = 1.125](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/vae/train_sequence_vae_all_code_partials_doubledope_old_resnet_len_128_weak_kl_factor_1125_annealed.ipynb) | [beta = 1.25](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/vae/train_sequence_vae_all_code_partials_doubledope_old_resnet_len_128_weak_kl_factor_125_annealed.ipynb) | [beta = 1.5](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/vae/train_sequence_vae_all_code_partials_doubledope_old_resnet_len_128_weak_kl_factor_15_annealed.ipynb))<br/>
Notebook 0b*: VAE Training Script (Strong APA - Annealed) ([beta = 1.0](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/vae/train_sequence_vae_all_code_partials_doubledope_old_resnet_len_128_strong_kl_factor_1_annealed.ipynb) | [beta = 1.125](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/vae/train_sequence_vae_all_code_partials_doubledope_old_resnet_len_128_strong_kl_factor_1125_annealed.ipynb) | [beta = 1.25](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/vae/train_sequence_vae_all_code_partials_doubledope_old_resnet_len_128_strong_kl_factor_125_annealed.ipynb) | [beta = 1.5](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/vae/train_sequence_vae_all_code_partials_doubledope_old_resnet_len_128_strong_kl_factor_15_annealed.ipynb))<br/>
(*Note*: These versions are not used in downstream analyses, but included to show that beta-annealing does not significantly improve separability between Strong / Weak APA test sets. Compare to non-annealed VAEs with beta = 1.0 above.)<br/>

[Notebook 1: Evaluate Likelihood-bounded DENs (Weak VAE)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_max_isoform_genesis_doubledope_kl_vae_weak_lower_fitness_eval_generators.ipynb)<br/>
Notebook 1a/b/c/d: DEN Training Scripts (Weak VAE) ([Only Fitness](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_max_isoform_genesis_doubledope_kl_only_fitness.ipynb) | [Margin -2](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_max_isoform_genesis_doubledope_kl_vae_weak_vae_margin_neg_2_lower_fitness.ipynb) | [Margin 0](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_max_isoform_genesis_doubledope_kl_vae_weak_margin_0_lower_fitness.ipynb) | [Margin +2](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_max_isoform_genesis_doubledope_kl_vae_weak_vae_margin_pos_2_lower_fitness.ipynb))<br/>
[Notebook 2: Evaluate Likelihood-bounded DENs (Strong VAE)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_max_isoform_genesis_doubledope_kl_vae_strong_lower_fitness_eval_generators.ipynb)<br/>
Notebook 2a/b/c/d: DEN Training Scripts (Strong VAE) ([Only Fitness](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_max_isoform_genesis_doubledope_kl_only_fitness.ipynb) | [Margin -2](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_max_isoform_genesis_doubledope_kl_vae_strong_margin_neg_2_lower_fitness.ipynb) | [Margin 0](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_max_isoform_genesis_doubledope_kl_vae_strong_margin_0_lower_fitness.ipynb) | [Margin +2](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/apa/apa_max_isoform_genesis_doubledope_kl_vae_strong_vae_margin_pos_2_lower_fitness.ipynb))<br/>

#### Alternative Splicing
Training and evaluation of Exploration networks for engineering (differential) Alternative Splicing.

[Notebook 1: Engineering Splicing Isoforms (HEK)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/splicing/splicing_target_isoform_genesis_hek.ipynb)<br/>
[Notebook 2: Engineering De-Novo Splice Sites (HEK)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/splicing/splicing_cleavage_genesis_hek.ipynb)<br/>
[Notebook 3a: Differential - CHO vs. MCF7 (CNN Predictor)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/splicing/splicing_differential_genesis_cnn_cho_vs_mcf7.ipynb)<br/>
[Notebook 3b: Differential - CHO vs. MCF7 (Hexamer Regressor)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/splicing/splicing_differential_genesis_logistic_regression_both_regions_cho_vs_mcf7.ipynb)<br/>
[Notebook 3c: Differential - CHO vs. MCF7 (Both Predictors)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/splicing/splicing_differential_genesis_cnn_and_logistic_regression_both_regions_cho_vs_mcf7.ipynb)<br/>

#### GFP
Evaluation of *Likelihood-bounded* DENs for engineering GFP variants. Here we combine importance sampling of a variational autoencoder (VAE) and straight-through approximation to propagate likelihood gradients to the generator. The benchmarking test bed is adapted from ([Brookes et. al., 2019](https://arxiv.org/abs/1901.10060); [Github](https://github.com/dhbrookes/CbAS)).

[Notebook 0: Importance-Sampled Train Set Likelihoods (VAE)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/gfp/gfp_test_vaes.ipynb)<br/>
[Notebook 1: Likelihood-bounded DEN Training](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/gfp/gfp_sf_kl_den_repeats_with_edit_distances.ipynb)<br/>
[Notebook 2a: Plot Bar Chart Comparison](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/gfp/plotting_kl_den_sf_with_edit_distances.ipynb)<br/>
[Notebook 2b: Plot Trajectory Comparison](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/gfp/plotting_kl_den_sf_traj.ipynb)<br/>

#### SPI1 TF Binding (DragoNN)
Benchmark evaluation for the DragoNN fitness predictor.

[Notebook 1a: Maximal TF Binding Score (Sequence Diversity)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/dragonn/dragonn_genesis_max_spi1_earthmover.ipynb)<br/>
[Notebook 1b: Maximal TF Binding Score (Latent Diversity)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/dragonn/dragonn_genesis_max_spi1_earthmover_latent_penalty.ipynb)<br/>
[Notebook 2a: Benchmark Comparison](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/dragonn/dragonn_genesis_max_spi1_eval_models.ipynb)<br/>
[Notebook 2b: Benchmark Comparison (Computational Cost)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/dragonn/dragonn_genesis_max_spi1_eval_basinhopping_trajs.ipynb)<br/>

#### Transcriptional Activity (MPRA-DragoNN)
Benchmark evaluation for the MPRA-DragoNN fitness predictor.

[Notebook 1: Maximal Transcriptional Activity](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/mpradragonn/mpradragonn_genesis_sv40_max_activity_earthmover.ipynb)<br/>
[Notebook 2a: Benchmark Comparison](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/mpradragonn/mpradragonn_genesis_sv40_max_activity_eval_models.ipynb)<br/>
[Notebook 2b: Benchmark Comparison (Computational Cost)](https://nbviewer.jupyter.org/github/johli/genesis/blob/master/analysis/mpradragonn/mpradragonn_genesis_sv40_max_activity_eval_basinhopping_trajs.ipynb)<br/>

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
