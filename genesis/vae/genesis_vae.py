import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Input, Lambda, Concatenate, Reshape, Multiply
from keras import backend as K

import tensorflow as tf

import isolearn.keras as iso

import numpy as np

from genesis.generator import st_sampled_softmax, st_hardmax_softmax

from scipy.stats import norm

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter

from genesis.visualization import letterAt

def plot_gan_logo(pwm, score, sequence_template=None, figsize=(12, 3), width_ratios=[1, 7], logo_height=1.0, plot_start=0, plot_end=164) :

    #Slice according to seq trim index
    pwm = pwm[plot_start: plot_end, :]
    sequence_template = sequence_template[plot_start: plot_end]

    pwm += 0.0001
    for j in range(0, pwm.shape[0]) :
        pwm[j, :] /= np.sum(pwm[j, :])

    entropy = np.zeros(pwm.shape)
    entropy[pwm > 0] = pwm[pwm > 0] * -np.log2(pwm[pwm > 0])
    entropy = np.sum(entropy, axis=1)
    conservation = 2 - entropy

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(1, 2, width_ratios=[width_ratios[0], width_ratios[-1]])

    ax2 = plt.subplot(gs[0])
    ax3 = plt.subplot(gs[1])

    plt.sca(ax2)
    plt.axis('off')


    annot_text = '\nScore = ' + str(round(score, 4))

    ax2.text(0.99, 0.5, annot_text, horizontalalignment='right', verticalalignment='center', transform=ax2.transAxes, color='black', fontsize=12, weight="bold")

    height_base = (1.0 - logo_height) / 2.

    for j in range(0, pwm.shape[0]) :
        sort_index = np.argsort(pwm[j, :])

        for ii in range(0, 4) :
            i = sort_index[ii]

            nt_prob = pwm[j, i] * conservation[j]

            nt = ''
            if i == 0 :
                nt = 'A'
            elif i == 1 :
                nt = 'C'
            elif i == 2 :
                nt = 'G'
            elif i == 3 :
                nt = 'T'

            color = None
            if sequence_template[j] != 'N' :
                color = 'black'

            if ii == 0 :
                letterAt(nt, j + 0.5, height_base, nt_prob * logo_height, ax3, color=color)
            else :
                prev_prob = np.sum(pwm[j, sort_index[:ii]] * conservation[j]) * logo_height
                letterAt(nt, j + 0.5, height_base + prev_prob, nt_prob * logo_height, ax3, color=color)

    plt.sca(ax3)

    plt.xlim((0, plot_end - plot_start))
    plt.ylim((0, 2))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.axis('off')
    ax3.axhline(y=0.01 + height_base, color='black', linestyle='-', linewidth=2)


    for axis in fig.axes :
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)

    plt.tight_layout()

    plt.show()

#One-hot encode sequences
def one_hot_encode(seq, n=1, singleton_axis=-1) :

    one_hots = None
    if singleton_axis == 1 :
        one_hots = np.zeros((n, 1, len(seq), 4))
        for j in range(len(seq)) :
            if seq[j] == 'A' :
                one_hots[:, 0, j, 0] = 1.
            elif seq[j] == 'C' :
                one_hots[:, 0, j, 1] = 1.
            elif seq[j] == 'G' :
                one_hots[:, 0, j, 2] = 1.
            elif seq[j] == 'T' :
                one_hots[:, 0, j, 3] = 1.
    else :
        one_hots = np.zeros((n, len(seq), 4, 1))
        for j in range(len(seq)) :
            if seq[j] == 'A' :
                one_hots[:, j, 0, 0] = 1.
            elif seq[j] == 'C' :
                one_hots[:, j, 1, 0] = 1.
            elif seq[j] == 'G' :
                one_hots[:, j, 2, 0] = 1.
            elif seq[j] == 'T' :
                one_hots[:, j, 3, 0] = 1.
    
    return one_hots

def get_z_sample_numpy(z_mean, z_log_var, n_samples=1) :
    
    n = z_mean.shape[0]
    m = z_mean.shape[2]
    
    epsilon = np.random.normal(loc=0., scale=1., size=(n, n_samples, m))
    
    return z_mean + np.exp(0.5 * z_log_var) * epsilon

#Evaluate VAE Likelihood (ELBO) on supplied data
def evaluate_elbo(vae_encoder_model, vae_decoder_model, sequence_one_hots, pwm_start=0, pwm_end=-1, n_samples=1) :
    _epsilon = 10**-6
    
    if pwm_end == -1 :
        pwm_end = sequence_one_hots.shape[2]
    
    #Get sequence VAE encodings
    z_mean, z_log_var, _ = vae_encoder_model.predict(x=sequence_one_hots, batch_size=32, verbose=False)

    z_mean = np.tile(np.expand_dims(z_mean, axis=1), (1, n_samples, 1))
    z_log_var = np.tile(np.expand_dims(z_log_var, axis=1), (1, n_samples, 1))
    z = get_z_sample_numpy(z_mean, z_log_var, n_samples=n_samples)
    
    #Get re-decoded sequence PWMs
    s_dummy = np.zeros((sequence_one_hots.shape[0], 1))
    
    decoded_pwms = np.zeros((sequence_one_hots.shape[0], n_samples) + sequence_one_hots.shape[1:])

    for sample_ix in range(n_samples) :
        _, decoded_pwm, _ = vae_decoder_model.predict(x=[s_dummy, z[:, sample_ix, :]], batch_size=32, verbose=False)
        decoded_pwms[:, sample_ix, :, :, :] = decoded_pwm

    sequence_one_hots_expanded = np.tile(np.expand_dims(sequence_one_hots, 1), (1, n_samples, 1, 1, 1))
    
    #Calculate reconstruction log prob
    log_p_x_given_z = np.sum(np.sum(sequence_one_hots_expanded[:, :, :, pwm_start:pwm_end, :] * np.log(np.clip(decoded_pwms[:, :, :, pwm_start:pwm_end, :], _epsilon, 1. - _epsilon)) / np.log(10.), axis=(2, 4)), axis=2)

    #Calculate standard normal and importance log probs
    log_p_std_normal = np.sum(norm.logpdf(z, 0., 1.) / np.log(10.), axis=-1)
    log_p_importance = np.sum(norm.logpdf(z, z_mean, np.sqrt(np.exp(z_log_var))) / np.log(10.), axis=-1)

    #Calculate per-sample ELBO
    log_p_vae = log_p_x_given_z + log_p_std_normal - log_p_importance
    log_p_vae_div_n = log_p_vae - np.log(n_samples) / np.log(10.)

    #Calculate mean ELBO across samples (log-sum-exp trick)
    max_log_p_vae = np.max(log_p_vae_div_n, axis=-1)
    
    log_mean_p_vae = max_log_p_vae + np.log(np.sum(10**(log_p_vae_div_n - np.expand_dims(max_log_p_vae, axis=-1)), axis=-1)) / np.log(10.)
    mean_log_p_vae = np.mean(log_mean_p_vae)
    
    return log_mean_p_vae, mean_log_p_vae, log_p_vae


#Plot join histograms
def plot_joint_histo(measurements, labels, colors, x_label, y_label, n_bins=50, figsize=(6, 4), save_fig=False, fig_name="default_1", fig_dpi=150, min_val=None, max_val=None, max_y_val=None) :
    
    min_hist_val = np.min(measurements[0])
    max_hist_val = np.max(measurements[0])
    for i in range(1, len(measurements)) :
        min_hist_val = min(min_hist_val, np.min(measurements[i]))
        max_hist_val = max(max_hist_val, np.max(measurements[i]))
    
    if min_val is not None :
        min_hist_val = min_val
    if max_val is not None :
        max_hist_val = max_val

    hists = []
    bin_edges = []
    means = []
    for i in range(len(measurements)) :
        hist, b_edges = np.histogram(measurements[i], range=(min_hist_val, max_hist_val), bins=n_bins, density=True)
        
        hists.append(hist)
        bin_edges.append(b_edges)
        means.append(np.mean(measurements[i]))
    
    bin_width = bin_edges[0][1] - bin_edges[0][0]


    #Compare Log Likelihoods
    f = plt.figure(figsize=figsize)

    for i in range(len(measurements)) :
        plt.bar(bin_edges[i][1:] - bin_width/2., hists[i], width=bin_width, linewidth=2, edgecolor='black', color=colors[i], label=labels[i])
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.xlim(min_hist_val, max_hist_val)
    if max_y_val is not None :
        plt.ylim(0, max_y_val)

    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)

    for i in range(len(measurements)) :
        plt.axvline(x=means[i], linewidth=2, color=colors[i], linestyle="--")

    plt.legend(fontsize=14, loc='upper left')

    plt.tight_layout()
    
    if save_fig :
        plt.savefig(fig_name + ".eps")
        plt.savefig(fig_name + ".svg")
        plt.savefig(fig_name + ".png", dpi=fig_dpi, transparent=True)
    
    plt.show()

#Helper dummy function for loading keras models
def min_pred(y_true, y_pred) :
        return y_pred

#Keras function to calculate normal distribution log pdf
def normal_log_prob(x, loc=0., scale=1.) :
    return _normal_log_unnormalized_prob(x, loc, scale) - _normal_log_normalization(scale)

def _normal_log_unnormalized_prob(x, loc, scale):
    return -0.5 * K.square((x - loc) / scale)

def _normal_log_normalization(scale):
    return 0.5 * K.log(2. * K.constant(np.pi)) + K.log(scale)

#Keras function to sample latent vectors
def get_z_sample(z_inputs) :
    
    z_mean, z_log_var = z_inputs
    
    batch_size = K.shape(z_mean)[0]
    latent_dim = K.int_shape(z_mean)[1]
    
    epsilon = K.random_normal(shape=(batch_size, latent_dim))
    
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

#Keras function to sample (multiple) latent vectors
def get_z_samples(z_inputs, n_z_samples=1) :
    
    z_mean, z_log_var = z_inputs
    
    batch_size = K.shape(z_mean)[0]
    n_samples = K.shape(z_mean)[1]
    latent_dim = K.int_shape(z_mean)[3]
    
    epsilon = K.random_normal(shape=(batch_size, n_samples, n_z_samples, latent_dim))
    
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

#Code for constructing a (differentiable) VAE ELBO estimator in Keras
def build_vae(generator, encoder_model_path, decoder_model_path, batch_size=1, seq_length=205, n_samples=1, n_z_samples=1, vae_latent_dim=100, vae_pwm_start=0, vae_pwm_end=-1, vae_upstream_padding="", vae_downstream_padding="", transform_adversary=False) :
    
    #Connect generated sequence samples from generator to vae
    generated_sequence_pwm = generator.outputs[3]
    generated_sequence_adv = generator.outputs[4]
    generated_sequence_samples = generator.outputs[5]
    generated_sequence_adv_samples = generator.outputs[6]
    
    if vae_pwm_end == -1 :
        vae_pwm_end = seq_length
    
    #Load encoder model
    saved_vae_encoder_model = load_model(encoder_model_path, custom_objects={'st_sampled_softmax':st_sampled_softmax, 'st_hardmax_softmax':st_hardmax_softmax, 'min_pred':min_pred})
    saved_vae_encoder_model.trainable = False
    saved_vae_encoder_model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999))
    
    #Load decoder model
    saved_vae_decoder_model = load_model(decoder_model_path, custom_objects={'st_sampled_softmax':st_sampled_softmax, 'st_hardmax_softmax':st_hardmax_softmax, 'min_pred':min_pred})
    saved_vae_decoder_model.trainable = False
    saved_vae_decoder_model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999))
    
    
    #Construct upstream sequence padding constant for vae
    vae_upstream_mat = None
    if len(vae_upstream_padding) > 0 :
        vae_upstream_mat = np.tile(np.expand_dims(one_hot_encode(vae_upstream_padding, n=1, singleton_axis=-1), axis=1), (batch_size, n_samples, 1, 1, 1))
    
    #Construct downstream sequence padding constant for vae
    vae_downstream_mat = None
    if len(vae_downstream_padding) > 0 :
        vae_downstream_mat = np.tile(np.expand_dims(one_hot_encode(vae_downstream_padding, n=1, singleton_axis=-1), axis=1), (batch_size, n_samples, 1, 1, 1))
    
    
    #Construct vae elbo keras function (lambda layer)
    def _vae_elbo_func(pwm_and_sampled_pwm, vae_upstream_mat=vae_upstream_mat, vae_downstream_mat=vae_downstream_mat, batch_size=batch_size, n_samples=n_samples, n_z_samples=n_z_samples, transform_adversary=transform_adversary) :
        
        pwm_1, pwm_2, sampled_pwm_1, sampled_pwm_2 = pwm_and_sampled_pwm
        
        def _encode_and_sample(saved_vae_encoder_model, pwm, sampled_pwm, vae_pwm_start, vae_pwm_end, vae_upstream_mat, vae_downstream_mat, vae_latent_dim, n_z_samples) :
            vae_pwm = pwm[:, vae_pwm_start:vae_pwm_end, :, :]
            vae_sampled_pwm = sampled_pwm[:, :, vae_pwm_start:vae_pwm_end, :, :]
            if vae_upstream_mat is not None :
                vae_pwm = K.concatenate([K.constant(vae_upstream_mat[:, 0, ...]), vae_pwm], axis=1)
                vae_sampled_pwm = K.concatenate([K.constant(vae_upstream_mat), vae_sampled_pwm], axis=2)
            if vae_downstream_mat is not None :
                vae_pwm = K.concatenate([vae_pwm, K.constant(vae_downstream_mat[:, 0, ...])], axis=1)
                vae_sampled_pwm = K.concatenate([vae_sampled_pwm, K.constant(vae_downstream_mat)], axis=2)

            vae_sampled_pwm_permuted = K.permute_dimensions(vae_sampled_pwm, (1, 0, 4, 2, 3))

            z_param_collection = tf.map_fn(lambda x_in: K.concatenate(saved_vae_encoder_model(x_in), axis=-1)[..., :2*vae_latent_dim], vae_sampled_pwm_permuted, parallel_iterations=16)

            z_mean = K.permute_dimensions(z_param_collection[..., :vae_latent_dim], (1, 0, 2))
            z_log_var = K.permute_dimensions(z_param_collection[..., vae_latent_dim:2*vae_latent_dim], (1, 0, 2))

            z_mean = K.tile(K.expand_dims(z_mean, axis=2), (1, 1, n_z_samples, 1))
            z_log_var = K.tile(K.expand_dims(z_log_var, axis=2), (1, 1, n_z_samples, 1))

            z = get_z_samples([z_mean, z_log_var], n_z_samples=n_z_samples)
            
            return vae_pwm, vae_sampled_pwm, z_mean, z_log_var, z
        
        vae_pwm_1, vae_sampled_pwm_1, z_mean_1, z_log_var_1, z_1 = _encode_and_sample(saved_vae_encoder_model, pwm_1, sampled_pwm_1, vae_pwm_start, vae_pwm_end, vae_upstream_mat, vae_downstream_mat, vae_latent_dim, n_z_samples)
        
        if transform_adversary :
            vae_pwm_2, vae_sampled_pwm_2, z_mean_2, z_log_var_2, z_2 = _encode_and_sample(saved_vae_encoder_model, pwm_2, sampled_pwm_2, vae_pwm_start, vae_pwm_end, vae_upstream_mat, vae_downstream_mat, vae_latent_dim, n_z_samples)
            
        z_1_permuted = K.permute_dimensions(z_1, (1, 2, 0, 3))

        s_1 = K.zeros((batch_size, 1))
        
        decoded_pwm_1 = tf.map_fn(lambda z_in: tf.map_fn(lambda z_in_in: saved_vae_decoder_model([s_1, z_in_in])[1], z_in, parallel_iterations=16), z_1_permuted, parallel_iterations=16)
        decoded_pwm_1 = K.permute_dimensions(decoded_pwm_1, (2, 0, 1, 4, 5, 3))

        vae_pwm_tiled_1 = K.tile(K.expand_dims(vae_pwm_1, axis=1), (1, n_z_samples, 1, 1, 1))
        vae_sampled_pwm_tiled_1 = K.tile(K.expand_dims(vae_sampled_pwm_1, axis=2), (1, 1, n_z_samples, 1, 1, 1))

        if transform_adversary :
            return [vae_pwm_tiled_1, vae_sampled_pwm_tiled_1, z_mean_1, z_log_var_1, z_1, decoded_pwm_1, vae_pwm_2, vae_sampled_pwm_2, z_mean_2, z_log_var_2, z_2]
        else :
            return [vae_pwm_tiled_1, vae_sampled_pwm_tiled_1, z_mean_1, z_log_var_1, z_1, decoded_pwm_1]
    
    vae_elbo_layer = Lambda(_vae_elbo_func)
    
    #Call vae elbo estimator on generator sequences
    vae_elbo_outputs = vae_elbo_layer([generated_sequence_pwm, generated_sequence_adv, generated_sequence_samples, generated_sequence_adv_samples])
    
    return vae_elbo_outputs
