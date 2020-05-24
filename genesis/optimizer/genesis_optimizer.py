import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, LSTM, ConvLSTM2D, GRU, BatchNormalization, LocallyConnected2D, Permute
from keras.layers import Concatenate, Reshape, Softmax, Conv2DTranspose, Embedding, Multiply
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
from keras import backend as K
import keras.losses

import tensorflow as tf

import isolearn.keras as iso

import numpy as np

from genesis.generator import sample_pwm, max_pwm


#GENESIS loss helper functions

def get_target_entropy_mse(pwm_start=0, pwm_end=100, target_bits=2.0) :
    
    def target_entropy_mse(pwm) :
        pwm_section = pwm[:, pwm_start:pwm_end, :, :]
        entropy = pwm_section * -K.log(K.clip(pwm_section, K.epsilon(), 1. - K.epsilon())) / K.log(2.0)
        entropy = K.sum(entropy, axis=(2, 3))
        conservation = 2.0 - entropy

        return K.mean((conservation - target_bits)**2, axis=-1)
    
    return target_entropy_mse

def get_target_entropy_mae(pwm_start=0, pwm_end=100, target_bits=2.0) :
    
    def target_entropy_mae(pwm) :
        pwm_section = pwm[:, pwm_start:pwm_end, :, :]
        entropy = pwm_section * -K.log(K.clip(pwm_section, K.epsilon(), 1. - K.epsilon())) / K.log(2.0)
        entropy = K.sum(entropy, axis=(2, 3))
        conservation = 2.0 - entropy

        return K.mean(K.abs(conservation - target_bits), axis=-1)
    
    return target_entropy_mae

def get_target_entropy_sme(pwm_start=0, pwm_end=100, target_bits=2.0) :
    
    def target_entropy_sme(pwm) :
        pwm_section = pwm[:, pwm_start:pwm_end, :, :]
        entropy = pwm_section * -K.log(K.clip(pwm_section, K.epsilon(), 1. - K.epsilon())) / K.log(2.0)
        entropy = K.sum(entropy, axis=(2, 3))
        conservation = 2.0 - entropy

        return (K.mean(conservation, axis=-1) - target_bits)**2
    
    return target_entropy_sme

def get_target_entropy_ame(pwm_start=0, pwm_end=100, target_bits=2.0) :
    
    def target_entropy_ame(pwm) :
        pwm_section = pwm[:, pwm_start:pwm_end, :, :]
        entropy = pwm_section * -K.log(K.clip(pwm_section, K.epsilon(), 1. - K.epsilon())) / K.log(2.0)
        entropy = K.sum(entropy, axis=(2, 3))
        conservation = 2.0 - entropy

        return K.abs(K.mean(conservation, axis=-1) - target_bits)
    
    return target_entropy_ame

def get_margin_entropy_ame(pwm_start=0, pwm_end=100, min_bits=1.0) :
    
    def margin_entropy_ame(pwm) :
        pwm_section = pwm[:, pwm_start:pwm_end, :, :]
        entropy = pwm_section * -K.log(K.clip(pwm_section, K.epsilon(), 1. - K.epsilon())) / K.log(2.0)
        entropy = K.sum(entropy, axis=(2, 3))
        conservation = 2.0 - entropy

        mean_conservation = K.mean(conservation, axis=-1)

        margin_conservation = K.switch(mean_conservation < K.constant(min_bits, shape=(1,)), K.constant(min_bits, shape=(1,)) - mean_conservation, K.zeros_like(mean_conservation))
    
        return margin_conservation

    
    return margin_entropy_ame

def get_target_entropy_sme_masked(pwm_start=0, pwm_end=100, target_bits=2.0) :
    
    def target_entropy_sme_masked(pwm, pwm_mask) :
        pwm_section = pwm[:, pwm_start:pwm_end, :, :]
        entropy = pwm_section * -K.log(K.clip(pwm_section, K.epsilon(), 1. - K.epsilon())) / K.log(2.0)
        entropy = K.sum(entropy, axis=(2, 3))
        conservation = 2.0 - entropy

        pwm_mask_section = pwm_mask[:, pwm_start:pwm_end, :, :]
        mask = K.max(pwm_mask_section, axis=(2, 3))
        n_unmasked = K.sum(mask, axis=-1)

        return ((K.sum(conservation * mask, axis=-1) / n_unmasked) - target_bits)**2
    
    return target_entropy_sme_masked

def get_margin_entropy_ame_masked(pwm_start=0, pwm_end=100, min_bits=2.0) :
    
    def margin_entropy_ame(pwm, pwm_mask) :
        pwm_section = pwm[:, pwm_start:pwm_end, :, :]
        entropy = pwm_section * -K.log(K.clip(pwm_section, K.epsilon(), 1. - K.epsilon())) / K.log(2.0)
        entropy = K.sum(entropy, axis=(2, 3))
        conservation = 2.0 - entropy

        pwm_mask_section = pwm_mask[:, pwm_start:pwm_end, :, :]
        mask = K.max(pwm_mask_section, axis=(2, 3))
        n_unmasked = K.sum(mask, axis=-1)
        
        mean_conservation = K.sum(conservation * mask, axis=-1) / n_unmasked
        
        margin_conservation = K.switch(mean_conservation < K.constant(min_bits, shape=(1,)), K.constant(min_bits, shape=(1,)) - mean_conservation, K.zeros_like(mean_conservation))

        return margin_conservation
    
    return margin_entropy_ame


def get_pwm_cross_entropy(pwm_start=0, pwm_end=100) :

    def pwm_cross_entropy(pwm1, pwm2) :
        pwm1 = K.clip(pwm1, K.epsilon(), 1.0 - K.epsilon())[:, pwm_start:pwm_end, :, :]
        pwm2 = K.clip(pwm2, K.epsilon(), 1.0 - K.epsilon())[:, pwm_start:pwm_end, :, :]
        
        return K.mean(K.sum(pwm2 * K.log(pwm2 / pwm1), axis=(2, 3)), axis=-1)

    return pwm_cross_entropy

def get_pwm_sample_entropy(pwm_start=0, pwm_end=100, shift_1_nt=False) :

    def pwm_sample_entropy(pwm1, pwm2) :
        sampled_pwm_1 = pwm1[..., pwm_start:pwm_end, :, :]
        sampled_pwm_2 = pwm2[..., pwm_start:pwm_end, :, :]
        
        mean_sample_ent = K.mean(K.sum(sampled_pwm_1 * sampled_pwm_2, axis=(-2, -1)), axis=-1)
        mean_sample_ent_shift_l_1 = K.mean(K.sum(sampled_pwm_1[..., 1:, :, :] * sampled_pwm_2[..., :-1, :, :], axis=(-2, -1)), axis=-1)
        mean_sample_ent_shift_r_1 = K.mean(K.sum(sampled_pwm_1[..., :-1, :, :] * sampled_pwm_2[..., 1:, :, :], axis=(-2, -1)), axis=-1)

        if shift_1_nt :
            return mean_sample_ent + mean_sample_ent_shift_l_1 + mean_sample_ent_shift_r_1
        else :
            return mean_sample_ent

    return pwm_sample_entropy

def get_pwm_margin_sample_entropy(pwm_start=0, pwm_end=100, margin=0.5, shift_1_nt=False) : 

    def pwm_margin_sample_entropy(pwm1, pwm2) :
        sampled_pwm_1 = pwm1[..., pwm_start:pwm_end, :, :]
        sampled_pwm_2 = pwm2[..., pwm_start:pwm_end, :, :]
        
        mean_sample_ent = K.mean(K.sum(sampled_pwm_1 * sampled_pwm_2, axis=(-2, -1)), axis=-1)
        mean_sample_ent_shift_l_1 = K.mean(K.sum(sampled_pwm_1[..., 1:, :, :] * sampled_pwm_2[..., :-1, :, :], axis=(-2, -1)), axis=-1)
        mean_sample_ent_shift_r_1 = K.mean(K.sum(sampled_pwm_1[..., :-1, :, :] * sampled_pwm_2[..., 1:, :, :], axis=(-2, -1)), axis=-1)
        
        margin_sample_ent = K.switch(mean_sample_ent > K.constant(margin, shape=(1,)), mean_sample_ent - margin, K.zeros_like(mean_sample_ent))
        margin_sample_ent_l_1 = K.switch(mean_sample_ent_shift_l_1 > K.constant(margin, shape=(1,)), mean_sample_ent_shift_l_1 - margin, K.zeros_like(mean_sample_ent))
        margin_sample_ent_r_1 = K.switch(mean_sample_ent_shift_r_1 > K.constant(margin, shape=(1,)), mean_sample_ent_shift_r_1 - margin, K.zeros_like(mean_sample_ent))
        
        if shift_1_nt :
            return margin_sample_ent + margin_sample_ent_l_1 + margin_sample_ent_r_1
        else :
            return margin_sample_ent

    return pwm_margin_sample_entropy


def get_pwm_sample_entropy_masked(pwm_start=0, pwm_end=100, shift_1_nt=False) :

    def pwm_sample_entropy_masked(pwm1, pwm2, pwm_mask) :
        sampled_pwm_1 = pwm1[..., pwm_start:pwm_end, :, :]
        sampled_pwm_2 = pwm2[..., pwm_start:pwm_end, :, :]

        sampled_pwm_mask = pwm_mask[..., pwm_start:pwm_end, :, :]
        mask = K.max(sampled_pwm_mask, axis=(-2, -1))
        n_unmasked = K.sum(mask, axis=-1)
        
        mean_sample_ent = K.sum(K.sum(sampled_pwm_1 * sampled_pwm_2, axis=(-2, -1)) * mask, axis=-1) / n_unmasked
        mean_sample_ent_shift_l_1 = K.sum(K.sum(sampled_pwm_1[..., 1:, :, :] * sampled_pwm_2[..., :-1, :, :], axis=(-2, -1)) * mask[..., 1:], axis=-1) / n_unmasked
        mean_sample_ent_shift_r_1 = K.sum(K.sum(sampled_pwm_1[..., :-1, :, :] * sampled_pwm_2[..., 1:, :, :], axis=(-2, -1)) * mask[..., :-1], axis=-1) / n_unmasked

        if shift_1_nt :
            return mean_sample_ent + mean_sample_ent_shift_l_1 + mean_sample_ent_shift_r_1
        else :
            return mean_sample_ent

    return pwm_sample_entropy_masked

def get_pwm_margin_sample_entropy_masked(pwm_start=0, pwm_end=100, margin=0.5, shift_1_nt=False) : 

    def pwm_margin_sample_entropy_masked(pwm1, pwm2, pwm_mask) :
        sampled_pwm_1 = pwm1[..., pwm_start:pwm_end, :, :]
        sampled_pwm_2 = pwm2[..., pwm_start:pwm_end, :, :]

        sampled_pwm_mask = pwm_mask[..., pwm_start:pwm_end, :, :]
        mask = K.max(sampled_pwm_mask, axis=(-2, -1))
        n_unmasked = K.sum(mask, axis=-1)
        
        mean_sample_ent = K.sum(K.sum(sampled_pwm_1 * sampled_pwm_2, axis=(-2, -1)) * mask, axis=-1) / n_unmasked
        mean_sample_ent_shift_l_1 = K.sum(K.sum(sampled_pwm_1[..., 1:, :, :] * sampled_pwm_2[..., :-1, :, :], axis=(-2, -1)) * mask[..., 1:], axis=-1) / n_unmasked
        mean_sample_ent_shift_r_1 = K.sum(K.sum(sampled_pwm_1[..., :-1, :, :] * sampled_pwm_2[..., 1:, :, :], axis=(-2, -1)) * mask[..., :-1], axis=-1) / n_unmasked
        
        margin_sample_ent = K.switch(mean_sample_ent > K.constant(margin, shape=(1,)), mean_sample_ent - margin, K.zeros_like(mean_sample_ent))
        margin_sample_ent_l_1 = K.switch(mean_sample_ent_shift_l_1 > K.constant(margin, shape=(1,)), mean_sample_ent_shift_l_1 - margin, K.zeros_like(mean_sample_ent))
        margin_sample_ent_r_1 = K.switch(mean_sample_ent_shift_r_1 > K.constant(margin, shape=(1,)), mean_sample_ent_shift_r_1 - margin, K.zeros_like(mean_sample_ent))
        
        if shift_1_nt :
            return margin_sample_ent + margin_sample_ent_l_1 + margin_sample_ent_r_1
        else :
            return margin_sample_ent

    return pwm_margin_sample_entropy_masked

'''
def get_pwm_sample_entropy_masked(pwm_start=0, pwm_end=100, shift_1_nt=False) :

    def pwm_sample_entropy_masked(pwm1, pwm2, pwm_mask) :
        sampled_pwm_1 = pwm1[..., pwm_start:pwm_end, :, :]
        sampled_pwm_2 = pwm2[..., pwm_start:pwm_end, :, :]

        sampled_pwm_mask = pwm_mask[..., pwm_start:pwm_end, :, :]
        n_unmasked = K.sum(sampled_pwm_mask, axis=(-3, -2, -1)) / 4.0
        
        mean_sample_ent = K.sum(sampled_pwm_1 * sampled_pwm_2 * sampled_pwm_mask, axis=(-3, -2, -1)) / n_unmasked
        mean_sample_ent_shift_l_1 = K.sum(sampled_pwm_1[..., 1:, :, :] * sampled_pwm_2[..., :-1, :, :] * sampled_pwm_mask[..., 1:, :, :], axis=(-3, -2, -1)) / n_unmasked
        mean_sample_ent_shift_r_1 = K.sum(sampled_pwm_1[..., :-1, :, :] * sampled_pwm_2[..., 1:, :, :] * sampled_pwm_mask[..., :-1, :, :], axis=(-3, -2, -1)) / n_unmasked

        if shift_1_nt :
            return mean_sample_ent + mean_sample_ent_shift_l_1 + mean_sample_ent_shift_r_1
        else :
            return mean_sample_ent

    return pwm_sample_entropy_masked

def get_pwm_margin_sample_entropy_masked(pwm_start=0, pwm_end=100, margin=0.5, shift_1_nt=False) : 

    def pwm_margin_sample_entropy_masked(pwm1, pwm2, pwm_mask) :
        sampled_pwm_1 = pwm1[..., pwm_start:pwm_end, :, :]
        sampled_pwm_2 = pwm2[..., pwm_start:pwm_end, :, :]

        sampled_pwm_mask = pwm_mask[..., pwm_start:pwm_end, :, :]
        n_unmasked = K.sum(sampled_pwm_mask, axis=(-3, -2, -1)) / 4.0
        
        mean_sample_ent = K.sum(sampled_pwm_1 * sampled_pwm_2 * sampled_pwm_mask, axis=(-3, -2, -1)) / n_unmasked
        mean_sample_ent_shift_l_1 = K.sum(sampled_pwm_1[..., 1:, :, :] * sampled_pwm_2[..., :-1, :, :] * sampled_pwm_mask[..., 1:, :, :], axis=(-3, -2, -1)) / n_unmasked
        mean_sample_ent_shift_r_1 = K.sum(sampled_pwm_1[..., :-1, :, :] * sampled_pwm_2[..., 1:, :, :] * sampled_pwm_mask[..., :-1, :, :], axis=(-3, -2, -1)) / n_unmasked
        
        margin_sample_ent = K.switch(mean_sample_ent > K.constant(margin, shape=(1,)), mean_sample_ent - margin, K.zeros_like(mean_sample_ent))
        margin_sample_ent_l_1 = K.switch(mean_sample_ent_shift_l_1 > K.constant(margin, shape=(1,)), mean_sample_ent_shift_l_1 - margin, K.zeros_like(mean_sample_ent))
        margin_sample_ent_r_1 = K.switch(mean_sample_ent_shift_r_1 > K.constant(margin, shape=(1,)), mean_sample_ent_shift_r_1 - margin, K.zeros_like(mean_sample_ent))
        
        if shift_1_nt :
            return margin_sample_ent + margin_sample_ent_l_1 + margin_sample_ent_r_1
        else :
            return margin_sample_ent

    return pwm_margin_sample_entropy_masked
'''

def get_punish_cse(pwm_start, pwm_end) :
    
    def punish(pwm) :

        aataaa_score = K.sum(pwm[..., pwm_start:pwm_end-5, 0, 0] * pwm[..., pwm_start+1:pwm_end-4, 0, 0] * pwm[..., pwm_start+2:pwm_end-3, 3, 0] * pwm[..., pwm_start+3:pwm_end-2, 0, 0] * pwm[..., pwm_start+4:pwm_end-1, 0, 0] * pwm[..., pwm_start+5:pwm_end, 0, 0], axis=-1)
        attaaa_score = K.sum(pwm[..., pwm_start:pwm_end-5, 0, 0] * pwm[..., pwm_start+1:pwm_end-4, 3, 0] * pwm[..., pwm_start+2:pwm_end-3, 3, 0] * pwm[..., pwm_start+3:pwm_end-2, 0, 0] * pwm[..., pwm_start+4:pwm_end-1, 0, 0] * pwm[..., pwm_start+5:pwm_end, 0, 0], axis=-1)

        return aataaa_score + attaaa_score
    
    return punish

def get_punish_c(pwm_start, pwm_end) :
    
    def punish(pwm) :

        c_score = K.sum(pwm[..., pwm_start:pwm_end, 1, 0], axis=-1)
    
        return c_score
    
    return punish

def get_punish_g(pwm_start, pwm_end) :
    
    def punish(pwm) :

        g_score = K.sum(pwm[..., pwm_start:pwm_end, 2, 0], axis=-1)
    
        return g_score
    
    return punish

def get_punish_aa(pwm_start, pwm_end) :
    
    def punish(pwm) :

        aa_score = K.sum(pwm[..., pwm_start:pwm_end-1, 0, 0] * pwm[..., pwm_start+1:pwm_end, 0, 0], axis=-1)
    
        return aa_score
    
    return punish

def get_punish_cc(pwm_start, pwm_end) :
    
    def punish(pwm) :

        cc_score = K.sum(pwm[..., pwm_start:pwm_end-1, 1, 0] * pwm[..., pwm_start+1:pwm_end, 1, 0], axis=-1)
    
        return cc_score
    
    return punish

def get_punish_gg(pwm_start, pwm_end) :
    
    def punish(pwm) :

        gg_score = K.sum(pwm[..., pwm_start:pwm_end-1, 2, 0] * pwm[..., pwm_start+1:pwm_end, 2, 0], axis=-1)
    
        return gg_score
    
    return punish

def get_reward_ggcc(pwm_start, pwm_end) :
    
    def reward(pwm) :

        ggcc_score = K.sum(pwm[..., pwm_start:pwm_end-3, 2, 0] * pwm[..., pwm_start+1:pwm_end-2, 2, 0] * pwm[..., pwm_start+2:pwm_end-1, 1, 0] * pwm[..., pwm_start+3:pwm_end, 1, 0], axis=-1)
        
        return -ggcc_score
    
    return reward

def get_reward_gg_and_cc(pwm_start, pwm_end) :
    
    def reward(pwm) :

        ggcc_score = K.sum(pwm[..., pwm_start:pwm_end-1, 2, 0] * pwm[..., pwm_start+1:pwm_end, 2, 0] * pwm[..., pwm_start:pwm_end-1, 1, 0] * pwm[..., pwm_start+1:pwm_end, 1, 0], axis=-1)
        
        return -ggcc_score
    
    return reward

def get_reward_gg_or_cc(pwm_start, pwm_end) :
    
    def punish(pwm) :

        gg_score = K.sum(pwm[..., pwm_start:pwm_end-1, 2, 0] * pwm[..., pwm_start+1:pwm_end, 2, 0], axis=-1)
        cc_score = K.sum(pwm[..., pwm_start:pwm_end-1, 1, 0] * pwm[..., pwm_start+1:pwm_end, 1, 0], axis=-1)
        
        return -(gg_score + cc_score)
    
    return punish

def kl_divergence(y_true, y_pred) :
	y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
	y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())
	
	return K.sum(y_true * K.log(y_true / y_pred), axis=-1)

def symmetric_kl_divergence(y_true, y_pred) :
	y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
	y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())
	
	return K.sum(y_true * K.log(y_true / y_pred), axis=-1) + K.sum(y_pred * K.log(y_pred / y_true), axis=-1)

def sigmoid_kl_divergence(y_true, y_pred) :
	y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
	y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())
	
	return K.sum(y_true * K.log(y_true / y_pred) + (1.0 - y_true) * K.log((1.0 - y_true) / (1.0 - y_pred)), axis=-1)

def symmetric_sigmoid_kl_divergence(y_true, y_pred) :
	y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
	y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())
	
	return K.sum(y_true * K.log(y_true / y_pred) + (1.0 - y_true) * K.log((1.0 - y_true) / (1.0 - y_pred)), axis=-1) + K.sum(y_pred * K.log(y_pred / y_true) + (1.0 - y_pred) * K.log((1.0 - y_pred) / (1.0 - y_true)), axis=-1)

def mean_squared_logit_target_error(y_pred, target=8.0) :
	#return K.mean((y_pred - target)**2, axis=-1)
	
	y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
	return K.mean((K.log(y_pred / (1.0 - y_pred)) - target)**2, axis=-1)

def mean_abs_logit_target_error(y_pred, target=8.0) :
	#return K.mean(K.abs(y_pred - target), axis=-1)
	
	y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
	return K.mean(K.abs(K.log(y_pred / (1.0 - y_pred)) - target), axis=-1)



def build_loss_model(predictor_model, loss_func, extra_loss_tensors=[]) :

	loss_out = Lambda(lambda out: loss_func(out), output_shape = (1,))(predictor_model.inputs + predictor_model.outputs + extra_loss_tensors)

	loss_model = Model(predictor_model.inputs, loss_out)#extra_loss_tensors

	return 'loss_model', loss_model