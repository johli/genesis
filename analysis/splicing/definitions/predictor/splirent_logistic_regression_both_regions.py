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

import pandas as pd

#APARENT Saved Model definition

def load_saved_predictor(model_path, library_contexts=None) :

	def _initialize_predictor_weights(predictor_model) :
		#Load nmer weights
		nmer_weights_1 = pd.read_csv(model_path + 'alt_5ss_6mers_both_regions_1.csv', sep='\t')
		nmer_weights_2 = pd.read_csv(model_path + 'alt_5ss_6mers_both_regions_2.csv', sep='\t')
		nmers = list(nmer_weights_1.iloc[1:]['nmer'].values)

		hek_w_0 = nmer_weights_1.iloc[0]['hek']
		hek_w = np.concatenate([np.array(nmer_weights_1.iloc[1:]['hek'].values), np.array(nmer_weights_2['hek'].values)], axis=0)

		hela_w_0 = nmer_weights_1.iloc[0]['hela']
		hela_w = np.concatenate([np.array(nmer_weights_1.iloc[1:]['hela'].values), np.array(nmer_weights_2['hela'].values)], axis=0)

		mcf7_w_0 = nmer_weights_1.iloc[0]['mcf7']
		mcf7_w = np.concatenate([np.array(nmer_weights_1.iloc[1:]['mcf7'].values), np.array(nmer_weights_2['mcf7'].values)], axis=0)

		cho_w_0 = nmer_weights_1.iloc[0]['cho']
		cho_w = np.concatenate([np.array(nmer_weights_1.iloc[1:]['cho'].values), np.array(nmer_weights_2['cho'].values)], axis=0)
		
		#Initialize nmer detection kernels
		mer6_kernels = np.zeros((6, 4, 1, 4096))
		mer6_bias = np.ones(4096) * -5

		for k in range(len(nmers)) :
			nmer = nmers[k]
			for pos in range(6) :
				if nmer[pos] == 'A' :
					mer6_kernels[pos, 0, 0, k] = 1.
				elif nmer[pos] == 'C' :
					mer6_kernels[pos, 1, 0, k] = 1.
				elif nmer[pos] == 'G' :
					mer6_kernels[pos, 2, 0, k] = 1.
				elif nmer[pos] == 'T' :
					mer6_kernels[pos, 3, 0, k] = 1.

		predictor_model.get_layer('mer6_conv').set_weights([mer6_kernels, mer6_bias])


		mer4_kernels = np.zeros((4, 4, 1, 256))
		mer4_bias = np.ones(256) * -3
		mer4 = []
		for b1 in ['A', 'C', 'G', 'T'] :
			for b2 in ['A', 'C', 'G', 'T'] :
				for b3 in ['A', 'C', 'G', 'T'] :
					for b4 in ['A', 'C', 'G', 'T'] :
						mer4.append(b1 + b2 + b3 + b4)

		for k in range(256) :
			nmer = mer4[k]
			for pos in range(4) :
				if nmer[pos] == 'A' :
					mer4_kernels[pos, 0, 0, k] = 1.
				elif nmer[pos] == 'C' :
					mer4_kernels[pos, 1, 0, k] = 1.
				elif nmer[pos] == 'G' :
					mer4_kernels[pos, 2, 0, k] = 1.
				elif nmer[pos] == 'T' :
					mer4_kernels[pos, 3, 0, k] = 1.

		predictor_model.get_layer('mer4_conv').set_weights([mer4_kernels, mer4_bias])


		#Initialize nmer logistic regression weights
		predictor_model.get_layer('mer6_weights').set_weights(
			[
				np.concatenate(
					[
						hek_w.reshape(-1, 1),
						hela_w.reshape(-1, 1),
						mcf7_w.reshape(-1, 1),
						cho_w.reshape(-1, 1),
					],
					axis=1
				),
				np.array([hek_w_0, hela_w_0, mcf7_w_0, cho_w_0])
			]
		)

	def _load_predictor_func(sequence_input, sequence_class) :
		
		#Predictor model definition
		mer6_layer = Conv2D(4096, (6, 4), padding='valid', activation='relu', name='mer6_conv')
		mer6_out = mer6_layer(sequence_input)

		mer4_layer = Conv2D(256, (4, 4), padding='valid', activation='relu', name='mer4_conv')
		mer4_out = mer4_layer(sequence_input)

		mer6_count_region_1 = Lambda(lambda x: K.sum(x[:, 5:40-5, :, :], axis=(1, 2)))(mer6_out)
		mer4_count_region_1 = Lambda(lambda x: K.sum(x[:, 7:38-3, :, :], axis=(1, 2)))(mer4_out)

		mer6_count_region_2 = Lambda(lambda x: K.sum(x[:, 48:83-5, :, :], axis=(1, 2)))(mer6_out)
		mer4_count_region_2 = Lambda(lambda x: K.sum(x[:, 50:81-3, :, :], axis=(1, 2)))(mer4_out)

		mer6_count = Concatenate(axis=-1)([mer6_count_region_1, mer6_count_region_2])
		mer4_count = Concatenate(axis=-1)([mer4_count_region_1, mer4_count_region_2])

		plasmid_score_iso = Dense(4, activation='linear', kernel_initializer='zeros', name='mer6_weights')(mer6_count)
		plasmid_out_iso = Lambda(lambda x: K.sigmoid(x))(plasmid_score_iso)

		predictor_inputs = []
		predictor_outputs = [mer4_count, mer6_count, plasmid_score_iso, plasmid_out_iso]

		return predictor_inputs, predictor_outputs, _initialize_predictor_weights

	return _load_predictor_func
