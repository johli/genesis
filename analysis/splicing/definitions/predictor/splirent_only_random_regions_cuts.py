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


#APARENT Saved Model definition

def load_saved_predictor(model_path, library_contexts=None) :

	saved_model = load_model(model_path)

	def _initialize_predictor_weights(predictor_model, saved_model=saved_model) :
		#Load pre-trained model
		predictor_model.get_layer('splirent_conv_1').set_weights(saved_model.get_layer('conv1d_1').get_weights())
		predictor_model.get_layer('splirent_conv_1').trainable = False

		predictor_model.get_layer('splirent_conv_2').set_weights(saved_model.get_layer('conv1d_2').get_weights())
		predictor_model.get_layer('splirent_conv_2').trainable = False

		predictor_model.get_layer('splirent_dense_1').set_weights(saved_model.get_layer('dense_1').get_weights())
		predictor_model.get_layer('splirent_dense_1').trainable = False

		predictor_model.get_layer('splirent_hek_dense').set_weights(saved_model.get_layer('dense_2').get_weights())
		predictor_model.get_layer('splirent_hek_dense').trainable = False

		predictor_model.get_layer('splirent_hela_dense').set_weights(saved_model.get_layer('dense_3').get_weights())
		predictor_model.get_layer('splirent_hela_dense').trainable = False

		predictor_model.get_layer('splirent_mcf7_dense').set_weights(saved_model.get_layer('dense_4').get_weights())
		predictor_model.get_layer('splirent_mcf7_dense').trainable = False

		predictor_model.get_layer('splirent_cho_dense').set_weights(saved_model.get_layer('dense_5').get_weights())
		predictor_model.get_layer('splirent_cho_dense').trainable = False

	def _load_predictor_func(sequence_input, sequence_class) :
		
		#Shared Model Definition (Applied to each randomized sequence region)
		conv_layer_1 = Conv1D(96, 8, padding='same', activation='relu', name='splirent_conv_1')
		pool_layer_1 = MaxPooling1D(pool_size=2)
		conv_layer_2 = Conv1D(128, 6, padding='same', activation='relu', name='splirent_conv_2')

		def shared_model(seq_input) :
			return Flatten()(
				conv_layer_2(
					pool_layer_1(
						conv_layer_1(
							seq_input
						)
					)
				)
			)

		seq_input_1 = Lambda(lambda x: x[:, 5:40, :, 0])(sequence_input)
		seq_input_2 = Lambda(lambda x: x[:, 48:83, :, 0])(sequence_input)

		shared_out_1 = shared_model(seq_input_1)
		shared_out_2 = shared_model(seq_input_2)

		#Layers applied to the concatenated hidden representation
		layer_dense = Dense(256, activation='relu', name='splirent_dense_1')
		layer_drop = Dropout(0.2)

		concat_out = Concatenate(axis=-1)([shared_out_1, shared_out_2])

		dense_out = layer_dense(concat_out)
		dropped_out = layer_drop(dense_out, training=False)

		#Final cell-line specific regression layers

		layer_cuts_hek = Dense(101, activation='softmax', kernel_initializer='zeros', name='splirent_hek_dense')
		layer_cuts_hela = Dense(101, activation='softmax', kernel_initializer='zeros', name='splirent_hela_dense')
		layer_cuts_mcf7 = Dense(101, activation='softmax', kernel_initializer='zeros', name='splirent_mcf7_dense')
		layer_cuts_cho = Dense(101, activation='softmax', kernel_initializer='zeros', name='splirent_cho_dense')

		pred_usage_hek = layer_cuts_hek(dropped_out)
		pred_usage_hela = layer_cuts_hela(dropped_out)
		pred_usage_mcf7 = layer_cuts_mcf7(dropped_out)
		pred_usage_cho = layer_cuts_cho(dropped_out)
		
		predictor_inputs = []
		predictor_outputs = [pred_usage_hek, pred_usage_hela, pred_usage_mcf7, pred_usage_cho]

		return predictor_inputs, predictor_outputs, _initialize_predictor_weights

	return _load_predictor_func
