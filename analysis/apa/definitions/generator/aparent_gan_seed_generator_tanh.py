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


#GENESIS Generator Model definitions
def get_load_generator_network_func(gan_func) :
	
	def load_generator_network(batch_size, sequence_class, n_classes=1, seq_length=205, supply_inputs=False, gan_func=gan_func) :

		sequence_class_onehots = np.eye(n_classes)

		#Generator network parameters
		latent_size = 100
		out_seed_size = 100

		#Generator inputs
		latent_input_1 = Input(tensor=K.ones((batch_size, latent_size)), name='noise_input_1')
		latent_input_2 = Input(tensor=K.ones((batch_size, latent_size)), name='noise_input_2')
		latent_input_1_out = Lambda(lambda inp: inp * K.random_uniform((batch_size, latent_size), minval=-1.0, maxval=1.0), name='lambda_rand_input_1')(latent_input_1)
		latent_input_2_out = Lambda(lambda inp: inp * K.random_uniform((batch_size, latent_size), minval=-1.0, maxval=1.0), name='lambda_rand_input_2')(latent_input_2)

		class_embedding = Lambda(lambda x: K.gather(K.constant(sequence_class_onehots), K.cast(x[:, 0], dtype='int32')))(sequence_class)

		seed_input_1 = Concatenate(axis=-1)([latent_input_1_out, class_embedding])
		seed_input_2 = Concatenate(axis=-1)([latent_input_2_out, class_embedding])


		#Policy network definition
		policy_dense_0 = Dense(128, activation='linear', kernel_initializer='glorot_uniform', name='policy_dense_0')
		batch_norm_0 = BatchNormalization(name='policy_batch_norm_0')
		relu_0 = Lambda(lambda x: K.relu(x))

		policy_dense_1 = Dense(128, activation='linear', kernel_initializer='glorot_uniform', name='policy_dense_1')
		batch_norm_1 = BatchNormalization(name='policy_batch_norm_1')
		relu_1 = Lambda(lambda x: K.relu(x))

		policy_dense_2 = Dense(out_seed_size, activation='linear', kernel_initializer='glorot_uniform', name='policy_dense_2')
		batch_norm_2 = BatchNormalization(name='policy_batch_norm_2')
		tanh_2 = Lambda(lambda x: K.tanh(x))

		seed_out_1 = tanh_2(batch_norm_2(policy_dense_2(relu_1(batch_norm_1(policy_dense_1(relu_0(batch_norm_0(policy_dense_0(seed_input_1)))))))))
		seed_out_2 = tanh_2(batch_norm_2(policy_dense_2(relu_1(batch_norm_1(policy_dense_1(relu_0(batch_norm_0(policy_dense_0(seed_input_2)))))))))

		policy_out_1 = gan_func(seed_out_1)
		policy_out_2 = gan_func(seed_out_2)

		return [latent_input_1, latent_input_2], [policy_out_1, policy_out_2], [seed_out_1, seed_out_2]
	
	return load_generator_network
