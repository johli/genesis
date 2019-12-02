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

def get_shallow_copy_function(master_generator, copy_number=1) :
	
	def copy_generator_network(batch_size, sequence_class, n_classes=1, seq_length=205, supply_inputs=False, master_generator=master_generator, copy_number=copy_number) :
		
		sequence_class_onehots = np.eye(n_classes)

		#Generator network parameters
		latent_size = 100

		#Generator inputs
		latent_input_1, latent_input_2, latent_input_1_out, latent_input_2_out = None, None, None, None
		if not supply_inputs :
			latent_input_1 = Input(tensor=K.ones((batch_size, latent_size)), name='noise_input_1')
			latent_input_2 = Input(tensor=K.ones((batch_size, latent_size)), name='noise_input_2')
			latent_input_1_out = Lambda(lambda inp: inp * K.random_uniform((batch_size, latent_size), minval=-1.0, maxval=1.0), name='lambda_rand_input_1')(latent_input_1)
			latent_input_2_out = Lambda(lambda inp: inp * K.random_uniform((batch_size, latent_size), minval=-1.0, maxval=1.0), name='lambda_rand_input_2')(latent_input_2)
		else :
			latent_input_1 = Input(batch_shape=(batch_size, latent_size), name='noise_input_1')
			latent_input_2 = Input(batch_shape=(batch_size, latent_size), name='noise_input_2')
			latent_input_1_out = Lambda(lambda inp: inp, name='lambda_rand_input_1')(latent_input_1)
			latent_input_2_out = Lambda(lambda inp: inp, name='lambda_rand_input_2')(latent_input_2)

		class_embedding = Lambda(lambda x: K.gather(K.constant(sequence_class_onehots), K.cast(x[:, 0], dtype='int32')))(sequence_class)

		seed_input_1 = Concatenate(axis=-1)([latent_input_1_out, class_embedding])
		seed_input_2 = Concatenate(axis=-1)([latent_input_2_out, class_embedding])


		#Policy network definition
		policy_dense_1 = master_generator.get_layer('policy_dense_1')

		policy_dense_1_reshape = Reshape((21, 1, 384))

		policy_deconv_0 = master_generator.get_layer('policy_deconv_0')

		policy_deconv_1 = master_generator.get_layer('policy_deconv_1')

		policy_deconv_2 = master_generator.get_layer('policy_deconv_2')

		policy_conv_3 = master_generator.get_layer('policy_conv_3')

		policy_conv_4 = master_generator.get_layer('policy_conv_4')

		policy_conv_5 = master_generator.get_layer('policy_conv_5')

		#policy_deconv_3 = Conv2DTranspose(4, (7, 1), strides=(1, 1), padding='valid', activation='linear', kernel_initializer='glorot_normal', name='policy_deconv_3')

		batch_norm_0 = master_generator.get_layer('policy_batch_norm_0')
		relu_0 = Lambda(lambda x: K.relu(x))
		batch_norm_1 = master_generator.get_layer('policy_batch_norm_1')
		relu_1 = Lambda(lambda x: K.relu(x))
		batch_norm_2 = master_generator.get_layer('policy_batch_norm_2')
		relu_2 = Lambda(lambda x: K.relu(x))

		batch_norm_3 = master_generator.get_layer('policy_batch_norm_3')
		relu_3 = Lambda(lambda x: K.relu(x))

		batch_norm_4 = master_generator.get_layer('policy_batch_norm_4')
		relu_4 = Lambda(lambda x: K.relu(x))

		policy_out_1 = Reshape((seq_length, 4, 1))(policy_conv_5(relu_4(batch_norm_4(policy_conv_4(relu_3(batch_norm_3(policy_conv_3(relu_2(batch_norm_2(policy_deconv_2(relu_1(batch_norm_1(policy_deconv_1(relu_0(batch_norm_0(policy_deconv_0(policy_dense_1_reshape(policy_dense_1(seed_input_1))), training=True))), training=True))), training=True))), training=True))), training=True))))
		policy_out_2 = Reshape((seq_length, 4, 1))(policy_conv_5(relu_4(batch_norm_4(policy_conv_4(relu_3(batch_norm_3(policy_conv_3(relu_2(batch_norm_2(policy_deconv_2(relu_1(batch_norm_1(policy_deconv_1(relu_0(batch_norm_0(policy_deconv_0(policy_dense_1_reshape(policy_dense_1(seed_input_2))), training=True))), training=True))), training=True))), training=True))), training=True))))

		return [latent_input_1, latent_input_2], [policy_out_1, policy_out_2], []
	
	return copy_generator_network

#GENESIS Generator Model definitions
def load_generator_network(batch_size, sequence_class, n_classes=1, seq_length=205, supply_inputs=False) :

	sequence_class_onehots = np.eye(n_classes)

	#Generator network parameters
	latent_size = 100
	
	#Generator inputs
	latent_input_1, latent_input_2, latent_input_1_out, latent_input_2_out = None, None, None, None
	if not supply_inputs :
		latent_input_1 = Input(tensor=K.ones((batch_size, latent_size)), name='noise_input_1')
		latent_input_2 = Input(tensor=K.ones((batch_size, latent_size)), name='noise_input_2')
		latent_input_1_out = Lambda(lambda inp: inp * K.random_uniform((batch_size, latent_size), minval=-1.0, maxval=1.0), name='lambda_rand_input_1')(latent_input_1)
		latent_input_2_out = Lambda(lambda inp: inp * K.random_uniform((batch_size, latent_size), minval=-1.0, maxval=1.0), name='lambda_rand_input_2')(latent_input_2)
	else :
		latent_input_1 = Input(batch_shape=K.ones(batch_size, latent_size), name='noise_input_1')
		latent_input_2 = Input(batch_shape=K.ones(batch_size, latent_size), name='noise_input_2')
		latent_input_1_out = Lambda(lambda inp: inp, name='lambda_rand_input_1')(latent_input_1)
		latent_input_2_out = Lambda(lambda inp: inp, name='lambda_rand_input_2')(latent_input_2)
	
	class_embedding = Lambda(lambda x: K.gather(K.constant(sequence_class_onehots), K.cast(x[:, 0], dtype='int32')))(sequence_class)

	seed_input_1 = Concatenate(axis=-1)([latent_input_1_out, class_embedding])
	seed_input_2 = Concatenate(axis=-1)([latent_input_2_out, class_embedding])
	
	
	#Policy network definition
	policy_dense_1 = Dense(21 * 384, activation='relu', kernel_initializer='glorot_uniform', name='policy_dense_1')
	
	policy_dense_1_reshape = Reshape((21, 1, 384))
	
	policy_deconv_0 = Conv2DTranspose(256, (7, 1), strides=(2, 1), padding='valid', activation='linear', kernel_initializer='glorot_normal', name='policy_deconv_0')
	
	policy_deconv_1 = Conv2DTranspose(192, (8, 1), strides=(2, 1), padding='valid', activation='linear', kernel_initializer='glorot_normal', name='policy_deconv_1')
	
	policy_deconv_2 = Conv2DTranspose(128, (7, 1), strides=(2, 1), padding='valid', activation='linear', kernel_initializer='glorot_normal', name='policy_deconv_2')
	
	policy_conv_3 = Conv2D(128, (8, 1), strides=(1, 1), padding='same', activation='linear', kernel_initializer='glorot_normal', name='policy_conv_3')

	policy_conv_4 = Conv2D(64, (8, 1), strides=(1, 1), padding='same', activation='linear', kernel_initializer='glorot_normal', name='policy_conv_4')

	policy_conv_5 = Conv2D(4, (8, 1), strides=(1, 1), padding='same', activation='linear', kernel_initializer='glorot_normal', name='policy_conv_5')

	#policy_deconv_3 = Conv2DTranspose(4, (7, 1), strides=(1, 1), padding='valid', activation='linear', kernel_initializer='glorot_normal', name='policy_deconv_3')
	
	batch_norm_0 = BatchNormalization(name='policy_batch_norm_0')
	relu_0 = Lambda(lambda x: K.relu(x))
	batch_norm_1 = BatchNormalization(name='policy_batch_norm_1')
	relu_1 = Lambda(lambda x: K.relu(x))
	batch_norm_2 = BatchNormalization(name='policy_batch_norm_2')
	relu_2 = Lambda(lambda x: K.relu(x))

	batch_norm_3 = BatchNormalization(name='policy_batch_norm_3')
	relu_3 = Lambda(lambda x: K.relu(x))

	batch_norm_4 = BatchNormalization(name='policy_batch_norm_4')
	relu_4 = Lambda(lambda x: K.relu(x))

	policy_out_1 = Reshape((seq_length, 4, 1))(policy_conv_5(relu_4(batch_norm_4(policy_conv_4(relu_3(batch_norm_3(policy_conv_3(relu_2(batch_norm_2(policy_deconv_2(relu_1(batch_norm_1(policy_deconv_1(relu_0(batch_norm_0(policy_deconv_0(policy_dense_1_reshape(policy_dense_1(seed_input_1)))))))))))))))))))
	policy_out_2 = Reshape((seq_length, 4, 1))(policy_conv_5(relu_4(batch_norm_4(policy_conv_4(relu_3(batch_norm_3(policy_conv_3(relu_2(batch_norm_2(policy_deconv_2(relu_1(batch_norm_1(policy_deconv_1(relu_0(batch_norm_0(policy_deconv_0(policy_dense_1_reshape(policy_dense_1(seed_input_2)))))))))))))))))))
	
	return [latent_input_1, latent_input_2], [policy_out_1, policy_out_2], []
