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

#GENESIS Predictor helper functions


#GENESIS Predictor Model definitions

#Predictor that predicts the function of the generated input sequence
def build_predictor(generator_model, load_predictor_function, batch_size, n_samples=None, eval_mode='pwm') :

	use_samples = True
	if n_samples is None :
		use_samples = False
		n_samples = 1

	#Get PWM outputs from Generator Model
	sequence_class = generator_model.outputs[0]
	pwm_logits = generator_model.outputs[1]
	pwm = generator_model.outputs[3]
	sampled_pwm = generator_model.outputs[5]


	seq_input = None
	class_input = sequence_class
	if eval_mode == 'pwm' :
		seq_input = pwm
	elif eval_mode == 'sample' :
		seq_input = sampled_pwm
		if use_samples :
			seq_input = Lambda(lambda x: K.reshape(x, (K.shape(x)[0] * K.shape(x)[1], K.shape(x)[2], K.shape(x)[3], K.shape(x)[4])))(seq_input)
			class_input = Lambda(lambda x: K.tile(x[0], (K.shape(x[1])[1], 1)))([class_input, sampled_pwm])

	predictor_inputs, predictor_outputs, post_compile_function = load_predictor_function(seq_input, class_input)
	
	
	#Optionally create sample axis
	if use_samples :
		predictor_outputs = [
			Lambda(lambda x: K.reshape(x, (batch_size, n_samples, K.shape(x)[-1])))(predictor_output)
			#Lambda(lambda x: K.reshape(x, (n_samples, batch_size, K.shape(x)[-1])))(predictor_output)
			for predictor_output in predictor_outputs
		]

	predictor_model = Model(
		inputs = generator_model.inputs + predictor_inputs,
		outputs = generator_model.outputs + predictor_outputs
	)

	post_compile_function(predictor_model)

	#Lock all layers except policy layers
	for predictor_layer in predictor_model.layers :
		predictor_layer.trainable = False
		
		if 'policy' in predictor_layer.name :
			predictor_layer.trainable = True

	return 'genesis_predictor', predictor_model

#Predictor that predicts the function of the generated input sequence
def build_predictor_w_adversary(generator_model, load_predictor_function, batch_size, n_samples=None, eval_mode='pwm') :

	use_samples = True
	if n_samples is None :
		use_samples = False
		n_samples = 1

	#Get PWM outputs from Generator Model
	sequence_class = generator_model.outputs[0]
	pwm = generator_model.outputs[3]
	sampled_pwm = generator_model.outputs[5]
	pwm_adv = generator_model.outputs[4]
	sampled_pwm_adv = generator_model.outputs[6]


	seq_input = None
	seq_input_adv = None
	class_input = sequence_class
	if eval_mode == 'pwm' :
		seq_input = pwm
		seq_input_adv = pwm_adv
	elif eval_mode == 'sample' :
		seq_input = sampled_pwm
		seq_input_adv = sampled_pwm_adv
		if use_samples :
			seq_input = Lambda(lambda x: K.reshape(x, (K.shape(x)[0] * K.shape(x)[1], K.shape(x)[2], K.shape(x)[3], K.shape(x)[4])))(seq_input)
			seq_input_adv = Lambda(lambda x: K.reshape(x, (K.shape(x)[0] * K.shape(x)[1], K.shape(x)[2], K.shape(x)[3], K.shape(x)[4])))(seq_input_adv)
			class_input = Lambda(lambda x: K.tile(x[0], (K.shape(x[1])[1], 1)))([class_input, sampled_pwm])

	predictor_inputs, shared_callables, predictor_function, post_compile_function = load_predictor_function()
	
	shared_inputs = [shared_callable(class_input) for shared_callable in shared_callables]
	
	predictor_outputs = predictor_function(seq_input, class_input, predictor_inputs, shared_inputs)
	predictor_outputs_adv = predictor_function(seq_input_adv, class_input, predictor_inputs, shared_inputs)
	
	#Optionally create sample axis
	if use_samples :
		predictor_outputs = [
			#Lambda(lambda x: K.reshape(x, (batch_size, n_samples, K.shape(x)[-1])))(predictor_output)
			Lambda(lambda x: K.reshape(x, [batch_size, n_samples] + list(K.int_shape(x))[1:]))(predictor_output)
			for predictor_output in predictor_outputs
		]
		predictor_outputs_adv = [
			#Lambda(lambda x: K.reshape(x, (batch_size, n_samples, K.shape(x)[-1])))(predictor_output)
			Lambda(lambda x: K.reshape(x, [batch_size, n_samples] + list(K.int_shape(x))[1:]))(predictor_output)
			for predictor_output in predictor_outputs_adv
		]

	predictor_model = Model(
		inputs = generator_model.inputs + predictor_inputs,
		outputs = generator_model.outputs + predictor_outputs + predictor_outputs_adv
	)

	post_compile_function(predictor_model)

	#Lock all layers except policy layers
	for predictor_layer in predictor_model.layers :
		predictor_layer.trainable = False
		
		if 'policy' in predictor_layer.name :
			predictor_layer.trainable = True

	return 'genesis_predictor', predictor_model
