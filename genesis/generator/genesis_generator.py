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
from tensorflow.python.framework import ops

import isolearn.keras as iso

import numpy as np

#Stochastic Binarized Neuron helper functions (Tensorflow)
#ST Estimator code adopted from https://r2rt.com/beyond-binary-ternary-and-one-hot-neurons.html
#See Github https://github.com/spitis/

def st_sampled_softmax(logits):
	with ops.name_scope("STSampledSoftmax") as namescope :
		nt_probs = tf.nn.softmax(logits)
		onehot_dim = logits.get_shape().as_list()[1]
		sampled_onehot = tf.one_hot(tf.squeeze(tf.multinomial(logits, 1), 1), onehot_dim, 1.0, 0.0)
		with tf.get_default_graph().gradient_override_map({'Ceil': 'Identity', 'Mul': 'STMul'}):
			return tf.ceil(sampled_onehot * nt_probs)

def st_hardmax_softmax(logits):
	with ops.name_scope("STHardmaxSoftmax") as namescope :
		nt_probs = tf.nn.softmax(logits)
		onehot_dim = logits.get_shape().as_list()[1]
		sampled_onehot = tf.one_hot(tf.argmax(nt_probs, 1), onehot_dim, 1.0, 0.0)
		with tf.get_default_graph().gradient_override_map({'Ceil': 'Identity', 'Mul': 'STMul'}):
			return tf.ceil(sampled_onehot * nt_probs)

@ops.RegisterGradient("STMul")
def st_mul(op, grad):
	return [grad, grad]

#PWM Masking and Sampling helper functions

def mask_pwm(inputs) :
	pwm, onehot_template, onehot_mask = inputs

	return pwm * onehot_mask + onehot_template

def sample_pwm_only(pwm_logits) :
	#n_sequences = pwm_logits.get_shape().as_list()[0]
	#seq_length = pwm_logits.get_shape().as_list()[1]
	n_sequences = K.shape(pwm_logits)[0]
	seq_length = K.shape(pwm_logits)[1]
	
	flat_pwm = K.reshape(pwm_logits, (n_sequences * seq_length, 4))
	sampled_pwm = st_sampled_softmax(flat_pwm)

	return K.reshape(sampled_pwm, (n_sequences, seq_length, 4, 1))

def sample_pwm(pwm_logits) :
	#n_sequences = pwm_logits.get_shape().as_list()[0]
	#seq_length = pwm_logits.get_shape().as_list()[1]
	n_sequences = K.shape(pwm_logits)[0]
	seq_length = K.shape(pwm_logits)[1]
	
	flat_pwm = K.reshape(pwm_logits, (n_sequences * seq_length, 4))
	sampled_pwm = K.switch(K.learning_phase(), st_sampled_softmax(flat_pwm), st_hardmax_softmax(flat_pwm))

	return K.reshape(sampled_pwm, (n_sequences, seq_length, 4, 1))

def max_pwm(pwm_logits) :
	#n_sequences = pwm_logits.get_shape().as_list()[0]
	#seq_length = pwm_logits.get_shape().as_list()[1]
	n_sequences = K.shape(pwm_logits)[0]
	seq_length = K.shape(pwm_logits)[1]
	
	flat_pwm = K.reshape(pwm_logits, (n_sequences * seq_length, 4))
	sampled_pwm = st_hardmax_softmax(flat_pwm)

	return K.reshape(sampled_pwm, (n_sequences, seq_length, 4, 1))


#GENESIS helper functions

def initialize_sequence_templates(generator, sequence_templates) :

	embedding_templates = []
	embedding_masks = []

	for k in range(len(sequence_templates)) :
		sequence_template = sequence_templates[k]
		onehot_template = iso.OneHotEncoder(seq_length=len(sequence_template))(sequence_template).reshape((len(sequence_template), 4, 1))
		
		for j in range(len(sequence_template)) :
			if sequence_template[j] not in ['N', 'X'] :
				nt_ix = np.argmax(onehot_template[j, :, 0])
				onehot_template[j, :, :] = -4.0
				onehot_template[j, nt_ix, :] = 10.0
			elif sequence_template[j] == 'X' :
				onehot_template[j, :, :] = -1.0

		onehot_mask = np.zeros((len(sequence_template), 4, 1))
		for j in range(len(sequence_template)) :
			if sequence_template[j] == 'N' :
				onehot_mask[j, :, :] = 1.0
		
		embedding_templates.append(onehot_template.reshape(1, -1))
		embedding_masks.append(onehot_mask.reshape(1, -1))

	embedding_templates = np.concatenate(embedding_templates, axis=0)
	embedding_masks = np.concatenate(embedding_masks, axis=0)

	generator.get_layer('template_dense').set_weights([embedding_templates])
	generator.get_layer('template_dense').trainable = False

	generator.get_layer('mask_dense').set_weights([embedding_masks])
	generator.get_layer('mask_dense').trainable = False


#Generator construction function
def build_generator(batch_size, seq_length, load_generator_function, n_classes=1, n_samples=None, sequence_templates=None, batch_normalize_pwm=False, anneal_pwm_logits=False, validation_sample_mode='max', supply_inputs=False) :

	use_samples = True
	if n_samples is None :
		use_samples = False
		n_samples = 1

	sequence_class_input, sequence_class = None, None
	#Seed class input for all dense/embedding layers
	if not supply_inputs :
		sequence_class_input = Input(tensor=K.ones((batch_size, 1)), dtype='int32', name='sequence_class_seed')
		sequence_class = Lambda(lambda inp: K.cast(K.round(inp * K.random_uniform((batch_size, 1), minval=-0.4999, maxval=n_classes-0.5001)), dtype='int32'), name='lambda_rand_sequence_class')(sequence_class_input)
	else :
		sequence_class_input = Input(batch_shape=(batch_size, 1), dtype='int32', name='sequence_class_seed')
		sequence_class = Lambda(lambda inp: inp, name='lambda_rand_sequence_class')(sequence_class_input)


	#Get generated policy pwm logits (non-masked)
	generator_inputs, [raw_logits_1, raw_logits_2], extra_outputs = load_generator_function(batch_size, sequence_class, n_classes=n_classes, seq_length=seq_length, supply_inputs=supply_inputs)

	reshape_layer = Reshape((seq_length, 4, 1))
	
	onehot_template_dense = Embedding(n_classes, seq_length * 4, embeddings_initializer='zeros', name='template_dense')
	onehot_mask_dense = Embedding(n_classes, seq_length * 4, embeddings_initializer='ones', name='mask_dense')
	
	onehot_template = reshape_layer(onehot_template_dense(sequence_class))
	onehot_mask = reshape_layer(onehot_mask_dense(sequence_class))



	#Initialize Templating and Masking Lambda layer
	masking_layer = Lambda(mask_pwm, output_shape = (seq_length, 4, 1), name='masking_layer')

	#Batch Normalize PWM Logits
	if batch_normalize_pwm :
		raw_logit_batch_norm = BatchNormalization(name='policy_raw_logit_batch_norm')
		raw_logits_1 = raw_logit_batch_norm(raw_logits_1)
		raw_logits_2 = raw_logit_batch_norm(raw_logits_2)
	
	#Add Template and Multiply Mask
	pwm_logits_1 = masking_layer([raw_logits_1, onehot_template, onehot_mask])
	pwm_logits_2 = masking_layer([raw_logits_2, onehot_template, onehot_mask])
	
	#Compute PWMs (Nucleotide-wise Softmax)
	pwm_1 = Softmax(axis=-2, name='pwm_1')(pwm_logits_1)
	pwm_2 = Softmax(axis=-2, name='pwm_2')(pwm_logits_2)
	
	anneal_temp = None
	if anneal_pwm_logits :
		anneal_temp = K.variable(1.0)
		
		interpolated_pwm_1 = Lambda(lambda x: (1. - anneal_temp) * x + anneal_temp * 0.25)(pwm_1)
		interpolated_pwm_2 = Lambda(lambda x: (1. - anneal_temp) * x + anneal_temp * 0.25)(pwm_2)
		
		pwm_logits_1 = Lambda(lambda x: K.log(x / (1. - x)))(interpolated_pwm_1)
		pwm_logits_2 = Lambda(lambda x: K.log(x / (1. - x)))(interpolated_pwm_2)
	
	#Sample proper One-hot coded sequences from PWMs
	sampled_pwm_1, sampled_pwm_2, sampled_onehot_mask = None, None, None

	sample_func = sample_pwm
	if validation_sample_mode == 'sample' :
		sample_func = sample_pwm_only

	#Optionally tile each PWM to sample from and create sample axis
	if use_samples :
		pwm_logits_upsampled_1 = Lambda(lambda x: K.tile(x, [n_samples, 1, 1, 1]))(pwm_logits_1)
		pwm_logits_upsampled_2 = Lambda(lambda x: K.tile(x, [n_samples, 1, 1, 1]))(pwm_logits_2)
		sampled_onehot_mask = Lambda(lambda x: K.tile(x, [n_samples, 1, 1, 1]))(onehot_mask)

		sampled_pwm_1 = Lambda(sample_func, name='pwm_sampler_1')(pwm_logits_upsampled_1)
		#sampled_pwm_1 = Lambda(lambda x: K.reshape(x, (n_samples, batch_size, seq_length, 4, 1)))(sampled_pwm_1)
		sampled_pwm_1 = Lambda(lambda x: K.permute_dimensions(K.reshape(x, (n_samples, batch_size, seq_length, 4, 1)), (1, 0, 2, 3, 4)))(sampled_pwm_1)

		sampled_pwm_2 = Lambda(sample_func, name='pwm_sampler_2')(pwm_logits_upsampled_2)
		#sampled_pwm_2 = Lambda(lambda x: K.reshape(x, (n_samples, batch_size, seq_length, 4, 1)))(sampled_pwm_2)
		sampled_pwm_2 = Lambda(lambda x: K.permute_dimensions(K.reshape(x, (n_samples, batch_size, seq_length, 4, 1)), (1, 0, 2, 3, 4)))(sampled_pwm_2)

		
		#sampled_onehot_mask = Lambda(lambda x: K.reshape(x, (n_samples, batch_size, seq_length, 4, 1)), (1, 0, 2, 3, 4))(sampled_onehot_mask)
		sampled_onehot_mask = Lambda(lambda x: K.permute_dimensions(K.reshape(x, (n_samples, batch_size, seq_length, 4, 1)), (1, 0, 2, 3, 4)))(sampled_onehot_mask)

	else :
		sampled_pwm_1 = Lambda(sample_func, name='pwm_sampler_1')(pwm_logits_1)
		sampled_pwm_2 = Lambda(sample_func, name='pwm_sampler_2')(pwm_logits_2)
		sampled_onehot_mask = onehot_mask
	
	
	generator_model = Model(
		inputs=[
			sequence_class_input
		] + generator_inputs,
		outputs=[
			sequence_class,
			pwm_logits_1,
			pwm_logits_2,
			pwm_1,
			pwm_2,
			sampled_pwm_1,
			sampled_pwm_2

			,onehot_mask
			,sampled_onehot_mask
		] + extra_outputs
	)

	if sequence_templates is not None :
		initialize_sequence_templates(generator_model, sequence_templates)

	#Lock all generator layers except policy layers
	for generator_layer in generator_model.layers :
		generator_layer.trainable = False
		
		if 'policy' in generator_layer.name :
			generator_layer.trainable = True

	if anneal_pwm_logits :
		return 'genesis_generator', generator_model, anneal_temp
	return 'genesis_generator', generator_model


#Generator copy function
def get_generator_copier(master_generator, copy_number=1) :
	
	def copy_generator(batch_size, seq_length, load_generator_function, n_classes=1, n_samples=None, sequence_templates=None, batch_normalize_pwm=False, anneal_pwm_logits=False, validation_sample_mode='max', supply_inputs=False, master_generator=master_generator, copy_number=copy_number) :

		use_samples = True
		if n_samples is None :
			use_samples = False
			n_samples = 1

		sequence_class_input, sequence_class = None, None
		#Seed class input for all dense/embedding layers
		if not supply_inputs :
			sequence_class_input = Input(tensor=K.ones((batch_size, 1)), dtype='int32', name='sequence_class_seed_copy_' + str(copy_number))
			sequence_class = Lambda(lambda inp: K.cast(K.round(inp * K.random_uniform((batch_size, 1), minval=-0.4999, maxval=n_classes-0.5001)), dtype='int32'), name='lambda_rand_sequence_class_copy_' + str(copy_number))(sequence_class_input)
		else :
			sequence_class_input = Input(batch_shape=(batch_size, 1), dtype='int32', name='sequence_class_seed_copy_' + str(copy_number))
			sequence_class = Lambda(lambda inp: inp, name='lambda_rand_sequence_class_copy_' + str(copy_number))(sequence_class_input)


		#Get generated policy pwm logits (non-masked)
		generator_inputs, [raw_logits_1, raw_logits_2], extra_outputs = load_generator_function(batch_size, sequence_class, n_classes=n_classes, seq_length=seq_length, supply_inputs=supply_inputs)

		reshape_layer = Reshape((seq_length, 4, 1))

		onehot_template_dense = master_generator.get_layer('template_dense')
		onehot_mask_dense = master_generator.get_layer('mask_dense')

		onehot_template = reshape_layer(onehot_template_dense(sequence_class))
		onehot_mask = reshape_layer(onehot_mask_dense(sequence_class))



		#Initialize Templating and Masking Lambda layer
		masking_layer = Lambda(mask_pwm, output_shape = (seq_length, 4, 1), name='masking_layer_copy_' + str(copy_number))

		#Batch Normalize PWM Logits
		if batch_normalize_pwm :
			raw_logit_batch_norm = master_generator.get_layer('policy_raw_logit_batch_norm')
			raw_logits_1 = raw_logit_batch_norm(raw_logits_1, training=True)
			raw_logits_2 = raw_logit_batch_norm(raw_logits_2, training=True)

		#Add Template and Multiply Mask
		pwm_logits_1 = masking_layer([raw_logits_1, onehot_template, onehot_mask])
		pwm_logits_2 = masking_layer([raw_logits_2, onehot_template, onehot_mask])

		#Compute PWMs (Nucleotide-wise Softmax)
		pwm_1 = Softmax(axis=-2, name='pwm_1')(pwm_logits_1)
		pwm_2 = Softmax(axis=-2, name='pwm_2')(pwm_logits_2)

		anneal_temp = None
		if anneal_pwm_logits :
			anneal_temp = K.variable(1.0)

			interpolated_pwm_1 = Lambda(lambda x: (1. - anneal_temp) * x + anneal_temp * 0.25)(pwm_1)
			interpolated_pwm_2 = Lambda(lambda x: (1. - anneal_temp) * x + anneal_temp * 0.25)(pwm_2)

			pwm_logits_1 = Lambda(lambda x: K.log(x / (1. - x)))(interpolated_pwm_1)
			pwm_logits_2 = Lambda(lambda x: K.log(x / (1. - x)))(interpolated_pwm_2)

		#Sample proper One-hot coded sequences from PWMs
		sampled_pwm_1, sampled_pwm_2, sampled_onehot_mask = None, None, None

		sample_func = sample_pwm
		if validation_sample_mode == 'sample' :
			sample_func = sample_pwm_only

		#Optionally tile each PWM to sample from and create sample axis
		if use_samples :
			pwm_logits_upsampled_1 = Lambda(lambda x: K.tile(x, [n_samples, 1, 1, 1]))(pwm_logits_1)
			pwm_logits_upsampled_2 = Lambda(lambda x: K.tile(x, [n_samples, 1, 1, 1]))(pwm_logits_2)
			sampled_onehot_mask = Lambda(lambda x: K.tile(x, [n_samples, 1, 1, 1]))(onehot_mask)

			sampled_pwm_1 = Lambda(sample_func, name='pwm_sampler_1_copy_' + str(copy_number))(pwm_logits_upsampled_1)
			#sampled_pwm_1 = Lambda(lambda x: K.reshape(x, (n_samples, batch_size, seq_length, 4, 1)))(sampled_pwm_1)
			sampled_pwm_1 = Lambda(lambda x: K.permute_dimensions(K.reshape(x, (n_samples, batch_size, seq_length, 4, 1)), (1, 0, 2, 3, 4)))(sampled_pwm_1)

			sampled_pwm_2 = Lambda(sample_func, name='pwm_sampler_2_copy_' + str(copy_number))(pwm_logits_upsampled_2)
			#sampled_pwm_2 = Lambda(lambda x: K.reshape(x, (n_samples, batch_size, seq_length, 4, 1)))(sampled_pwm_2)
			sampled_pwm_2 = Lambda(lambda x: K.permute_dimensions(K.reshape(x, (n_samples, batch_size, seq_length, 4, 1)), (1, 0, 2, 3, 4)))(sampled_pwm_2)


			#sampled_onehot_mask = Lambda(lambda x: K.reshape(x, (n_samples, batch_size, seq_length, 4, 1)), (1, 0, 2, 3, 4))(sampled_onehot_mask)
			sampled_onehot_mask = Lambda(lambda x: K.permute_dimensions(K.reshape(x, (n_samples, batch_size, seq_length, 4, 1)), (1, 0, 2, 3, 4)))(sampled_onehot_mask)

		else :
			sampled_pwm_1 = Lambda(sample_func, name='pwm_sampler_1_copy_' + str(copy_number))(pwm_logits_1)
			sampled_pwm_2 = Lambda(sample_func, name='pwm_sampler_2_copy_' + str(copy_number))(pwm_logits_2)
			sampled_onehot_mask = onehot_mask


		generator_model = Model(
			inputs=[
				sequence_class_input
			] + generator_inputs,
			outputs=[
				sequence_class,
				pwm_logits_1,
				pwm_logits_2,
				pwm_1,
				pwm_2,
				sampled_pwm_1,
				sampled_pwm_2

				,onehot_mask
				,sampled_onehot_mask
			] + extra_outputs
		)

		if sequence_templates is not None :
			initialize_sequence_templates(generator_model, sequence_templates)

		#Lock all generator layers except policy layers
		for generator_layer in generator_model.layers :
			generator_layer.trainable = False

			if 'policy' in generator_layer.name :
				generator_layer.trainable = True

		if anneal_pwm_logits :
			return 'genesis_generator', generator_model, anneal_temp
		return 'genesis_generator', generator_model
	
	return copy_generator


#(Re-)Initialize PWM weights
def reset_generator(generator_model) :
	session = K.get_session()
	for generator_layer in generator_model.layers :
		if 'policy' in generator_layer.name :
			for v in generator_layer.__dict__:
				v_arg = getattr(generator_layer, v)
				if hasattr(v_arg,'initializer'):
					initializer_method = getattr(v_arg, 'initializer')
					initializer_method.run(session=session)
					print('reinitializing layer {}.{}'.format(generator_layer.name, v))
