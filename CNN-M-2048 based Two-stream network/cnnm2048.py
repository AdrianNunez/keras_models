import gc
import numpy as np
# Always fix the randomness seed value
from numpy.random import seed
seed(7)
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Convolution2D, Lambda, MaxPooling2D, ZeroPadding2D, Flatten, Dense, Dropout, Activation, Concatenate

# This layer normalizes the outputs of a neural network layer. Similar to Batch Normalization layer but deprecated in use.
def lrn(input, radius=5, alpha=0.0005, beta=0.75, name='LRN', bias=1.0):
        return tf.nn.local_response_normalization(input, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name) 

def cnn_m_2048(name, input_shape, keep_prob_1, keep_prob_2=1., num_classes=None, get_feature_vector=True):
    inputs = Input(shape=input_shape)
    # BLOCK 1
    conv1 = Convolution2D(96, 7, 7, subsample=(2,2), border_mode='valid', activation='relu', name='conv1_{}'.format(name))(inputs)   
    lrn1 = Lambda(lrn)(conv1)
    pool1 = MaxPooling2D((3,3), strides=(2,2), border_mode='valid')(lrn1)
    
    # BLOCK 2
    padding2 = ZeroPadding2D(padding=(1,1))(pool1)
    conv2 = Convolution2D(256, 5, 5, subsample=(2,2), border_mode='valid', activation='relu', name='conv2_{}'.format(name))(padding2)
    lrn2 = Lambda(lrn)(conv2)
    pool2 = MaxPooling2D((3,3), strides=(2,2), border_mode='same')(lrn2)
    
    # BLOCK 3
    padding3 = ZeroPadding2D(padding=(1,1))(pool2)
    conv3 = Convolution2D(512, 3, 3, subsample=(1,1), border_mode='valid', activation='relu', name='conv3_{}'.format(name))(padding3)
    
    # BLOCK 4
    padding4 = ZeroPadding2D(padding=(1,1))(conv3)
    conv4 = Convolution2D(512, 3, 3, subsample=(1,1), border_mode='valid', activation='relu', name='conv4_{}'.format(name))(padding4)
    
    # BLOCK 5
    padding5 = ZeroPadding2D(padding=(1,1))(conv4)
    conv5 = Convolution2D(512, 3, 3, subsample=(1,1), border_mode='valid', activation='relu', name='conv5_{}'.format(name))(padding5)
    pool5 = MaxPooling2D((3,3), strides=(2,2), border_mode='valid', name='pool5')(conv5)

    flatten = Flatten()(pool5)
    
    # MULTILAYER PERCEPTRON (CLASSIFIER)
    dense1 = Dense(4096, activation='relu', name='fc6_{}'.format(name), init='glorot_uniform')(flatten)
    dropout1 = Dropout(keep_prob_1)(dense1)
    dense2 = Dense(2048, activation='relu', name='fc7_{}'.format(name), init='glorot_uniform')(dropout1)
    
    if get_feature_vector:
        return dense2
        
    dropout2 = Dropout(keep_prob_2)(dense2)
    logits = Dense(num_classes, bias=True, init='glorot_uniform', name='fc8_{}'.format(name))(dropout2)
    softmax = Activation('softmax')(logits)
    return Model(inputs=inputs, outputs=softmax)   
    
def load_npy_weights(model, name, weights_file, initialize_last_layer=False):
    # Add the Caffe weights (in .npy format) to the network
    data = np.load(weights_file).item()
    keys = data.keys()
    keys.sort()
    for key in keys:
        #if model.get_layer(name=key) == None: continue
        if not initialize_last_layer and key == 'fc8_{}'.format(name): continue
        
        w, b = data[key]['weights'], data[key]['biases']
        w = np.asarray(w, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        model.get_layer(name=key).set_weights((w,b))
        del w, b
        gc.collect()
    del data
    gc.collect()
    return model
    
def two_stream_network(parameters):
    dropout_spatialnet_1 = parameters['dropout_spatialnet_1']
    dropout_temporalnet_1 = parameters['dropout_temporalnet_1']
    input_shape_spatialnet = (parameters['width'], parameters['height'], parameters['channels'])
    input_shape_temporalnet = (parameters['width'], parameters['height'], 2*parameters['L'])
    nb_classes = parameters['nb_classes']
    imagenet_weights_path = parameters['imagenet_weights_path']
    ucf101_weights_path = parameters['ucf101_weights_path']
    
    # Load the each branch's feature vector (second fully connected layer)
    spatialnet = cnn_m_2048(name='spatialnet',
                            input_shape=input_shape_spatialnet, 
                            keep_prob_1=dropout_spatialnet_1)
    pre_trained_spatialnet = load_npy_weights(model=spatialnet, name='spatialnet', weights_file=imagenet_weights_path)
    
    temporalnet = cnn_m_2048(name='temporalnet',
                            input_shape=input_shape_temporalnet, 
                            keep_prob_1=dropout_temporalnet_1) 
    pre_trained_temporalnet = load_npy_weights(model=temporalnet, name='temporalnet', weights_file=ucf101_weights_path)
                            
    merged_network = Concatenate([pre_trained_spatialnet, pre_trained_temporalnet], axis=1)
    logits = Dense(nb_classes, name='logits', init='glorot_uniform')(merged_network)
    softmax = Activation('softmax')(logits)
    return Model(inputs=[pre_trained_spatialnet.input, pre_trained_temporalnet.input], outputs=softmax)