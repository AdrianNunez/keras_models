import numpy as np
# Always fix the randomness seed value
from numpy import seed
seed(7)
from keras.optimizers import Adam
import json
from cnnm2048 import two_stream_network

def train(parameters):
    # Load parameters
    learning_rate = parameters['learning_rate']
    metrics = parameters['metrics']
    
    model = two_stream_network(parameters)

    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0005)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=metrics[1:])
    
    
    
if __name__ == '__main__':
    with open('parameters.json') as f:
        parameters = json.load(f)
    train(parameters)