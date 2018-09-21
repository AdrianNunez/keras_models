import numpy as np
# Always fix the randomness seed value
from numpy import seed
seed(7)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import json
from sklearn.metrics import confusion_matrix

from cnnm2048 import two_stream_network
from data import load_image_dataset
from utils import get_classes, calculate_evaluation_metrics, plot_training_info

def train(test_subject, parameters):
    # Load parameters
    learning_rate = parameters['learning_rate']
    metrics = parameters['metrics']
    batch_size = parameters['batch_size']
    nb_epoch = parameters['nb_epoch']
    saved_weights_file = parameters['saved_weights_file_path']
    plot_folder = parameters['plots_folder']
    
    # Load the network
    model = two_stream_network(parameters)
    
    # Load the optimizer and compile the model
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0005)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=metrics[1:])
    
    # Load the dataset
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_image_dataset(parameters, test_subject)
    
    # Get the necessary callbacks to train the model
    earlystopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0)    
    modelcheckpoint = ModelCheckpoint(saved_weights_file, monitor='val_loss', save_best_only=True, verbose=0)
    callbacks = [earlystopping, modelcheckpoint]
    
    # Train the model
    history = model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(x_val,y_val), callbacks=callbacks)
    
    print('Validation accuracy:', max(history.history['val_acc']))
    print('Validation loss:', min(history.history['val_loss']))

    model.load_weights(saved_weights_file)
    yp = model.predict(x_test, batch_size=batch_size, verbose=1)
    ypreds = np.argmax(yp, axis=1)
    ytrue = np.argmax(y_test, axis=1)
    
    cm = confusion_matrix(ytrue, ypreds)
    title = 'Confusion matrix in test set ({} fold)'.format(test_subject)
    cm_path = '{}cm_{}.pdf'.format(plot_folder, test_subject)
    plot_confusion_matrix(cm, classes, path, normalize=True, title=title, cmap='coolwarm', font_size=5)    
    
    metrics = calculate_evaluation_metrics(ytrue, ypreds)
    print "Scikit metrics"
    print 'accuracy: ', metrics['acc']
    print 'precision:', metrics['precision']
    print 'recall:', metrics['recall']
    print 'f1:', metrics['f1'] 
    
    plot_training_info(test_subject, metrics, save=True, history.history)

if __name__ == '__main__':
    with open('parameters.json') as f:
        parameters = json.load(f)
    
    # Here you should do cross validation in order to evaluate your model
    # Create a loop to reproduce the same experiment with different test subjects
    test_subject = 'Ahmad'
    train(test_subject, parameters)