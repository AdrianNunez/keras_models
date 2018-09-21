import numpy as np
# Always fix the randomness seed value
from numpy import seed
seed(7)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import json
import gc
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
    saved_weights_file = parameters['saved_weights_file_path'] + '_{}.h5'.format(test_subject)
    plot_folder = parameters['plots_folder']
    
    # Load the network
    model = two_stream_network(parameters)
    
    # Load the optimizer and compile the model
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0005)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=metrics[1:])
    
    # Load the dataset
    training_set, validation_set = load_train_val_image_dataset(parameters, test_subject)
    nb_inputs_train = training_set['inputs_per_video'][-1]
    nb_batches_train = nb_inputs_train // batch_size
    if nb_inputs_train % batch_size > 0:
        nb_batches_train += 1
    
    # Get the necessary callbacks to train the model
    earlystopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0)    
    modelcheckpoint = ModelCheckpoint(saved_weights_file, monitor='val_loss', save_best_only=True, verbose=0)
    callbacks = [earlystopping, modelcheckpoint]
    
    # Train the model 
    for e in range(nb_epoch):
        next_batch_train = batch_generator('train', parameters, training_set)
        next_batch_val = batch_generator('val', parameters, validation_set)
        losses = {'train': [], 'val': []}
        accuracies = {'train': [], 'val': []}
        for b in range(nb_batches_train):
            image, ofstack, label = next_batch_train.next()
            loss, accuracy = model.train_on_batch([image, ofstack], label)
            losses['train'].append(loss)
            accuracies['train'].append(accuracy)      
        preds, gt = [], []
        loss_val, acc_val = [], []
        for b in range(nb_batches_val):
            image, ofstack, label = next_batch_val.next()
            pred = model.predict([image, ofstack], batch_size=batch_size)
            gt.append(label)
            preds.append(pred)
            loss, accuracy = model.trdy_on_batch([image, ofstack], label)
            losses['val'].append(loss)
            accuracies['val'].append(accuracy)   
        val_f1 = metrics.f1_score(
            np.argmax(gt,1), np.argmax(preds,1), average='macro'
        )
        val_acc = metrics.accuracy_score(np.argmax(gt,1), np.argmax(preds,1))
        print('Epoch {} - Train Loss: {}, Train Acc: {} / Val Acc: {},'
              'Val F1: {}'.format(
                  e, np.mean(loss_train), np.mean(acc_train), val_acc, val_f1
                  )
              )
        plot_training_info(test_subject, metrics, save=True,
                           losses, accuracies
                           )
    
    print('Validation accuracy:', max(history.history['val_acc']))
    print('Validation loss:', min(history.history['val_loss']))
    del training_set, validation_set
    gc.collect()

    # Load best model
    model.load_weights(saved_weights_file)

    # Load the test set and compute metrics and confusion matrix
    test_set = load_test_image_dataset(parameters, test_subject)
    next_batch_test = batch_generator('test', parameters, test_set)
    gt, preds = [], []
    for i in range(nb_batches_test):
        image, ofstack, label = next_batch_test.next()
        pred = model.predict([image, ofstack], batch_size=batch_size)
        gt.append(label)
        preds.append(pred)
    
    ytrue = np.argmax(gt,1)
    ypreds = np.argmax(preds,1)
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

if __name__ == '__main__':
    with open('parameters.json') as f:
        parameters = json.load(f)
    
    # Here you should do cross validation in order to evaluate your model
    # Create a loop to reproduce the same experiment with different test subjects
    test_subject = 'Ahmad'
    train(test_subject, parameters)