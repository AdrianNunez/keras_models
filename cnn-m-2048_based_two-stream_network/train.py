# Always fix the randomness seed value
from numpy.random import seed
seed(7)
import numpy as np
import os
import time
from keras.optimizers import Adam
import json
import gc
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from collections import Counter
from cnnm2048 import two_stream_network
from data import (load_train_val_image_dataset, load_test_image_dataset,
                  batch_generator)
from utils import (get_classes, calculate_evaluation_metrics,
                   plot_training_info, plot_confusion_matrix)

# Specify which GPU is used
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def train(test_subject, parameters):
    # Load parameters
    learning_rate = parameters['learning_rate']
    metrics = parameters['metrics']
    batch_size = parameters['batch_size']
    nb_epoch = parameters['nb_epoch']
    saved_weights_file = (parameters['saved_weights_file_path'] +
                            '_{}.h5'.format(test_subject))
    plot_folder = parameters['plots_folder']
    weights_folder = parameters['weights_folder']
    classes_file = parameters['classes_file']
    
    # Create any necessary folder
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    if not os.path.exists(weights_folder):
        os.makedirs(weights_folder)
    
    # Load the network
    model = two_stream_network(parameters)
    
    # Load the optimizer and compile the model
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999,
                epsilon=1e-08, decay=0.0005)
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=metrics[1:])

    if nb_epoch > 0:  
        # Load the dataset
        training_set, validation_set = load_train_val_image_dataset(
                                                    parameters, test_subject
                                        )
        nb_inputs_train = training_set['inputs_per_video'][-1]
        nb_batches_train = nb_inputs_train // batch_size
        if nb_inputs_train % batch_size > 0:
            nb_batches_train += 1
        nb_inputs_val = validation_set['inputs_per_video'][-1]
        nb_batches_val = nb_inputs_val // batch_size
        if nb_inputs_val % batch_size > 0:
            nb_batches_val += 1
        
        # Train the model (validate with validation set in each epoch) 
        best_loss, best_f1, best_epoch = 1e8, 0, 0
        losses = {'train': [], 'val': []}
        accuracies = {'train': [], 'val': []}
        for e in range(nb_epoch):
            next_batch_train = batch_generator('train', parameters, training_set)
            next_batch_val = batch_generator('val', parameters, validation_set)
            train_acc, train_loss = 0, 0
            # Training
            train_time = time.time()
            for b in range(nb_batches_train):
                image, ofstack, label = next_batch_train.next()
                loss, accuracy = model.train_on_batch([image, ofstack], label)
                train_acc += accuracy
                train_loss += loss
            
            losses['train'].append(float(train_loss)/float(nb_batches_train))
            accuracies['train'].append(float(train_acc)/float(nb_batches_train)) 
            train_time = time.time() - train_time
            
            preds, gt = np.zeros((nb_inputs_val)), np.zeros((nb_inputs_val))
            val_loss = 0
            #Validation
            val_time = time.time()
            for b in range(nb_batches_val):
                image, ofstack, label = next_batch_val.next()
                pred = model.predict([image, ofstack], batch_size=batch_size)
                gt[b*batch_size:b*batch_size+label.shape[0]] = np.argmax(label,1)
                preds[b*batch_size:b*batch_size+pred.shape[0]] = np.argmax(pred,1)
                loss, _ = model.test_on_batch([image, ofstack], label)
                val_loss += loss
            
            val_acc = accuracy_score(gt, preds)
            losses['val'].append(float(val_loss)/float(nb_batches_val))
            accuracies['val'].append(val_acc) 
            val_f1 = f1_score(gt, preds, average='macro')
            val_time = time.time() - val_time
            
            print('Epoch {} - Train Loss: {}, Train Acc: {}, Train time: {} s |||'
                  'Val Acc: {}, Val F1: {}, Val time: {} s'.format(
                          e, np.mean(losses['train']), 
                          np.mean(accuracies['train']),
                          train_time, val_acc, val_f1, val_time
                      )
                  )
                  
            # Plot training and validation loss and accuracy
            plot_training_info(
                test_subject, parameters, metrics, True, losses, accuracies
            )
            # Save weights of the model if a better loss is found
            if val_f1 >= best_f1:
                if losses['val'][-1] < best_loss:
                    best_epoch = e
                    best_loss = losses['val'][-1]
                    best_f1 = val_f1
                    model.save_weights(weights_folder + saved_weights_file)
                    print('Weights saved in {}'.format(weights_folder + saved_weights_file))
        
        del training_set, validation_set
        gc.collect()
        
        print('Best validation epoch: {} - Loss: {}, F1: {})'.format(best_epoch, best_loss, best_f1))

    # Load best model
    model.load_weights(weights_folder + saved_weights_file)
    print('Best weights loaded')

    # Load the test set
    test_set = load_test_image_dataset(parameters, test_subject)
    nb_inputs_test = test_set['inputs']
    nb_batches_test = nb_inputs_test // batch_size
    if nb_inputs_test % batch_size > 0:
        nb_batches_test += 1
        
    next_batch_test = batch_generator('test', parameters, test_set)
    preds, gt = np.zeros((test_set['inputs'])), np.zeros((test_set['inputs']))
    # Test
    test_time = time.time()
    for b in range(nb_inputs_test):
        images, ofstacks, label = next_batch_test.next()
        pred = model.predict([images, ofstacks], batch_size=batch_size)
        cnt = Counter(np.argmax(pred,1))
        gt[b*batch_size:b*batch_size+label.shape[0]] = np.argmax(label,0)
        preds[b*batch_size:b*batch_size+pred.shape[0]] = cnt.most_common(1)[0][0]
        
    test_time = time.time() - test_time
    print('Time to complete the test: {} seconds'.format(test_time))
    cm = confusion_matrix(gt, preds)
    title = 'Normalized confusion matrix in test set ({} fold)'.format(
        test_subject
    )
    cm_path = '{}cm_{}.pdf'.format(plot_folder, test_subject)
    classes = get_classes(classes_file)
    # Save the confusion matrix
    plot_confusion_matrix(
        cm, classes, cm_path, normalize=True, title=title, cmap='coolwarm',
        font_size=5
    )    
    
    metrics = calculate_evaluation_metrics(gt, preds)
    print "Scikit metrics"
    print 'accuracy: ', metrics['acc']
    print 'precision:', metrics['precision']
    print 'recall:', metrics['recall']
    print 'f1:', metrics['f1'] 

if __name__ == '__main__':
    with open('parameters.json') as f:
        parameters = json.load(f)
    
    # Here you should do cross validation in order to evaluate your model
    # Create a loop to reproduce the same experiment with different
    # test subjects
    test_subject = 'Ahmad'
    train(test_subject, parameters)