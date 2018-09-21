import sys
import gc
import numpy as np
# Always fix the randomness seed value
from numpy import seed
seed(7)
import glob
from utils import get_classes
from sklearn.model_selection import StratifiedShuffleSplit
import cv2
from collections import Counter

def load_images(mode, test_subject, parameters, val_index=None):
    subjects = parameters['subjects']
    data_files_folder = parameters['data_files_folder']
    images_folder = parameters['images_folder']
    val_size = parameters['percentage_of_train_for_validation']
    classes_file = parameters['classes_file']
    
    dic = dict()
    for actor in subjects:
        dic[actor] = dict()
    
    if mode == 'train' or mode == 'val':
        data_file = data_files_folder + '/train_{}.txt'.format(test_subject)
    elif mode == 'test':
        data_file = data_files_folder + '/test_{}.txt'.format(test_subject)
    else:
        print('No valid mode to load data. Options: train, val, test.')
        sys.exit()
        
    # Load video folders and their respective classes to memory
    folders, labels = [], []
    class_dict = get_classes(classes_file)
    with open(data_file, 'r') as f:
        content = f.readlines()
        for i in range(len(content)):
            folder, label = content[i].strip().split(' ')
            folders.append(folder)
            class_name = folder[folder.find('/')+1:folder.rfind('/')]
            labels.append(class_dict[class_name])
       
    # Stratify the dataset in train to get a validation partition:
    # i.e., get a subset of the training set with the same class distsribution
    if mode == 'train':
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=0)
        indices = sss.split(folders, labels)
        val_index = indices.next()[1]
    
    #
    X, Y = [], []
    for i in range(folders):
        class_name = folder[folder.find('/')+1:folder.rfind('/')]
        actor = folder[:folder.find('/')]
        # Only include the elements with index inside val_index for the validation set
        if mode == 'val':
            if i in val_index:
                X.append(images_folder + folders[i])
                Y.append(int(labels[i]))
        # Do not include elements of the validation set in the training set
        elif mode == 'train':
            if not i in val_index:
                X.append(images_folder + folders[i])
                Y.append(int(labels[i]))
        elif mode == 'test':
            X.append(images_folder + folders[i])
            Y.append(int(labels[i]))
    del folders, labels
    gc.collect()
   
    cnt = Counter()
    folders_in_class = dict()
    class_of_folder = dict()
    nb_total_images = 0
        
    for folder, label in zip(X, Y):
        frames = glob.glob(folder + '/frame*')
        # Count the number of images inside a folder/video
        nb_images = len(frames)
        nb_total_images += nb_images
        
        temp = folder[:folder.rfind('/')]
        class_name = temp[temp.rfind('/')+1:]

        # Store the folders by class
        if not folders_in_class.has_key(class_name):  
            folders_in_class[class_name] = []
        folders_in_class[class_name].append(folder)
        # Store the number of stacks per class (for data replication in training time)
        if not cnt.has_key(class_name):  
            cnt[class_name] = nb_images
        else:
            cnt[class_name] += nb_images
        # Store the class of a given folder name
        class_of_folder[folder] = label
  
    # In order to get the same amount of samples in each class, replicate data
    #if mode == 'train':
    #    if replicate == 'max':
    #        max_class = cnt.most_common()[0][0]
    #while True:     
        # perm is used to randomize the order of the classes
    perm = np.random.permutation(len(cnt.keys()))
    batches, batch_labels = [], []
    #batch_labels_video = []
    #video_durations = []
    
    # p contains a class index, randomized by perm
    for p in np.asarray(folders_in_class.keys())[list(perm)]:         
        folders = folders_in_class[p]
        images_of_class, labels_of_class, labels_by_video = [], [], []
        for element in folders:
            frames = glob.glob(element + '/frame*')
            for i in xrange(len(frames)):        
                img = cv2.imread(frames[i])
                images_of_class.append(img)
                labels_of_class.append(class_of_folder[element])
            labels_by_video.append(class_of_folder[element])
            #video_durations.append(len(frames))
        # Data replication: repeat data to get the amount of data in the class with maximum number of samples
        #if mode == 'train' and data_replication:
        #    temp_x = images_of_class
        #    temp_y = labels_of_class
            #if replicate == 'max':
            # Need to achieve cnt[max_class] samples
            #    while len(images_of_class) < cnt[max_class]:    
            #        for _x, _y in zip(temp_x, temp_y):
            #            images_of_class.append(_x)
            #            labels_of_class.append(_y)
            #            if len(images_of_class) >= cnt[max_class]:
            #                break
            #elif replicate == 'median':
            #    select = np.random.choice(len(temp_x), size=median, replace=True)
            #    images_of_class = []
            #    labels_of_class = []
            #    for s in select:
            #        images_of_class.append(temp_x[s])
            #    for s in select:
            #        labels_of_class.append(temp_y[s])
            #del temp_x, temp_y
            #gc.collect()
        
        # Copies data to the batches and batch_labels arrays
        for elem in images_of_class:
            batches.append(elem)
        for elem in labels_of_class:
            batch_labels.append(elem)
        #for elem in labels_by_video:
        #    batch_labels_video.append(elem)
        del images_of_class, labels_of_class
        gc.collect()
    #stack_size = len(batches)
    #num_batches = stack_size // batch_size
    #rest = stack_size % batch_size
    #class_labels = class_dict.values()
    #class_labels.sort()
    return batches, batch_labels, val_index
    
def load_image_dataset(parameters, test_subject):
    training_set_images, training_set_labels, val_index = load_images('train', test_subject, parameters)
    validation_set_images, validation_set_labels, _ = load_images('train', test_subject, parameters, val_index=val_index)
    test_set_images, test_set_labels, _ = load_images('train', test_subject, parameters)
    print(len(training_set_images), len(training_set_labels))
    print(len(validation_set_images), len(validation_set_labels))
    print(len(test_set_images), len(test_set_labels))
    return (training_set_images, training_set_labels), (validation_set_images, validation_set_labels), (test_set_images, test_set_labels)