import sys
import gc
from numpy.random import seed
seed(7)
import numpy as np
import glob
from utils import get_classes
from sklearn.model_selection import StratifiedShuffleSplit
import cv2
from collections import Counter
from keras.utils import to_categorical
import scipy.io as sio

def load_inputs(mode, test_subject, parameters, val_index=None):
    subjects = parameters['subjects']
    data_files_folder = parameters['data_files_folder']
    images_folder = parameters['images_folder']
    of_folder = parameters['of_folder']
    val_size = parameters['percentage_of_train_for_validation']
    classes_file = parameters['classes_file']
    L = parameters['L']
    of_mean = parameters['of_mean']
    image_mean = sio.loadmat(parameters['image_mean'])['image_mean']
    image_shape = (parameters['width'], parameters['height'])
    
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
    for i in range(len(folders)):
        folder = folders[i]
        class_name = folder[folder.find('/')+1:folder.rfind('/')]
        actor = folder[:folder.find('/')]
        # Only include the elements with index inside val_index for the validation set
        if mode == 'val':
            if i in val_index:
                X.append(folder)
                Y.append(int(labels[i]))
        # Do not include elements of the validation set in the training set
        elif mode == 'train':
            if not i in val_index:
                X.append(folder)
                Y.append(int(labels[i]))
        elif mode == 'test':
            X.append(folder)
            Y.append(int(labels[i]))
    del folders, labels
    gc.collect()
   
    #cnt = Counter()
    folders_in_class = dict()
    class_of_folder = dict()
    #nb_total_images = 0
        
    for folder, label in zip(X, Y):
        #frames = glob.glob(folder + '/frame*')
        # Count the number of images inside a folder/video
        #nb_images = len(frames)
        #nb_total_images += nb_images
        
        temp = folder[:folder.rfind('/')]
        class_name = temp[temp.rfind('/')+1:]

        # Store the folders by class
        if not folders_in_class.has_key(class_name):  
            folders_in_class[class_name] = []
        folders_in_class[class_name].append(folder)
        # Store the number of stacks per class (for data replication in training time)
        #if not cnt.has_key(class_name):  
        #    cnt[class_name] = nb_images
        #else:
        #    cnt[class_name] += nb_images
        # Store the class of a given folder name
        class_of_folder[folder] = label
  
    # In order to get the same amount of samples in each class, replicate data
    #if mode == 'train':
    #    if replicate == 'max':
    #        max_class = cnt.most_common()[0][0]
    #while True:     
        # perm is used to randomize the order of the classes
    perm = np.random.permutation(len(folders_in_class.keys()))
    batches_images, batches_stacks, batches_labels, inputs_per_video, video_names = [], [], [], [],[]
    #batch_labels_video = []
    #video_durations = []
    nb_videos = 0
    
    # p contains a class index, randomized by perm
    for p in np.asarray(folders_in_class.keys())[list(perm)]:         
        folders = folders_in_class[p]
        for element in folders:  
            video_names.append(element)
            inputs_per_video.append(nb_videos)
            # LOAD IMAGES
            frames = glob.glob(images_folder + element + '/frame*')
            
            if len(frames) < L:
                reps = int(L/len(frames))
                temp = frames
                frames = []
                for i in range(len(temp)):
                    for _ in xrange(reps):
                        frames.append(temp[i])

            temp = []
            for i in xrange(len(frames)):        
                img = cv2.imread(frames[i])
                temp.append(img - image_mean)
            batches_images.append(temp)
            #labels_by_video.append(class_of_folder[element])
            
            # LOAD OPTICAL FLOW
            x_frames = glob.glob(of_folder + element + '/flow_x*')
            y_frames = glob.glob(of_folder + element + '/flow_y*')
            if len(frames) < L:
                reps = int(L/len(x_frames))
                temp_x, temp_y = x_frames, y_frames
                x_frames, y_frames = [], []
                for i in range(len(temp_x)):
                    for _ in xrange(reps):
                        x_frames.append(temp_x[i])
                        y_frames.append(temp_y[i])
            rest = len(x_frames) % L
            add = 0
            if rest > 0:
                add = 1
            temp = []
            for r in xrange((len(x_frames) // L) + add):        
                low, high = r*L, (r+1)*L
                if high > len(x_frames):
                    low, high = -L, len(x_frames)
                i = 0
                flow = np.zeros(shape=image_shape + (2*L,), dtype=np.float32)
                for flow_x_file, flow_y_file in zip(x_frames[low:high],y_frames[low:high]):
                    img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
                    img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
                    flow[:,:,2*i] = img_x
                    flow[:,:,2*i+1] = img_y
                    i += 1
                temp.append(flow - of_mean)
                nb_videos += 1
            batches_stacks.append(temp)
            batches_labels.append(class_of_folder[element])
            assert len(temp) == ((len(x_frames) // L)+add)
                    
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
        
    #stack_size = len(batches)
    #num_batches = stack_size // batch_size
    #rest = stack_size % batch_size
    #class_labels = class_dict.values()
    #class_labels.sort()
    return {'images': batches_images, 'stacks': batches_stacks, 
            'labels': batches_labels, 'inputs_per_video': inputs_per_video,
            'val_index': val_index, 'video_names': video_names}
    
def load_train_val_image_dataset(parameters, test_subject):
    training_set = load_inputs('train', test_subject, parameters)
    validation_set = load_inputs(
        'val', test_subject, parameters, val_index=training_set['val_index']
    )
    assert len(training_set['images']) == len(training_set['labels']), (
        'Number of inputs should be equal to the number of labels in '
        'training set but found: {} inputs and {} labels'.format(
            len(training_set['images']), len(training_set['labels'])
        )
    )
    assert len(validation_set['images']) == len(validation_set['labels']), (
        'Number of inputs should be equal to the number of labels in '
        'validation set but found: {} inputs and {} labels'.format(
        len(validation_set['images']), len(validation_set['labels']))
    )
    return training_set, validation_set
    
def load_test_image_dataset(parameters, test_subject):
    test_set = load_inputs('test', test_subject, parameters)
    assert len(test_set['images']) == len(test_set['labels']), (
        'Number of inputs should be equal to the number of labels in '
        'test set but found: {} inputs and {} labels'.format(
        len(test_set['images']), len(test_set['labels']))
    )   
    return test_set

def batch_generator(mode, parameters, dataset):
    nb_inputs = dataset['inputs_per_video'][-1]
    nb_classes = parameters['nb_classes']
    L = parameters['L']
    batch_size = parameters['batch_size']
    
    while True:
        perm = np.random.permutation(nb_inputs)
        batch_images, batch_ofs, batch_labels = [], [], []
        for i in xrange(nb_inputs): 
            pos = np.searchsorted(
                dataset['inputs_per_video'], perm[i], side='left'
            )
            if dataset['inputs_per_video'][pos] > perm[i]:
                pos -= 1
            pos_stack = perm[i]-dataset['inputs_per_video'][pos]
            
            if pos_stack*L+L >= len(dataset['images'][pos]):
                if mode == 'train':
                    inner_pos = len(dataset['images'][pos]) - 1 - np.random.randint(0, L)
                else:
                    inner_pos = len(dataset['images'][pos]) - 1 - 5
            else:
                if mode == 'train':
                    inner_pos = np.random.randint(pos_stack*L,pos_stack*L+L)
                else:
                    inner_pos = pos_stack*L + L/2 
           
            batch_images.append(dataset['images'][pos][inner_pos])
            batch_ofs.append(dataset['stacks'][pos][pos_stack])
            batch_labels.append(
                to_categorical(dataset['labels'][pos], nb_classes)
            )
            if len(batch_images) == batch_size:
                yield (np.asarray(batch_images), np.asarray(batch_ofs),
                       np.asarray(batch_labels))
                batch_images, batch_ofs, batch_labels = [], [], []
        if mode == 'val' or mode == 'test':
            if len(batch_images) > 0:
                yield (np.asarray(batch_images), np.asarray(batch_ofs),
                       np.asarray(batch_labels))
                batch_images, batch_ofs, batch_labels = [], [], []
        elif mode == 'train' and nb_inputs % batch_size > 0:
            indices = np.random.choice(
                range(nb_inputs), nb_inputs % batch_size
            )
            for i in indices:
                pos = np.searchsorted(
                    dataset['inputs_per_video'], i, side='left'
                )
                if dataset['inputs_per_video'][pos] > i:
                    pos -= 1
                pos_stack = i-dataset['inputs_per_video'][pos]
                if pos_stack*L+L >= len(dataset['images'][pos]):
                    if mode == 'train':
                        inner_pos = len(dataset['images'][pos]) - 1 - np.random.randint(0, L)
                    else:
                        inner_pos = len(dataset['images'][pos]) - 1 - 5
                else:
                    if mode == 'train':
                        inner_pos = np.random.randint(pos_stack*L,pos_stack*L+L)
                    else:
                        inner_pos = pos_stack*L + L/2 
                
                batch_images.append(dataset['images'][pos][inner_pos])
                batch_ofs.append(dataset['stacks'][pos][pos_stack])
                batch_labels.append(
                    to_categorical(dataset['labels'][pos], nb_classes)
                )
                if len(batch_images) == batch_size:
                    yield (np.asarray(batch_images), np.asarray(batch_ofs),
                           np.asarray(batch_labels))
                    batch_images, batch_ofs, batch_labels = [], [], []
            
            
            
    