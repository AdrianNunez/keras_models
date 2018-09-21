import sys
import numpy as np
# Always fix the randomness seed value
from numpy import seed
seed(7)
import glob
from utils import get_classes
from sklearn.model_selection import StratifiedShuffleSplit

def load_images(mode, test_subject, parameters):
    subjects = parameters['subjects']
    data_files_folder = parameters['data_files_folder']
    images_folder = parameters['images_folder']
    
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
    class_dict = get_classes()
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
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        indices = sss.split(folders, labels)
        val_index = indices.next()[1]
        
    for folder, label in zip(folders, labels):
        class_name = folder[folder.find('/')+1:folder.rfind('/')]
        actor = folder[:folder.find('/')]
    
    
    # Load the data depending on the mode parameter
    with open(data_file, 'r') as f:
        content = f.readlines()
        perm = range(len(content))
        if mode != 'test' and randomize_val_set and len(_perm) == 0:    
            perm = np.random.permutation(len(content))
        elif mode != 'test' and randomize_val_set:
            perm = _perm
        for i in perm:
            folder, label = content[i].strip().split(' ')
            class_name = folder[folder.find('/')+1:folder.rfind('/')]
            actor = folder[:folder.find('/')]
            if mode == 'train' or mode == 'val':
                if use_validation and not stratified_val and actor != 'Yin' and actor != 'Shaghayegh' and (dic[actor].get(class_name) == None or dic[actor].get(class_name) < action_per_actor_for_val):
                    if not dic[actor].has_key(class_name): dic[actor][class_name] = 1
                    else: dic[actor][class_name] += 1
                    #(not dic[actor].has_key(class_name) or dic[actor][class_name] < 2):
                    #if not dic[actor].has_key(class_name): dic[actor][class_name] = 0
                    #else: dic[actor][class_name] += 1
                    #1dic[actor][class_name] = folder
                    if mode == 'val':
                        X.append(data_folder + folder)
                        Y.append(int(label))
                elif use_validation and stratified_val and i in val_index:
                    if mode == 'val':
                        X.append(data_folder + folder)
                        Y.append(int(label))
                else:   
                    if mode == 'train':
                        X.append(data_folder + folder)
                        Y.append(int(label))
            elif mode == 'test':
                X.append(data_folder + folder)
                Y.append(int(label))

    cnt = Counter()
    folders_in_class = dict()
    class_of_folder = dict()
    nb_total_images = 0
    
    class_names = get_classes_names()
    durations_difficult_classes = dict()
    for c in [19,21,22,31,39,42]:
        durations_difficult_classes[class_names[c-1]] = []
    rest = dict()
    for folder, label in zip(X, Y):
        frames = glob.glob(folder + '/frame*')
        # Count the number of images inside a folder/video
        nb_images = len(frames)
        nb_total_images += nb_images
        
        temp = folder[:folder.rfind('/')]
        class_name = temp[temp.rfind('/')+1:]
        if class_name in durations_difficult_classes:
            durations_difficult_classes[class_name].append(nb_images)
        if class_name not in rest:
            rest[class_name] = []
        rest[class_name].append(nb_images)
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
        
    #for c in [19,21,22,31,39,42]:
    #    name = class_names[c-1]
    #    print(name, len(durations_difficult_classes[name]), np.mean(durations_difficult_classes[name]), np.std(durations_difficult_classes[name]))
    video_summary = dict()
    for k in rest.keys():
        video_summary[k] = [len(rest[k]), np.mean(rest[k]), np.std(rest[k])]
    
    values = cnt.values()
    values.sort()
    median = values[len(values)/2]
    # In order to get the same amount of samples in each class, replicate data
    if mode == 'train':
        if replicate == 'max':
            max_class = cnt.most_common()[0][0]
    while True:     
        # perm is used to randomize the order of the classes
        perm = np.random.permutation(len(cnt.keys()))
        batches = []
        batch_labels = []
        batch_labels_video = []
        video_durations = []
        
        # p contains a class index, randomized by perm
        for p in np.asarray(folders_in_class.keys())[list(perm)]:         
            folders = folders_in_class[p]
            images_of_class = []
            labels_of_class = []
            labels_by_video = []
            for element in folders:
                frames = glob.glob(element + '/frame*')
                for i in xrange(len(frames)):        
                    img = cv2.imread(frames[i])
                        # Resize from original size to resize_shape, then do a random crop
                    if mode == 'train':
                        img = imresize(img, resize_shape, interp='bilinear') 
                        #img_x = img_x[dx:dx+image_shape[0], dy:dy+image_shape[1]]
                        #img_y = img_y[dx:dx+image_shape[0], dy:dy+image_shape[1]]
                        # Random horizontal mirroring
                        #if mode == 'train' and rand > 0.5:
                        #    img_x = 255 - img_x[:, ::-1]
                        #    img_y = img_y[:, ::-1]
                    images_of_class.append(img)
                    labels_of_class.append(class_of_folder[element])
                labels_by_video.append(class_of_folder[element])
                video_durations.append(len(frames))
            # Data replication: repeat data to get the amount of data in the class with maximum number of samples
            if mode == 'train' and data_replication:
                temp_x = images_of_class
                temp_y = labels_of_class
                if replicate == 'max':
                # Need to achieve cnt[max_class] samples
                    while len(images_of_class) < cnt[max_class]:    
                        for _x, _y in zip(temp_x, temp_y):
                            images_of_class.append(_x)
                            labels_of_class.append(_y)
                            if len(images_of_class) >= cnt[max_class]:
                                break
                elif replicate == 'median':
                    select = np.random.choice(len(temp_x), size=median, replace=True)
                    images_of_class = []
                    labels_of_class = []
                    for s in select:
                        images_of_class.append(temp_x[s])
                    for s in select:
                        labels_of_class.append(temp_y[s])
                del temp_x, temp_y
                gc.collect()
            
            # Copies data to the batches and batch_labels arrays
            for elem in images_of_class:
                batches.append(elem)
            for elem in labels_of_class:
                batch_labels.append(elem)
            for elem in labels_by_video:
                batch_labels_video.append(elem)
            del images_of_class, labels_of_class
            gc.collect()
        stack_size = len(batches)
        num_batches = stack_size // batch_size
        rest = stack_size % batch_size
        class_labels = class_dict.values()
        class_labels.sort()