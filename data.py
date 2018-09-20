import numpy as np
# Always fix the randomness seed value
from numpy import seed
seed(7)

def load_images(mode, parameters):
    for actor in actors:
        dic[actor] = dict()
    class_dict = get_classes()
    stacks_per_class = dict()
    folders, labels = [], []
    with open(data_file, 'r') as f:
        content = f.readlines()
        for i in range(len(content)):
            folder, label = content[i].strip().split(' ')
            folders.append(folder)
            class_name = folder[folder.find('/')+1:folder.rfind('/')]
            labels.append(class_dict[class_name])
            x_frames = glob.glob(data_folder + folder + '/flow_x*')  
            # Compute number of stacks inside current folder
            rest = len(x_frames) % L
            instances = (len(x_frames) - rest) // L
            if rest > 0: instances += 1
            #stacks_per_class[class_dict[label]] += instances
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    indices = sss.split(folders, labels)
    val_index = indices.next()[1]
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
    nb_total_stacks = 0
    
    for folder, label in zip(X, Y):
        x_frames = glob.glob(folder + '/flow_x*')
        # Count the number of stacks inside a folder/video
        rest = len(x_frames) % L
        nb_stacks = len(x_frames) // L
        if rest > 0:
            nb_stacks += 1
        nb_total_stacks += nb_stacks
        
        temp = folder[:folder.rfind('/')]
        class_name = temp[temp.rfind('/')+1:]
        # Store the folders by class
        if not folders_in_class.has_key(class_name):  
            folders_in_class[class_name] = []
        folders_in_class[class_name].append(folder)
        # Store the number of stacks per class (for data replication in training time)
        if not cnt.has_key(class_name):  
            cnt[class_name] = nb_stacks
        else:
            cnt[class_name] += nb_stacks
        # Store the class of a given folder name
        class_of_folder[folder] = label
    
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
   
        # p contains a class index, randomized by perm
        for p in np.asarray(folders_in_class.keys())[list(perm)]:         
            folders = folders_in_class[p]
            stacks_of_class = []
            labels_of_class = []
            for element in folders:
                x_frames = glob.glob(element + '/flow_x*')
                y_frames = glob.glob(element + '/flow_y*')       
                rest = len(x_frames) % L
                for r in xrange((len(x_frames) - rest) // L):        
                    # Variables used for random transformations, necessary to do them here to apply the same transformation to the whole stack
                    dx = np.random.randint(0, resize_shape[0]-image_shape[0])
                    dy = np.random.randint(0, resize_shape[1]-image_shape[1])
                    rand = np.random.rand(1)
                    
                    i = 0
                    if mode == 'train':
                        flow = np.zeros(shape=resize_shape + (2*L,), dtype=np.float32)
                    else:
                        flow = np.zeros(shape=image_shape + (2*L,), dtype=np.float32)
                    for flow_x_file, flow_y_file in zip(x_frames[r*L:(r+1)*L],y_frames[r*L:(r+1)*L]):
                        img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
                        img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
                        # Resize from original size to resize_shape, then do a random crop
                        if mode == 'train':
                            img_x = imresize(img_x, resize_shape, interp='bilinear') 
                            img_y = imresize(img_y, resize_shape, interp='bilinear') 
                        #img_x = img_x[dx:dx+image_shape[0], dy:dy+image_shape[1]]
                        #img_y = img_y[dx:dx+image_shape[0], dy:dy+image_shape[1]]
                        # Random horizontal mirroring
                        #if mode == 'train' and rand > 0.5:
                        #    img_x = 255 - img_x[:, ::-1]
                        #    img_y = img_y[:, ::-1]
                        
                        flow[:,:,2*i] = img_x
                        flow[:,:,2*i+1] = img_y
                        i += 1
                    stacks_of_class.append(flow)
                    labels_of_class.append(class_of_folder[element])
                    
                # Create a stack with the remaining frames, as they are not enough to create an stack pick the last L optical flow images
                if rest > 0:
                     # Variables used for random transformations, necessary to do them here to apply the same transformation to the whole stack
                    dx = np.random.randint(0, resize_shape[0]-image_shape[0])
                    dy = np.random.randint(0, resize_shape[1]-image_shape[1])
                    rand = np.random.rand(1)
                    
                    i = 0
                    if mode == 'train':
                        flow = np.zeros(shape=resize_shape + (2*L,), dtype=np.float32)
                    else:
                        flow = np.zeros(shape=image_shape + (2*L,), dtype=np.float32)
                    for flow_x_file, flow_y_file in zip(x_frames[-L:],y_frames[-L:]):
                        img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
                        img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
                        if mode == 'train':
                            img_x = imresize(img_x, resize_shape, interp='bilinear') 
                            img_y = imresize(img_y, resize_shape, interp='bilinear')
                        # Random crop
                        #img_x = img_x[dx:dx+image_shape[0], dy:dy+image_shape[1]]
                        #img_y = img_y[dx:dx+image_shape[0], dy:dy+image_shape[1]]
                        # Random mirror
                        #if mode == 'train' and rand > 0.5:
                        #    img_x = 255 - img_x[:, ::-1]
                        #    img_y = img_y[:, ::-1]

                        flow[:,:,2*i] = img_x
                        flow[:,:,2*i+1] = img_y
                        i += 1
                    stacks_of_class.append(flow)
                    labels_of_class.append(class_of_folder[element])
            # Data replication: repeat data to get the amount of data in the class with maximum number of samples
            if mode == 'train' and data_replication:
                temp_x = stacks_of_class
                temp_y = labels_of_class
                if replicate == 'max':
                # Need to achieve cnt[max_class] samples
                    while len(stacks_of_class) < cnt[max_class]:    
                        for _x, _y in zip(temp_x, temp_y):
                            stacks_of_class.append(_x)
                            labels_of_class.append(_y)
                            if len(stacks_of_class) >= cnt[max_class]:
                                break
                elif replicate == 'median':
                    select = np.random.choice(len(temp_x), size=median, replace=True)
                    stacks_of_class = []
                    for s in select:
                        stacks_of_class.append(temp_x[s])
                    labels_of_class = []
                    for s in select:
                        labels_of_class.append(temp_y[s])
                del temp_x, temp_y
                gc.collect()
            
            # Copies data to the batches and batch_labels arrays
            for elem in stacks_of_class:
                batches.append(elem)
            for elem in labels_of_class:
                batch_labels.append(elem)
            del stacks_of_class, labels_of_class
            gc.collect()
        
        stack_size = len(batches)
        num_batches = stack_size // batch_size
        rest = stack_size % batch_size
            
        if mode == 'train':
            # For selecting horizontal components and vertical components of the optical flow stack
            evens = [n for n in range(20) if n % 2 == 0]
            #odds = [n for n in range(20) if n % 2 != 0]
            print('train loaded: {}'.format(stack_size))
            while True:
                perm = np.random.permutation(range(stack_size))
                _x, _y = [], []
                batches_sent = 0
                for b in perm:
                    _x.append(batches[b])
                    dx = np.random.randint(0, resize_shape[0]-image_shape[0])
                    dy = np.random.randint(0, resize_shape[1]-image_shape[1])
                    rand = np.random.rand(1) 
                    _x[-1] = _x[-1][dx:dx+image_shape[0], dy:dy+image_shape[1],:]
                    if rand > 0.5:
                        _x[-1][...,evens] = 255 - _x[-1][...,evens]
                    #    _x[-1][:,:,odds] = _x[-1][:,::-1,odds]
                    _y.append(batch_labels[b])
                    if len(_x) == batch_size:
                        yield np.asarray(_x) - mean, np.asarray(to_categorical(_y, num_classes))
                        del _x, _y
                        gc.collect()
                        _x, _y = [], []
                
                rest = len(perm) % batch_size
                if rest > 0:
                    temp = np.random.choice(stack_size, batch_size-rest)
                    for i in temp:
                        _x.append(batches[i])
                        dx = np.random.randint(0, resize_shape[0]-image_shape[0])
                        dy = np.random.randint(0, resize_shape[1]-image_shape[1])
                        rand = np.random.rand(1) 
                        # Random croppping
                        _x[-1] = _x[-1][dx:dx+image_shape[0], dy:dy+image_shape[1],:]
                        # Random flipping
                        if rand > 0.5:
                            _x[-1][...,evens] = 255 - _x[-1][...,evens]
                        #    _x[-1][:,:,evens] = 255 - _x[-1][:,::-1,evens]
                        #    _x[-1][:,:,odds] = _x[-1][:,::-1,odds]
                        _y.append(batch_labels[i])
                    yield np.asarray(_x) - mean, np.asarray(to_categorical(_y, num_classes))
                    del _x, _y
                    gc.collect()
        # For the training and validation cases data augmentation is not used
        else:
            while True:
                print('{} loaded: {}'.format(mode, stack_size))
                rest = 0
                if stack_size%batch_size > 0:
                    rest = 1
                for b in range(num_batches+rest):
                    up_lim = min((b+1)*batch_size, stack_size)
                    #_x = []
                    #for s in range(up_lim-(b*batch_size)):
                        #dx = np.random.randint(0, resize_shape[0]-image_shape[0])
                        #dy = np.random.randint(0, resize_shape[1]-image_shape[1])
                        # Random croppping
                        #_x.append(batches[b*batch_size+s][dx:dx+image_shape[0], dy:dy+image_shape[1],:])
                    #yield np.asarray(_x) - mean, np.asarray(to_categorical(batch_labels[b*batch_size:up_lim], num_classes))#, num_batches, int(nb_total_stacks/batch_size), nb_total_stacks, total_stacks_used
                    yield np.asarray(batches[b*batch_size:up_lim]) - mean, np.asarray(to_categorical(batch_labels[b*batch_size:up_lim], num_classes))    