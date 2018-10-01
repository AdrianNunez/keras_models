import os
import cv2
import re
import sys 
from collections import Counter

def getKey(item):
    return item['start']
    
def tweakLabel(text):
    """ 
    This function transforms the input parameter text's content to fit the
    format in the annotations. It removes unnecessary objects such as counter,
    groups objects such as knife and cupPlateBowl, etc.
    """
    label = text.strip()
    action = label[1:-1].replace('><','_').replace(',','_').strip()
    for elem in ['cup','bowl','plate']:
        if elem in action:
            #pos = action.find('plate_container')
            #if pos == -1:
            action = action.replace(elem, 'cupPlateBowl',1)
            #else:
            #    action = action.replace(elem[:pos], 'cupPlateBowl') + action[pos:]
            break
    for elem in ['spoon','fork']:
        if elem in action:
            action = action.replace(elem, 'knife',1)
            break
    action = action.replace('turn on', 'turn-on')
    action = action.replace('turn off', 'turn-off')
    return action

dataset_folder = 'Gaze+/'
classes_file = 'gaze+TrainTestlist/classInd.txt'
videos_folder = 'videos/'
labels_folder = 'labels/'
output_path = 'gaze_plus_images/'
actornames = ['Ahmad', 'Rahul', 'Carlos', 'Alireza', 'Yin', 'Shaghayegh']
frame_width, frame_height = 320, 240

if not os.path.exists(output_path):
    os.mkdir(output_path)

# Load classes' names
classes =  []
#num_label = 0
with open(classes_file, 'r') as class_file:
    text = class_file.readline()[:-1]
    temp = text.split(',')
    for t in temp:
        c = t.strip().replace(' ', '').replace('\t', '')
        classes.append(c) 
        #num_label += 1
        
actor_appears = {} 
nb_clips = 0
num_object = 0
object_dictionary = {}
activity_dictionary = {}

actors = {}
for actorname in actornames:
    actors[actorname] = Counter()
    
    
# Load the dataset's annotations: start and end frame from each video, the
# list of objects and its corresponding action
folders_of_labels = [f for f in os.listdir(dataset_folder + labels_folder) if os.path.isdir(os.path.join(dataset_folder + labels_folder, f))]
folders_of_labels.sort()
# American, Burger, Greek, Pasta, Pizza, Snack, Turkey
for folder_of_labels in folders_of_labels:
    #print(folder_of_labels)
    path = dataset_folder + labels_folder + folder_of_labels + '/'
    label_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    label_files.sort()  # Ahmad_American.txt, Alireza_Ahmerican.txt ...

    activity_dictionary[folder_of_labels] = {}
    actor_appears[folder_of_labels] = []
    #i = 0
    for label_file in label_files:
        label_name = label_file[:-4]
        actor = label_name[:label_name.index('_')]
        activity_dictionary[folder_of_labels][label_name] = [] # American, Ahmad_American
        lines = [line.rstrip('\n') for line in open(path + label_file, 'r')]
        for line in lines:
            line = re.sub('[{}!@#$]', '', line)
            items = line.split(' (')
            label = items[0]
            time = items[1]            
            action = tweakLabel(label)
            if action in classes:
                nb_clips += 1
                actors[actor][action] += 1
                start = int(time[:time.find('-')])
                end = int(time[time.find('-')+1:][:-1])
                objects = label[label[1:].find('<')+2:][:-1].split(',')
                obj_codes = []
                for obj in objects:
                    if obj.strip() not in object_dictionary:
                        object_dictionary[obj] = num_object
                        num_object += 1
                    obj_codes.append(object_dictionary[obj])
                activity_dictionary[folder_of_labels][label_name].append({'actionlabel': action, 'start': start, 'end': end, 'objects': obj, 'from_activity': label_file[:-4]})

num_labels = len(classes)
num_objects = len(object_dictionary)
print('Number of actions: %d' % num_labels)
print('Number of subclips (total): %d' % nb_clips)            
dic = dict(zip(classes, range(0,len(classes))))

# Trim the video to get subsequences of actions
folders_of_videos = [f for f in os.listdir(dataset_folder + videos_folder) if os.path.isdir(os.path.join(dataset_folder + videos_folder, f))]
folders_of_videos.sort()
for folder_of_videos in folders_of_videos:
    print('Folder: ' + folder_of_videos + ' -----------------------------------------------------------------------------')
    path = dataset_folder + videos_folder + folder_of_videos + '/'
    videos = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    videos.sort()
    for video in videos:
        actor = video[:video.find('_')]
        print('-- Processing: ' + path + video)
        capture = cv2.VideoCapture(path + video)
        count = 0      
        j = 0
        actions = sorted(activity_dictionary[folder_of_videos][video[:-4]], key=getKey)
        for action in actions:           
            if not action['actionlabel'] in classes: continue
            prefix = output_path + actor + '/' + action['actionlabel'] + '/{}_{}_{}'.format(action['from_activity'], action['start'], action['end'])
            if not os.path.exists(prefix): os.makedirs(prefix)
            else: continue
            # Set the position of the video to the start of the action
            capture.set(cv2.CAP_PROP_POS_FRAMES, action['start'])
            i = 0
            while int(capture.get(cv2.CAP_PROP_POS_FRAMES)) < action['end']:
                ret, frame = capture.read()
                if ret == True:
                    save_path = prefix + '/frame%.5d.jpeg' % i
                    cv2.imwrite(save_path, cv2.resize(frame, (224,224)), [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                    i += 1
        capture.release()            
            
    print('------- ' + folder_of_videos + ' processed. -----------')