import os
import cv2
import re
import sys 
from collections import Counter

dataset_folder = '/home/adrian/Downloads/Gaze+/'
videos_folder = 'videos/'
labels_folder = 'labels/'

output_path = '/home/adrian/project/gaze_plus_images/'

if not os.path.exists(output_path):
    os.mkdir(output_path)
frame_width, frame_height = 320, 240

#actionlabel_to_id = {}

#activity_dictionary = {}
#cmu = Counter()
#activities = []
#num_activities = 0
#activity = {}
#num_label = 0

    
actor_appears = {}


classes =  []
num_label = 0
with open('/home/adrian/project/classes.txt', 'r') as class_file:
    text = class_file.readline()[:-1]
    temp = text.split(',')
    for t in temp:
        c = t.strip().replace(' ', '').replace('\t', '')
        classes.append(c) 
        #actionlabel_to_id[c] = num_label
        #action_dictionary2[c] = 0
        num_label += 1
        

def tweakLabel(text):
    label = text.strip()
    if True:
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
    if False:
        pos = label[1:].find('<')+2
        pos2 = label[pos:].find('>')+pos
        objects = label[pos:pos2].split(',')
        action = label[:label[1:].find('<')][1:]
    
        if action == 'turn on':
            action = 'turn-on'
        elif action == 'turn off':
            action = 'turn-off'
           
        if len(objects) == 0:
            return action
        new_objects = []
        for obj in objects:
            if obj in ['hands', 'counter', 'cap']:
                continue
            if 'olive' in obj:
                new_objects.append(obj[obj.find('olive')+len('olive')+1:])   
                continue
            if 'spatula' in obj:
                new_objects.append('plastic_spatula')
                continue
            if obj in ['cup','plate','bowl']:
                new_objects.append('cupPlateBowl')
                continue
            if obj in ['spoon','fork']:
                new_objects.append('knife')
                continue
            new_objects.append(obj)
       
        label = action + '_' + '_'.join(new_objects)
        return label
    
def getKey(item):
    return item['start']
    
if False:
    count_classes = Counter()
    for actor in actornames:
        count_classes[actor] = dict()
    actions_set = set()
    folders_of_labels.sort()
    for folder_of_labels in folders_of_labels:
        label_files = [f for f in os.listdir(dataset_folder + labels_folder + folder_of_labels) if os.path.isfile(os.path.join(dataset_folder + labels_folder + folder_of_labels, f))]
        print(label_files)
        label_files.sort()
        path = dataset_folder + labels_folder + folder_of_labels + '/'
        i = 0
        for label_file in label_files:
            label_name = label_file[:-4]
            actor = label_name[:label_name.index('_')]
            lines = [line.rstrip('\n') for line in open(path + label_file, 'r')]
            for line in lines:
                line = re.sub('[{}!@#$]', '', line)
                items = line.split(' (')
                label = items[0]
                time = items[1]            
                action = tweakLabel(label)
                if count_classes[actor].has_key(action):
                    count_classes[actor][action] += 1
                else:
                    count_classes[actor][action] = 1
                actions_set.add(action)

    print(actornames[:-2])
    classes = []
    for action_selected in actions_set:
        take = 1
        for actor_selected in actornames[:-2]:   
            if not count_classes[actor_selected].has_key(action_selected) or count_classes[actor_selected][action_selected] < 2:
                take = 0
                break
        if take == 1:
            classes.append(action_selected)
    
nb_clips = 0
num_object = 0
object_dictionary = {}
activity_dictionary = {}
actornames = ['Ahmad', 'Rahul', 'Carlos', 'Alireza', 'Yin', 'Shaghayegh']
actors = {}
for actorname in actornames:
    actors[actorname] = Counter()
    
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
        #actor_appears[folder_of_labels].append(actor)
        activity_dictionary[folder_of_labels][label_name] = [] # American, Ahmad_American
        #print(label_name, actor)

        lines = [line.rstrip('\n') for line in open(path + label_file, 'r')]
        for line in lines:
            line = re.sub('[{}!@#$]', '', line)
            items = line.split(' (')
            label = items[0]
            time = items[1]            
            action = tweakLabel(label)
            if action in classes:
                nb_clips += 1
                #action_dictionary2[action] += 1
                #if action not in action_dictionary:
                #    action_dictionary[action] = num_label
                #    num_label += 1
            #if cmu[action] == 0:
            #    cmu[action] += 1
            #    activities.append(action)
            #else:
            #    cmu[action] += 1

                actors[actor][action] += 1
                #activity_dictionary[folder_of_labels][label_name].append({})
                #activity_dictionary[folder_of_labels][label_name][-1]['actionlabel'] = action
                #print(action)
                start = int(time[:time.find('-')])
                end = int(time[time.find('-')+1:][:-1])
                #activity_dictionary[folder_of_labels][label_name][-1]['start'] = start
                #activity_dictionary[folder_of_labels][label_name][-1]['end'] = end
                
                objects = label[label[1:].find('<')+2:][:-1].split(',')
                obj_codes = []
                for obj in objects:
                    if obj.strip() not in object_dictionary:
                        object_dictionary[obj] = num_object
                        num_object += 1
                    obj_codes.append(object_dictionary[obj])
                activity_dictionary[folder_of_labels][label_name].append({'actionlabel': action, 'start': start, 'end': end, 'objects': obj, 'from_activity': label_file[:-4]})
        #i += 1
        #if i == len(label_files):
        #    sorted(activity_dictionary[folder_of_labels][label_name], key=getKey)
        #    print(folder_of_labels, label_name)
        #    print(activity_dictionary[folder_of_labels][label_name])
        #    sys.exit()

num_labels = len(classes)
num_objects = len(object_dictionary)
print('Number of actions: %d' % num_labels)
print('Number of subclips (total): %d' % nb_clips)            
dic = dict(zip(classes, range(0,len(classes))))

# PROCESS ALL VIDEO FOLDERS
#indices = dict(zip(classes, [0]*len(classes)))

folders_of_videos = [f for f in os.listdir(dataset_folder + videos_folder) if os.path.isdir(os.path.join(dataset_folder + videos_folder, f))]
folders_of_videos.sort()
for folder_of_videos in folders_of_videos:
    print('Folder: ' + folder_of_videos + ' -----------------------------------------------------------------------------')
    path = dataset_folder + videos_folder + folder_of_videos + '/'
    videos = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    videos.sort()
    # FOR EACH VIDEO
    for video in videos:
        #labelfile = open(output_path + labels_folder + folder_of_videos + '/' + video[:-4] + '/' + 'labels.txt', 'w+', 0)
        #objectfile = open(output_path + labels_folder + folder_of_videos + '/' + video[:-4] + '/' + 'objects.txt', 'w+', 0)
        actor = video[:video.find('_')]
        #if actor!='Ahmad': continue
        print('--Processing: ' + path + video)
        capture = cv2.VideoCapture(path + video)
    
        count = 0      
        j = 0

        actions = sorted(activity_dictionary[folder_of_videos][video[:-4]], key=getKey)
       
        #pos_frame = capture.get(cv2.CAP_PROP_POS_FRAMES)
        #num_frame = 0
        
        for action in actions:           
            if not action['actionlabel'] in classes: continue
            # Set the position of the video to the start of the action
            
            prefix = output_path + actor + '/' + action['actionlabel'] + '/{}_{}_{}'.format(action['from_activity'], action['start'], action['end'])
            #print(prefix)
            if not os.path.exists(prefix): os.makedirs(prefix)
            else: continue
            
            capture.set(cv2.CAP_PROP_POS_FRAMES, action['start'])

            #print(action['start'], capture.get(cv2.CAP_PROP_POS_FRAMES), action['end'])
            i = 0
            while int(capture.get(cv2.CAP_PROP_POS_FRAMES)) < action['end']:
                ret, frame = capture.read()
                if ret == True:
                    save_path = prefix + '/frame%.5d.jpeg' % i
                     #cv2.imwrite(save_path, cv2.resize(frame, (224,224)), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    cv2.imwrite(save_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    # From string to integer with the action dictionary
                    #labelfile.write(str(dic[action['actionlabel']])+'\n')
                    #objectfile.write(str(object_dictionary[action['objects']]) + '\n')  
                    i += 1
            #indices[action['actionlabel']] += 1
        capture.release()            
            
    print('------- ' + folder_of_videos + ' processed. -----------')