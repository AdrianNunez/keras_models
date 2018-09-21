from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.random import seed
seed(7)
import numpy as np
from matplotlib.patches import Rectangle
import itertools
import psutil

def show_RAM():
    values = psutil.virtual_memory()
    used = values.used / (1024*1024)
    active = values.active / (1024*1024)
    print('RAM: {}MB, {}MB'.format(used, active))

def get_classes(ind_file):
    '''
    Returns an array with all the class names.
    Output:
    * classes: array of size num_classes with strings.
    '''
    classes = dict()
    with open(ind_file, 'r') as f:
         content = f.readlines()
         for c in content:
            num, class_name = c.strip().split(' ')
            classes[class_name] = int(num)-1
    return classes
    
def calculate_evaluation_metrics(y_gt, y_preds):
    """Calculates the evaluation metrics (precision, recall and F1) for the
    predicted examples. It calculates the micro, macro and weighted values
    of each metric.
            
    Usage example:
        y_gt = ['make_coffe', 'brush_teeth', 'wash_hands']
        y_preds = ['make_coffe', 'wash_hands', 'wash_hands']
        metrics = calculate_evaluation_metrics (y_ground_truth, y_predicted)
        
    Parameters
    ----------
        y_gt : array, shape = [n_samples]
            Classes that appear in the ground truth.
        
        y_preds: array, shape = [n_samples]
            Predicted classes. Take into account that the must follow the same
            order as in y_ground_truth
           
    Returns
    -------
        metric_results : dict
            Dictionary with the values for the metrics (precision, recall and 
            f1)    
    """
        
    metric_types =  ['micro', 'macro', 'weighted']
    metric_results = {
        'precision' : {},
        'recall' : {},
        'f1' : {},
        'acc' : -1.0        
    }
            
    for t in metric_types:
        metric_results['precision'][t] = metrics.precision_score(y_gt, y_preds, average = t)
        metric_results['recall'][t] = metrics.recall_score(y_gt, y_preds, average = t)
        metric_results['f1'][t] = metrics.f1_score(y_gt, y_preds, average = t)
        metric_results['acc'] = metrics.accuracy_score(y_gt, y_preds) 
                
    return metric_results

def plot_training_info(test_subject, parameters, metrics, save, losses, accuracies):
    plot_folder = parameters['plots_folder']
    
    # summarize history for accuracy
    if 'accuracy' in metrics:
        plt.plot(accuracies['train'])
        plt.plot(accuracies['val'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        lgd = plt.legend(['train', 'val'], bbox_to_anchor=(1.04,1), loc="upper left")
        if save == True:
            plt.savefig(plot_folder + '{}_accuracy.png'.format(test_subject), bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.gcf().clear()
        else:
            plt.show()

    # summarize history for loss
    if 'loss' in metrics:
        plt.plot(losses['train'])
        plt.plot(losses['val'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        #plt.ylim(1e-3, 1e-2)
        plt.yscale("log")
        plt.legend(['train', 'val'], bbox_to_anchor=(1.04,1), loc="upper left")
        if save == True:
            plt.savefig(plot_folder + '{}_loss.png'.format(test_subject), bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.gcf().clear()
        else:
            plt.show()
            
def plot_confusion_matrix(cm, classes, path, normalize=False, title='Confusion matrix', cmap='coolwarm', font_size=2):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = np.asarray(cm, dtype=np.float32)
        for i in range(cm.shape[0]):
            if cm[i,:].sum() > 0:
                row_total = np.float(np.sum(cm[i,:]))
                for j in range(cm.shape[1]):
                    cm[i,j] = float(cm[i,j]) / float(row_total)
            else:
                cm[i,...] = np.zeros((cm.shape[1]))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks_x = np.arange(len(classes))
    tick_marks_y = np.arange(len(classes))
    plt.xticks(tick_marks_x, classes, fontsize=4, rotation=90)
    plt.yticks(tick_marks_y, classes, fontsize=4)

    width, height = cm.shape
    fmt = '.2f' if normalize else 'd'
    for i, j in itertools.product(range(width), range(height)):
        plt.text(j, i, format(cm[j,i], fmt),
                 horizontalalignment="center", fontsize=font_size,
                 color="black")
      
    ax = plt.gca()
    for i in range(range(width)):
        rect = Rectangle((-0.5+i, -0.5+i), 1, 1, fill=False, edgecolor='black', lw=0.5)
        ax.add_patch(rect)
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path, bbox_inches='tight')
    plt.gcf().clear()