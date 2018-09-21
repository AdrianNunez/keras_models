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