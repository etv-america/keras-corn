import csv, os, random, time, cv2
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt


def read_image_data_pairs(csv_name):
    csv_file = open('./Input/' + csv_name + '.csv')  #  setting up csv reading
    csv_read, data = csv.reader(csv_file), []
    for row in csv_read:  #  read through the csv
        #concise_row = [row[3], row[1]] # isolate blight bools and image IDs in our csv
        data.append(row)  #  add to return list
    return data

def get_all_bools(csv_name, start=None, end=None):  #  return list of blight bools in order
    to_bool = {"True":1, "False":0}
    bools = [to_bool[i[1]] for i in read_image_data_pairs(csv_name)]
    return bools[start:end] #  add slice

def read_all_images(image_folder_name, csv_name, x=None, y=None, start=None, end=None, mute=0):  #  returns all images, resized as specified
    image_ID_list, start_time = ['./Input/' + image_folder_name + '/'+ str(i[0]) for i in read_image_data_pairs(csv_name)], time.time()
    image_ID_list, image_list = image_ID_list[start:end], []
    for image in image_ID_list:
        if mute == 0:
            print("Reading images: %.1f%% complete" % (100*image_ID_list.index(image)/len(image_ID_list)), end="\r")
        image = cv2.imread(image)
        if x != None and y != None:
            image = cv2.resize(image, (y, x))  #  optional resize
        image_list.append(image)
    if mute == 0:
        end_time = time.time()-start_time
        print("Reading images: 100% complete   ")   #  possible future error debug, size given is that of first image, not necessarily of all images
        print("Read {} images, sized {}, in {} min {} sec".format(len(image_list), image_list[0].shape, end_time//60, round(end_time%60, 3)))   
    return image_list


def images_to_pickle(filepath, new_file_name, image_folder_name, csv_name, x=None, y=None, start=None, end=None, mute=0):  #  create pickle of the desired data for quick access
    pickle_loc = filepath + '/' + new_file_name + '.pickle'
    if os.path.isfile(pickle_loc) == True:   #  check if file exists, make new file if not
        if mute == 0:
            print('Cannot write; {} already exists'.format(pickle_loc))
        return None
    else:
        os.mknod(pickle_loc)
    
    pictures = read_all_images(image_folder_name, csv_name, x=x, y=y, start=start, end=end, mute=mute)
    pickle_time = time.time()  #  start time is here to only calculate time taken to pickle, not to read (that's already done by the reader)
    
    with open(pickle_loc, 'wb') as pickle:  #  actual writing stage
        cPickle.dump(pictures, pickle)
    end_time = time.time()-pickle_time
    if mute == 0:
        print("\nCreated new file {}.pickle in {} min {} sec".format(new_file_name, end_time//60, round(end_time%60, 3)))
    
def make_pickle(set_name, start=None, end=None, mute=0):  #  compact pickle maker for our use, only needs the name of the dataset
    images_to_pickle('./Pickles', set_name + '_imgs', 'images_' + set_name, 'labels_' + set_name, x=256, y=256, start=start, end=end, mute=mute)
    
    
def parity_balance(set1, set2):  #  takes a set and associated boolean values in a second set and truncates the data so the number
    if len(set1) != len(set2):   #  of data points under each classification is equal, general purpose
        print('ERROR: Mismatched input sizes')
        return None, None
    else:    
        true_set, false_set, orig_size = list(zip(set2, set1)), [], len(set2)  #  lists are zipped to preserve bool match during transit
        for i in range(len(true_set)-1,-1,-1):  #  filtering out the ones to keep (true set is initially the full set and then reduced, saves memory)
            if true_set[i][0] == False:
                false_set.append(true_set[i])
                true_set.pop(i)    
        if len(true_set) > len(false_set):  #  determine which set is to be trimmed
            trim_set, static_set, cuts, trim_orig = true_set, false_set, True, len(true_set)
        elif len(true_set) < len(false_set):
            trim_set, static_set, cuts, trim_orig = false_set, true_set, False, len(false_set)
        else:
            print('Set input is balanced as is')
            return set1, set2
        random.shuffle(trim_set)  #  the kept trimmed set will be different each time, for a more robust trimmed sample over small/limited data
        trim_set = trim_set[:len(static_set)] 
        new_set = static_set + trim_set
        random.shuffle(new_set)  #  debatable as to whether this affects how well the model can predict over the trimmed set, output "looks" better 
        
        set1, set2 = [], []
        for i, val in enumerate(new_set):  #  unpacking the new lists
            set1.append(val[1])
            set2.append(val[0])
        print("Truncated set from %d to %d items by %d %s's: %d of each now present"%  #  lengthy but informative print out
              (orig_size, len(set2), trim_orig-len(trim_set), cuts, len(static_set)))
        return set1, set2

def get_features_and_labels(set_name, balance=False, start=None, end=None, mute=0): #  get the images and bools for a dataset, optionally trim it so the set is evenly split
    if os.path.isfile('./Pickles/' + set_name + '_imgs.pickle') == False:   #  upon first time read, make a pickle file of the images being fetched
        print('First time read, creating .pickle file...')
        make_pickle(set_name, start=start, end=end, mute=mute)
   
    with open('./Pickles/' + set_name + '_imgs.pickle', 'rb') as pick:  #  get pictures from relevant pickle file
        pictures = cPickle.load(pick)[start:end]
    blight = get_all_bools('labels_' + set_name, start=start, end=end)  #  get the relevant blight bools
    
    if balance == True:  #  optional balancing
        pictures, blight = parity_balance(pictures, blight)  
    return pictures, blight


def adaptive_graph(pics, titles, ncols):  #  neatly fits and displays an arbitrarily long list of images into the number of columns specified, general purpose
    if titles == None:
        titles = [''] * len(pics)
    if len(pics) != len(titles):  
        print("ERROR: Mismatched input sizes")
        return None  #  check mismatch, avoids uglier errors
    else:
        span = len(titles)
        nrows = ((span-1)//ncols)+1  #  scaling for arbitrary width
        fig, axs = plt.subplots(nrows,ncols,figsize=(19, round(19/ncols*nrows)))  #  19 seems to be a 'magic number' which displays very well the in jupyter window,
        for i in range(span):                         #   on the current monitor; for future, need to implement more intelligent calculation for cross-compatibility
            curr_plot = axs[(i)//ncols][i%ncols]  #  index is essentially iterated scaling
            curr_plot.set_title(titles[i])
            curr_plot.axis('off')
            curr_plot.imshow(pics[i])
        plt.tight_layout()
            
def preview_crops(set_name, ncols, balance=False, start=None, end=None, mute=0):  #  more specified grapher for directly previewing our data
    imgs, titles = get_features_and_labels(set_name=set_name, balance=balance, start=start, end=end, mute=mute) 
    labels = {1:"Sick", 0:"Healthy"}                                               
    names = [labels[i] for i in titles]
    adaptive_graph(imgs, names, ncols)  #  same 'inline' requirement as mentioned prior
