import csv, os
import cv2, h5py
import numpy as np

#  add path for csv of other data sets when using them    
csv_location = '/home/etv/Documents/notebooks/KerasTest/PlantData/testlst.csv'
image_folder = '/home/etv/Documents/notebooks/KerasTest/PlantData/images_handheld/'

def read_image_data_pairs():
    data = []
    csv_file = open(csv_location)  #  setting up csv reading
    csv_read = csv.reader(csv_file)
    for row in csv_read:  #  read through the csv
        row_data = row[0].split()  #  break up info in row
        concise_row = row_data[2].split("/")  # isolate image ID and blight bool
        data.append(concise_row)  #  add to list for ease of manipulation
    return data

def get_all_Bools(start=None, end=None):  #  return list of blight bools in order
    to_bool = {"True":1, "False":0}
    bools = [to_bool[i[0]] for i in read_image_data_pairs()]
    return bools[start:end] #  add slice

def read_all_images(x=None, y=None, start=None, end=None, mute=0):  #  returns all images, resized as specified
    image_ID_list = [image_folder + str(i[1]) for i in read_image_data_pairs()]
    image_ID_list = image_ID_list[start:end]  #  !!remove when using full dataset, this just saves time when iterating through and debugging!!
    image_list = []
    for image in image_ID_list:
        if mute == 0:
            print("Processing images: %.1f%% complete" % (100*image_ID_list.index(image)/len(image_ID_list)), end="\r")
        image = cv2.imread(image)
        if x != None and y != None:
            image = cv2.resize(image, (y, x))
        image_list.append(image)
    if mute == 0:
        print("\rProcessing images: 100% complete ")                                         #  possible future error debug, size given is that of first image,
        print("Processed {} images, sized {}".format(len(image_list), image_list[0].shape))  #  <----- not necessarily of all images
    return image_list

def get_features_and_labels(x=6000, y=4000, start=None, end=None):  #  Combined output of bools and images
    return np.array(read_all_images(x=x, y=y, start=start, end=end)), np.array(get_all_Bools(start=start, end=end))

def images_to_hdf5(file_name, data_name="pics", start=None, end=None, mute=0):  # write image data into an hdf5 file to save read time (<1/6 load time from observations)
    '''Use only when creating a new read file for crop data,
    read from existing file otherwise to save time and memory overhead'''
    filepath = './' + str(file_name) + '.hdf5'
    pictures = read_all_images(x=256, y=256, start=start, end=end, mute=mute)
    if os.path.isfile(filepath) == True:
        os.remove(filepath)
    os.mknod(filepath)
    pictures = np.array(pictures).tolist()  #  make into list
    with h5py.File(filepath, 'w') as f:
        f.create_dataset(data_name, data=pictures, dtype="uint8")