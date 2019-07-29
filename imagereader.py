import csv, os
import cv2, h5py
import numpy as np

def read_image_data_pairs(csv_name):
    data = []
    csv_file = open('./input/' + csv_name + '.csv')  #  setting up csv reading
    csv_read = csv.reader(csv_file)
    for row in csv_read:  #  read through the csv
        concise_row = [row[3], row[1]] # isolate and arrange blight bool and image ID
        data.append(concise_row)  
    return data

def get_all_Bools(csv_name, start=None, end=None):  #  return list of blight bools in order
    to_bool = {"True":1, "False":0}
    bools = [to_bool[i[0]] for i in read_image_data_pairs(csv_name)]
    return bools[start:end] #  add slice

def read_all_images(image_folder_name, csv_name, x=None, y=None, start=None, end=None, mute=0):  #  returns all images, resized as specified
    image_ID_list = ['./input/' + image_folder_name + '/'+ str(i[1]) for i in read_image_data_pairs(csv_name)]
    image_ID_list = image_ID_list[start:end]  #  add slice
    image_list = []
    for image in image_ID_list:
        if mute == 0:
            print("Processing images: %.1f%% complete" % (100*image_ID_list.index(image)/len(image_ID_list)), end="\r")
        image = cv2.imread(image)
        if x != None and y != None:  #  optional resize
            image = cv2.resize(image, (y, x))
        image_list.append(image)
    if mute == 0:
        print("\rProcessing images: 100% complete     ")                                     #  possible future error debug, size given is that of first image,
        print("Processed {} images, sized {}".format(len(image_list), image_list[0].shape))  #  <----- not necessarily of all images
    return image_list

def images_to_hdf5(new_file_name, image_folder_name, csv_name, data_name="pics", start=None, end=None, mute=0):  
    '''Write image data into an hdf5 file to save read time (<1/6 load time from observations)
    Use only when creating a new read file for crop data, read from existing file otherwise to save time and memory overhead'''
    filepath = './' + str(new_file_name) + '.hdf5'
    pictures = read_all_images(image_folder_name, csv_name, x=256, y=256, start=start, end=end, mute=mute)
    if os.path.isfile(filepath) == True:
        os.remove(filepath)
    os.mknod(filepath)
    pictures = np.array(pictures).tolist()  #  make into list
    with h5py.File(filepath, 'w', libver='latest') as f:
        f.create_dataset(data_name, data=pictures, dtype="uint8")
    print("\nCreated new file {}.hdf5".format(new_file_name))
        
