from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans

# 새로운 라벨과 색상 매핑
new_labels = {
    "sidewalk": [0, 0, 255],
    "crosswalk": [0, 255, 0],
    "bikeroads": [128, 0, 128],
    "crossroads": [255, 0, 255],
    "centerline": [0, 128, 128],
    "num1": [128, 128, 0],
    "stopline": [255, 0, 0],
    "Between": [255, 255, 0],
    "green_crosswalk": [128, 0, 0],
    "red_crosswalk": [0, 128, 0],
    "green_driveway": [75, 0, 130],
    "red_driveway": [128, 128, 128],
}

# 새로운 COLOR_DICT 생성
COLOR_DICT = np.array(list(new_labels.values()))

def adjustData(img, mask, flag_multi_class, num_class):
    if flag_multi_class:
        img = img / 255.0
        mask = mask / 255.0

        # Create one-hot encoded masks
        new_mask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], num_class))
        for i in range(num_class):
            new_mask[mask[:, :, :, 0] == i, i] = 1

        return img, new_mask
    else:
        # For binary classification
        img = img / 255.0
        mask = mask / 255.0
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        return img, mask

def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = True,num_class = 12,save_to_dir = None,target_size = (1920,1080),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = 'rgb',
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = 'rgb',
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)


def testGenerator(test_path, num_image=8, target_size=(1920, 1080), flag_multi_class=True, as_gray=False):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, "%d.jpg" % i), as_gray=as_gray)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (3,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img


def geneTrainNpy(image_path, mask_path, flag_multi_class=True, num_class=12, image_prefix="image", mask_prefix="mask", image_as_gray=False, mask_as_gray=False):
    image_name_arr = glob.glob(os.path.join(image_path, "%s*.jpg" % image_prefix))
    image_arr = []
    mask_arr = []
    for index, item in enumerate(image_name_arr):
        img = io.imread(item, as_gray=image_as_gray)
        img = np.reshape(img, img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path, mask_path).replace(image_prefix, mask_prefix), as_gray=mask_as_gray)
        mask = np.reshape(mask, mask.shape + (1,)) if mask_as_gray else mask
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr, mask_arr

def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255


def saveResult(save_path,npyfile,flag_multi_class = True,num_class = 12):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.jpg"%i),img)