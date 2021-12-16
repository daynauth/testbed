import os
import glob
import random
import shutil

path = 'kitti'
split = 0.8

if not os.path.isdir(path):
    os.mkdir(path)

#grab all the files in the directory
files = glob.glob('training/image_2/*')

#randomize the list just in case
random.shuffle(files)

#use the 80/20 split for training and validation
train_num = int(split * len(files))
train_files = files[:train_num]
val_files = files[train_num:]

train_dir = os.path.join(path, 'train')
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)

for f in train_files:
    shutil.copy(f, train_dir)

val_dir = os.path.join(path, 'val')
if not os.path.isdir(val_dir):
    os.mkdir(val_dir)

for f in val_files:
    shutil.copy(f, val_dir)


