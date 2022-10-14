import os, shutil

original_dataset_dir = r'/mnt/data_disk/Dataset/dogs-vs-cats/train_ori'
base_dir = r'/mnt/data_disk/Dataset/dogs-vs-cats/small_dataset'

if not os.path.isdir(base_dir):os.mkdir(base_dir)


train_dir = os.path.join(base_dir,'train')
if not os.path.isdir(train_dir):os.mkdir(train_dir)
val_dir = os.path.join(base_dir,'validation')
if not os.path.isdir(val_dir):os.mkdir(val_dir)
test_dir = os.path.join(base_dir,'test')
if not os.path.isdir(test_dir):os.mkdir(test_dir)


train_cats_dir = os.path.join(train_dir,'cats')
if not os.path.isdir(train_cats_dir):os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir,'dogs')
if not os.path.isdir(train_dogs_dir):os.mkdir(train_dogs_dir)

val_cats_dir = os.path.join(val_dir,'cats')
if not os.path.isdir(val_cats_dir):os.mkdir(val_cats_dir)

val_dogs_dir = os.path.join(val_dir,'dogs')
if not os.path.isdir(val_dogs_dir):os.mkdir(val_dogs_dir)

test_cats_dir = os.path.join(test_dir,'cats')
if not os.path.isdir(test_cats_dir):os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir,'dogs')
if not os.path.isdir(test_dogs_dir):os.mkdir(test_dogs_dir)

# %%  
def copy_file_to_dir(file_format_name,start_file_num,end_file_num,original_dataset_dir,target_dogs_dir):
    fnames = [file_format_name.format(i) for i in range(start_file_num,end_file_num)]

    for fname in fnames:
        src = os.path.join(original_dataset_dir,fname)
        dst = os.path.join(target_dogs_dir,fname)
        shutil.copyfile(src,dst)

# %%  
copy_file_to_dir('cat.{}.jpg',1,1000,original_dataset_dir,train_cats_dir)
copy_file_to_dir('cat.{}.jpg',1000,1500,original_dataset_dir,val_cats_dir)
copy_file_to_dir('cat.{}.jpg',1500,2000,original_dataset_dir,test_cats_dir)
# %%  
copy_file_to_dir('dog.{}.jpg',1,1000,original_dataset_dir,train_dogs_dir)
copy_file_to_dir('dog.{}.jpg',1000,1500,original_dataset_dir,val_dogs_dir)
copy_file_to_dir('dog.{}.jpg',1500,2000,original_dataset_dir,test_dogs_dir)

