import pandas as pd
import nibabel as nib
import numpy as np 
import matplotlib.pyplot as plt
import glob
import os
import pydicom as dicom
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from torchvision import transforms
#DEfault transformations in images unless specified
D_TRANSFORMS = transforms.Compose([transforms.ToTensor(), transforms.Resize((960, 320), antialias=True)])

#Get the paths of all DCM and segmentation images
class MiceDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=D_TRANSFORMS):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotation_csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.seg_path, self.images_path = self.get_paths()

        self.images_index = np.cumsum([len(i) for i in self.images_path]) # List containing the index of each image for each annotaiton
        self.images_index = np.insert(self.images_index, 0, 0, axis = 0)
    def get_paths(self):

        data = self.annotation_csv

        paths_folder = data["3D SLICER PATH"].dropna() # Paths of the folders where images and annotations are saves
        seg_path = [glob.glob(os.path.join(self.root_dir, i) + "/*.nii") for i in paths_folder] #Path of the segmentation
        images_path =  [glob.glob(os.path.join(self.root_dir, i) + "/*.dcm*") for i in paths_folder] #Path of the DCM

        #If there's no segmentation, take path out
        seg_path_2 = []
        images_path_2 = []
        for idx, i in enumerate(seg_path):
            if i:
                seg_path_2.append(i)
                images_path_2.append(images_path[idx])
        return seg_path_2, images_path_2

    def __len__(self):
        return self.images_index[-1]

    def __getitem__(self, idx):
        #Since not all MRI have 32 images this calculaitons must be done
        for idx_2, i in enumerate(self.images_index):
            if i > idx:
                image_index = idx_2 -1 # Get the value of the previous number
                break
        level_index = idx - self.images_index[image_index] # The "level" of the annotation
        assert level_index >= 0  and level_index <= len(self.images_path[image_index])
        #from the image image_index, get the level_index

        #Open the Dicom .dcmread(image_path)
        img = dicom.dcmread(self.images_path[image_index][level_index]).pixel_array
        #Open the .nii image
        try:
                seg = nib.load(self.seg_path[image_index][0]).get_fdata()
                seg = seg[:,:,level_index].transpose(1, 0)
        except:
                print("img")
                print(self.seg_path[2])
                print(nib.load(self.seg_path[image_index][0].get_fdata()).shape)
                print(len(self.seg_path))
        img = self.transform(img.astype(np.float32))
        seg = D_TRANSFORMS(seg.astype(np.float32))
        return img, seg



if __name__ == "__main__":
    dataset=  MiceDataset("/data/manifest-1686081801328/new_metadata.csv","/data/manifest-1686081801328")
    for i in range(len(dataset)):
       img, seg = dataset.__getitem__(i)
       if list(img.shape) != [1, 960, 320] or list(seg.shape) != [1, 960, 320]:
           print(i)
           break
   # cv2.imwrite("img.png", img[0].numpy())
    #cv2.imwrite("seg.png", seg[0].numpy())
