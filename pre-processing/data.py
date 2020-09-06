import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from random import shuffle
# import Datasets.landmark as Landmark
import cv2 as cv2
import numpy as np


class DataSetTrain(Dataset):

    def __init__(self, data_root, type_ds):
        # super(self).__init__()
        self.data_root = data_root
        self.type_ds  = type_ds
        self.classes = []
        self.imgs = []
        self.totensor=transforms.Compose([
                            transforms.Resize(size=(128, 128)),
                            # transforms.CenterCrop(128),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])
                        ])
        self.toTensorEye=transforms.Compose([
                            transforms.Resize(size=(40, 40)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])
                        ])
        self.toTensorNose=transforms.Compose([
                            transforms.Resize(size=(40, 32)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])
                        ])
        self.toTensorMouth=transforms.Compose([
                            transforms.Resize(size=(32, 48)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])
                        ])
        self.tobasictensor=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.template  = cv2.imread("template.")
        # Get images
        if type_ds == 'LFW':
            self.imgs = self.get_images_lfw(data_root)
        if type_ds == 'CPF':
            self.imgs = self.get_images_cpf(data_root)



    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        item = {}
        data = self.imgs[index]
        data_profile = Image.open(data[1]).convert(mode='RGB')
        path_profile = data[1]
        open_cv_prof = np.array(data_profile) 
        data_frontal = Image.open(data[2]).convert(mode='RGB')
        path_frontal = data[2]
        open_cv_fron = np.array(data_frontal) 

        #Resize opencv
        # dsize = (224, 224)
        # output_prof = cv2.resize(open_cv_prof, dsize)
        # output_fron = cv2.resize(open_cv_fron, dsize)



        if self.type_ds == 'CPF':
            print(data[0])
            item["name"]      = self.dic_classes[(int(data[0]))] # las clasese en listad van desde 0 499 pero paa los archivos son de 1 a 500
            item["label"]     = int((int(data[0])))
        if self.type_ds == 'LFW':
            item["name"]      = data[0]
            item["label"]     = self.dic_classes[data[0]]

        item["profile"]   = self.totensor(data_profile)
        item["frontal"]   = self.totensor(data_frontal)
        item["path_frontal"]   = path_frontal
        item["path_profile"]   = path_profile
        #item["frontal"] = self.totensor(Image.fromarray(data_frontal_align))
        # item["frontal32"] = self.tobasictensor(data_frontal.resize((32,32), Image.ANTIALIAS))
        # item["frontal64"] = self.tobasictensor(data_frontal.resize((64,64), Image.ANTIALIAS))

        return item

    def get_images_lfw(self, data_root):
        # Get classes
        list_dir = os.listdir(path=self.data_root)
        self.classes = sorted(list_dir)

        imgs = []
        for register in os.listdir(data_root):
            # print(register)
            register_folder = os.path.join(data_root, register)

            # Get frontal image
            frontal_file = register_folder
            name_frontals = []
            for pfile in os.listdir(frontal_file):
                path_frontal = os.path.join(frontal_file, pfile)
                name_frontals.append(path_frontal)

            ## Get profile image an to asociate it with frontal images
            profile_file = register_folder
            shuffle(name_frontals) #select a random but the same for all images

            for pfile in os.listdir(profile_file):
                file_profile = os.path.join(profile_file, pfile)

                file_frontal = name_frontals[0]
                print(register)
                imgs.append((register, file_profile, file_frontal))
        self.dic_classes = {x:i for i,x in enumerate(self.classes)}
        
        return imgs


    def get_images_cpf(self, data_root):
        # Get classes
        with open(self.data_root + "list_name.txt", 'r') as fp:
            # line = fp.readline()
            line = fp.readline() # not first line
            while line:
                self.classes.append(line.strip())
                line = fp.readline()

        # Get images
        imgs = []
        for register in os.listdir(self.data_root + "Images/"):
            # print(register)
            register_folder = os.path.join(self.data_root + "Images/", register)

            # Get frontal image
            frontal_file = os.path.join(register_folder, 'frontal')
            profile_file = os.path.join(register_folder, 'profile')
            
            for pfile in os.listdir(profile_file):
                img_profile = os.path.join(profile_file, pfile)
                # Associate four frontal image to each profile image
                for ffile in os.listdir(frontal_file):
                    img_frontal = os.path.join(frontal_file, ffile)
                    imgs.append((register, img_profile, img_frontal))
                   
                # file_frontal = frontal_file + "/01.jpg"
                # self.imgs.append((register, file_profile, file_frontal))

        self.dic_classes = {i+1:x for i,x in enumerate(self.classes)} # cuando comienza en 001 a 500
        
        return imgs