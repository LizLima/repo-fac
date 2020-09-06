import os
import torch
import torchvision.utils as vutils
from data import DataSetTrain

##############
# Load dataset
##############
def croppedCPF():
  path_cpf    = "/media/liz/TOSHIBA EXT/Data/cfp-dataset/Data/"
  data        = DataSetTrain(path_cpf, 'CPF') # LFW
  NUM_CLASSES = len(data.classes)
  batch_size = 1
  print(len(data))
  trainloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

  # Path to create a dataset cropped
  path_imagedata = '/media/liz/TOSHIBA EXT/Data/cfp-dataset/Data224/Images/' 
  for x in trainloader:
    data    = x
    # Inputs
    in_img   = data['profile']
  
    # Target
    tar_img   = data['frontal']

    # Label
    tar_label = data['label']

    #Create directory
    path_img = path_imagedata + "%03d" % tar_label.item()
    isFile = os.path.isdir(path_img)
    if not isFile:
      #cretae direcotry
      os.mkdir(path_img)

    # Dir frontal
    path_frontal = path_img + '/frontal'
    isFile = os.path.isdir(path_frontal)
    if not isFile:
      #Create direcotry
      os.mkdir(path_frontal)

    # Dir profile
    path_profile = path_img + '/profile'
    isFile = os.path.isdir(path_profile)
    if not isFile:
      #cretae direcotry
      os.mkdir(path_profile)

      

    # save image frontal croped
    id_img_front =  data['path_frontal'][0].split('/')
    id_img_front = id_img_front[-1]
    print(id_img_front)
    id_img_prof =  data['path_profile'][0].split('/')
    id_img_prof = id_img_prof[-1]
    print('label: ', data['name'])
    # print(data['path_frontal'])
    # print(id_img_prof, tar_label)
    # print('_________________________')

    # Save image 
    # id_img_front : contiene extension
    vutils.save_image(tar_img.data, path_frontal + '/' + id_img_front, normalize=True)
    vutils.save_image(in_img.data, path_profile + '/' + id_img_prof, normalize=True)


##############
# Load dataset
##############
def croppedLFW():
  path_cpf    = "/media/liz/TOSHIBA EXT/Data/LFW/lfw"
  data        = DataSetTrain(path_cpf, 'LFW') # LFW
  NUM_CLASSES = len(data.classes)
  batch_size = 1
  print(len(data))
  trainloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

  # Path to create a dataset cropped
  path_imagedata = '/media/liz/TOSHIBA EXT/Data/LFW/lfw128' 
  for x in trainloader:
    data    = x
    # Inputs
    in_img   = data['profile']
    path_img = data['path_profile']
    name     = data['name']

    #Create directory
    path_folder = path_imagedata + "/" + name[0]
    isFile = os.path.isdir(path_folder)
    if not isFile:
      #cretae direcotry
      os.mkdir(path_folder)

      

    # save image cropped
    temp = data['path_profile'][0].split(name[0])
    id_img_front = temp[-1]
    path_save = path_folder + '/' + name[0] + id_img_front
    # Save image 
    # id_img_front : contiene extension
    vutils.save_image(in_img.data, path_save , normalize=True)

croppedLFW()