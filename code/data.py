from torch.utils.data import Dataset
import cv2
import numpy as np
from torchvision import transforms
import torch
import pandas as pd

#7pt
pigment_network_label_list = [['absent'],['typical'],['atypical']]#'typical:1,atypical:2
streaks_label_list = [['absent'],['regular'],['irregular']]#regular:1, irregular:2
pigmentation_label_list = [['absent'],
                      ['diffuse regular','localized regular'],
                      ['localized irregular','diffuse irregular']]#regular:1, irregular:2
regression_structures_label_list = [['absent'],
                               ['blue areas','combinations','white areas']]# present:1
dots_and_globules_label_list = [['absent'],['regular'],['irregular']]#regular:1, irregular:2
blue_whitish_veil_label_list = [['absent'],['present']]#present:1
vascular_structures_label_list = [['absent'],
                             ['within regression','arborizing','comma','hairpin','wreath'],
                             ['linear irregular','dotted']]

class Derm7ptDatset(Dataset):
    def __init__(self,img_dir,img_info,file_list,img_size,is_test = False):
        #data_dir=args.data_dir,img_info=metadata_df,file_list=train_indexes,img_size=args.image_size,is_test=False
        self.is_test = is_test
        self.img_dir = img_dir
        self.file_list = file_list
        self.img_info = img_info
        self.img_size = img_size
        self.image_train_transforms = transforms.Compose([
            transforms.ToTensor(),
            
            # transforms.CenterCrop(512),
            # transforms.Resize(fundus_img_size[0][0])
            transforms.Resize([256,256]),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
        ])
        self.image_test_transforms = transforms.Compose([
            transforms.ToTensor(),
            
            # transforms.CenterCrop(512),
            # transforms.Resize(fundus_img_size[0][0])
            transforms.Resize([256,256]),
            transforms.CenterCrop(224),
        ])


    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        file_id = self.file_list[index]
        sub_img_info = self.img_info[file_id:file_id+1]

        derm_image_path = sub_img_info['derm'][file_id]
        derm_img = cv2.imread(self.img_dir+derm_image_path)
        derm_img = derm_img / 255.0
        if self.is_test == False:
            derm_img = self.image_train_transforms(derm_img.astype(np.float32))
        else:
            derm_img = self.image_test_transforms(derm_img.astype(np.float32))

        #数据增强？

        #nevus和melanoma二分类任务的标签
        diagnosis_label = sub_img_info['diagnosis'][file_id]
        # print(file_id)
        # print(diagnosis_label)

        # if 'melanoma' in diagnosis_label:
        #     label =torch.tensor([0,1])
        #     diagnosis_label = '[melanoma]'
        #     # learnable_prompt = 'A photo of a [melanoma]'
        # elif 'nevus' in diagnosis_label:
        #     label =torch.tensor([1,0])
        #     diagnosis_label = '[nevus]'
        #     # learnable_prompt = 'A photo of a [no melanoma]'
        # else:
        #     print(diagnosis_label)
        #     print(file_id)

        if 'melanoma' in diagnosis_label:
            label =torch.tensor([0,1])
            diagnosis_label = '[melanoma]'
            # learnable_prompt = 'A photo of a [melanoma]'
        else:
            label =torch.tensor([1,0])
            diagnosis_label = '[nevus]'
            # learnable_prompt = 'A photo of a [no melanoma]'
        # else:
        #     print(diagnosis_label)
        #     print(file_id)

        
        #clinical prompt
        pigment_network_label = sub_img_info['pigment_network'][file_id]
        pigment_network_prompt = pigment_network_label + ' pigment network'
        streaks_label = sub_img_info['streaks'][file_id]
        streaks_prompt = streaks_label +' streaks'
        pigmentation_label = sub_img_info['pigmentation'][file_id]
        pigmentation_prompt = pigmentation_label + ' pigmentation'
        regression_structures_label = sub_img_info['regression_structures'][file_id]
        regression_structures_prompt = regression_structures_label + ' regression structures'
        dots_and_globules_label = sub_img_info['dots_and_globules'][file_id]
        dots_and_globules_prompt = dots_and_globules_label +' dots and globules'
        blue_whitish_veil_label = sub_img_info['blue_whitish_veil'][file_id]
        blue_whitish_veil_prompt = blue_whitish_veil_label + ' blue whitish veil'
        vascular_structures_label = sub_img_info['vascular_structures'][file_id]
        vascular_structures_prompt = vascular_structures_label + ' vascular structures'
        clinical_prompt = 'A photo of a '+ diagnosis_label+ ', with ' +pigment_network_prompt+', '+streaks_prompt+', '+pigmentation_prompt+', '+regression_structures_prompt+ ', '+dots_and_globules_prompt+', '+blue_whitish_veil_prompt+', '+vascular_structures_prompt+'.'


        return derm_img,label,clinical_prompt


        
class PH2Datset(Dataset):
    def __init__(self,img_dir,list_dir,is_test=True):
        #data_dir=args.data_dir,img_info=metadata_df,file_list=train_indexes,img_size=args.image_size,is_test=False
        self.is_test = is_test
        self.img_dir = img_dir
        self.list_dir = list_dir
        self.test_indexes= list(pd.read_csv(list_dir)['indexes'])
        self.label_indexes = list(pd.read_csv(list_dir)['diagnosis'])
    
        self.image_train_transforms = transforms.Compose([
            transforms.ToTensor(),
            
            # transforms.CenterCrop(512),
            # transforms.Resize(fundus_img_size[0][0])
            transforms.Resize([256,256]),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
        ])
        self.image_test_transforms = transforms.Compose([
            transforms.ToTensor(),
            
            # transforms.CenterCrop(512),
            # transforms.Resize(fundus_img_size[0][0])
            transforms.Resize([256,256]),
            transforms.CenterCrop(224),
        ])

    
    def __len__(self):
        return len(self.test_indexes)
    
    def __getitem__(self, index):
        file_id = self.test_indexes[index]
        diagnosis_label = self.label_indexes[index]
        # sub_img_info = self.img_info[file_id:file_id+1]
        #/home/wenyu/Downloads/PH2Dataset/PH2Dataset/PH2Datasetimages/
        image_path = self.img_dir+file_id+'/'+file_id + '_Dermoscopic_Image/'+file_id+'.bmp'

        # derm_image_path = sub_img_info['derm'][file_id]
        derm_img = cv2.imread(image_path)
        derm_img = derm_img / 255.0
        if self.is_test == False:
            derm_img = self.image_train_transforms(derm_img.astype(np.float32))
        else:
            derm_img = self.image_test_transforms(derm_img.astype(np.float32))

        #数据增强？

        #nevus和melanoma二分类任务的标签
        
        # print(file_id)
        # print(diagnosis_label)

        # if 'melanoma' in diagnosis_label:
        #     label =torch.tensor([0,1])
        #     diagnosis_label = '[melanoma]'
        #     # learnable_prompt = 'A photo of a [melanoma]'
        # elif 'nevus' in diagnosis_label:
        #     label =torch.tensor([1,0])
        #     diagnosis_label = '[nevus]'
        #     # learnable_prompt = 'A photo of a [no melanoma]'
        # else:
        #     print(diagnosis_label)
        #     print(file_id)

        if 'Melanoma' in diagnosis_label:
            label =torch.tensor([0,1])
            diagnosis_label = '[melanoma]'
            # learnable_prompt = 'A photo of a [melanoma]'
        elif 'nerves' in diagnosis_label:
            label =torch.tensor([1,0])
            diagnosis_label = '[nevus]'
            # learnable_prompt = 'A photo of a [no melanoma]'
        # else:
        #     print(diagnosis_label)
        #     print(file_id)

        
 
        clinical_prompt = 'A photo of a '+ diagnosis_label   #推理过程中不会用到，随便写的。


        return derm_img,label,clinical_prompt
        
