
# import clip
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import sys
import os
import random
import argparse
import time
from data import Derm7ptDatset,PH2Datset
from torch.utils.data import DataLoader
from promptCustom import prompt_tuning,clip,prompt_tuning_v2,prompt_test                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
import os.path as osp
from promptCustom.cocoop import get_cocoop
from promptCustom.custom_clip import get_coop
from promptCustom.context_guided_coop import get_context_coop
from sklearn.metrics import roc_auc_score,f1_score
from torch.utils.tensorboard import SummaryWriter
local_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize',type=int, default=128,help='input batch size of training [default:4]')
    parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
    parser.add_argument('--dataset',type=str,default='ph2')
    parser.add_argument('--data_dir',type=str, default='/home/wenyu/Downloads/release_v0/')
    # parser.add_argument('--start_epoch',type=int,default=1)
    # parser.add_argument('--end_epoch',type=int,default=100)
    parser.add_argument('--model_name',type=str,default='',help='')
    parser.add_argument('--mode',type=str,default='test')
    parser.add_argument('--save_dir',type=str,default='')
    parser.add_argument('--log_dir',type=str,default='')
    parser.add_argument('--image_size',type=str,default='')
    parser.add_argument('--date', default=local_time.split(' ')[0], type=str)
    parser.add_argument('--time', default=local_time.split(' ')[1], type=str)
    parser.add_argument('--arch', type=str, default='ViT-B/16') # clip模型
    parser.add_argument('--ctx', type=str, default='a_photo_of_a') #初始提示词
    parser.add_argument('--num_ctx', type=int, default=4) #初始提示词单词数
    parser.add_argument('--load_prompt', type=str, default=None)
    parser.add_argument('--learned_cls',default=True)
    parser.add_argument('--categories',default=['nevus','melanoma'])
    parser.add_argument('--output_dir',type=str,default='/home/wenyu/Clip_prompt/sava/')
    parser.add_argument('--gpu_id',type=int,default=0)
    parser.add_argument('--lr_src',type=float,default=0.001)
    parser.add_argument('--tta_steps',type=int,default=1)
    parser.add_argument('--prompt_type',type=str,default='concept_coop')
    parser.add_argument('--max_epoch',type=int,default=50)
    parser.add_argument('--clip_update_bengin',type=int,default=0)
    parser.add_argument('--prompt_path',type=str,default='/home/wenyu/Clip_prompt/sava/seed3/prompt_model_78.pt')
    parser.add_argument('--clip_path',type=str,default='/home/wenyu/Clip_prompt/sava/seed3/Image_encoder_tuning78.pt')
    parser.add_argument('--image_enc',type=int,default=1)

    args = parser.parse_args()
    scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()  #CLIP中使用的scale

    if args.dataset == 'derm7pt':

        dir_images=os.path.join(args.data_dir, 'images/')
        metadata_df=pd.read_csv(os.path.join(args.data_dir, 'meta/meta.csv'))
        # train_indexes=list(pd.read_csv(os.path.join(args.data_dir, 'meta/train_indexes_twoclass.csv'))['indexes'])
        # valid_indexes=list(pd.read_csv(os.path.join(args.data_dir, 'meta/validation_indexes_twoclass.csv'))['indexes'])
        # test_indexes=list(pd.read_csv(os.path.join(args.data_dir, 'meta/test_indexes_twoclass.csv'))['indexes'])
        train_indexes=list(pd.read_csv(os.path.join(args.data_dir, 'meta/train_indexes.csv'))['indexes'])
        valid_indexes=list(pd.read_csv(os.path.join(args.data_dir, 'meta/valid_indexes.csv'))['indexes'])
        test_indexes=list(pd.read_csv(os.path.join(args.data_dir, 'meta/test_indexes.csv'))['indexes'])
        
        train_dataset = Derm7ptDatset(img_dir=dir_images,img_info=metadata_df,file_list=train_indexes,img_size=args.image_size,is_test=False)
        valid_dataset = Derm7ptDatset(img_dir=dir_images,img_info=metadata_df,file_list=valid_indexes,img_size=args.image_size,is_test=True)
        test_dataset = Derm7ptDatset(img_dir=dir_images,img_info=metadata_df,file_list=test_indexes,img_size=args.image_size,is_test=True)

        train_dataloader = DataLoader(train_dataset,batch_size=args.batchsize,shuffle=True)
        valid_dataloader = DataLoader(valid_dataset,batch_size=1)
        test_dataloader = DataLoader(test_dataset,batch_size =1)
    elif args.dataset == 'ph2':
        dir_images = '/home/wenyu/Downloads/PH2Dataset/PH2Dataset/PH2Datasetimages/'
        list_dir = '/home/wenyu/Downloads/PH2Dataset/PH2Dataset/test_indexes.csv'
        
        test_dataset = PH2Datset(img_dir=dir_images,list_dir=list_dir,is_test=True)
        test_dataloader = DataLoader(test_dataset,batch_size =1)


    else:
        print('There in no this dataset name')
        raise NameError
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    clip_model_vit16,preprocess = clip.load("ViT-B/16",device=device)
    #load saved model
    args.load_prompt =args.prompt_path
    text_fea = prompt_test.main_worker(args,test_dataloader)
    clip_model_vit16,preprocess = clip.load("ViT-B/16",device=device)
    if args.image_enc==1:
        clip_model_vit16.load_state_dict(torch.load(args.clip_path))
    


    with torch.no_grad():
            output_list = []
            output_label_list = []
            label_list = []
            correct_num = 0
            iter_test = iter(test_dataloader)
            for i in range(len(test_dataloader)):
                images, pesu_label,_ = next(iter_test) 
                images = images.cuda(int(args.gpu_id), non_blocking=True)
                images_fea = clip_model_vit16.encode_image(images)
                images_fea = images_fea/images_fea.norm(dim=1,keepdim=True)
                    # images_fea_tuning = net_FC(images_fea)
                    # images_fea_tuning = images_fea_tuning/images_fea_tuning.norm(dim=1,keepdim=True)
                out = scale*images_fea.half() @ text_fea.half().T
                out_so = nn.Softmax(dim=1)(out)
                out_one_hot = torch.argmax(out_so)
                label_one_hot = torch.argmax(pesu_label)
                if out_one_hot==label_one_hot:
                    correct_num +=1
                output_list.append(out_so[:,1].cpu().numpy())
                output_label_list.append(out_one_hot.cpu().numpy())
                label_list.append(pesu_label[:,1].cpu().numpy())

            test_auc = roc_auc_score(label_list,output_list)
            test_acc = correct_num/len(test_dataloader)
            test_f1 = f1_score(label_list,output_label_list,average='weighted')  #
            print("Test_ACC: %0.4f, Test_AUC: %0.4f,Test_F1-score: %0.4f" %(test_acc,test_auc,test_f1))

