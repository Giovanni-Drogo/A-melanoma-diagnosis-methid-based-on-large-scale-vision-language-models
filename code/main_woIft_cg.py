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
from data import Derm7ptDatset
from torch.utils.data import DataLoader
from promptCustom import prompt_tuning_wocg,clip
import os.path as osp
from sklearn.metrics import roc_auc_score,f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
local_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())

def set_seed(seed: int=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)                                                                                                                                            
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# set_seed(2024)
def get_text(categories, device):
    # (31,77) 即每一个类别都是一个77维的向量
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}.") for c in categories]).to(device)
    return text_inputs
def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer
cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
# cri = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10])).cuda()
weight = torch.tensor([1.0,3.0]).half().cuda()
cri = nn.CrossEntropyLoss(weight=weight).cuda()


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        # pt = torch.sigmoid(_input)
        pt = _input
        alpha = self.alpha
        loss = - alpha * (1 - pt) * target * torch.log(pt+1e-5) - (1 - alpha) * pt  * (1 - target) * torch.log(1 - pt+1e-5)
        # loss = - alpha ** (1 - pt)  * target * torch.log(pt+1e-5) -  (1 - target) * torch.log(1 - pt+1e-5)

        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
# class BCEFocalLoss(torch.nn.Module):
#     def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
#         super(BCEFocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = reduction
 
#     def forward(self, predict, target):
#         pt = predict
#         loss = - ((1 - self.alpha) * ((1 - pt+1e-5) ** self.gamma) * (target * torch.log(pt+1e-5)) +  self.alpha * (
#                 (pt+1e-5) ** self.gamma) * ((1 - target) * torch.log(1 - pt+1e-5)))
 
#         if self.reduction == 'mean':
#             loss = torch.mean(loss)
#         elif self.reduction == 'sum':
#             loss = torch.sum(loss)
#         return loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize',type=int, default=128,help='input batch size of training [default:4]')
    
    parser.add_argument('--dataset',type=str,default='derm7pt')
    parser.add_argument('--data_dir',type=str, default='/home/wenyu/Downloads/release_v0/')
    # parser.add_argument('--start_epoch',type=int,default=1)
    # parser.add_argument('--end_epoch',type=int,default=100)
    parser.add_argument('--model_name',type=str,default='',help='')
    parser.add_argument('--mode',type=str,default='train')
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
    parser.add_argument('--output_dir',type=str,default='/home/wenyu/Clip_prompt/sava/seed1/')
    parser.add_argument('--gpu_id',type=int,default=0)
    parser.add_argument('--lr_src',type=float,default=0.03)
    parser.add_argument('--lr_ft',type=float,default=5e-5,help='learning rate')
    parser.add_argument('--tta_steps',type=int,default=1)
    parser.add_argument('--prompt_type',type=str,default='concept_coop')
    parser.add_argument('--max_epoch',type=int,default=80)
    parser.add_argument('--clip_update_bengin',type=int,default=0)
    parser.add_argument('--FC_update_bengin',type=int,default=1)
    parser.add_argument('--isFC_tuning',default=True)   #是否添加可微调的FC
    parser.add_argument('--begin_lr_delay',type=int,default=40)
    parser.add_argument('--seed',type=int,default=2024)
    args = parser.parse_args()
    set_seed(args.seed)
    #写入tensorboard
    path_log = os.path.join(args.output_dir+'logs/',args.dataset+'_promptlr_'+str(args.lr_src)+'_ftlr_'+str(args.lr_ft)+ args.date+'_'+ args.time)
    writer = SummaryWriter(path_log)
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

    else:
        print('There in no this dataset name')
        raise NameError
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    clip_model_vit16,preprocess = clip.load("ViT-B/16",device=device)
    # clip_model_vit32,preprocess = clip.load("ViT-B/32",device=device)
    # clip_model_rn50,preprocess =clip.load("ViT-B/32",device=device)

    # param_group = []
    # learning_rate = args.lr_src*10
    # if args.isFC_tuning ==True:
    #     net_FC = network.feat_fc(in_dim=512,out_dim=512).cuda()
    #     for k,v in net_FC.named_parameters():
    #         param_group += [{'params': v, 'lr': learning_rate}]
    # optimizer =  torch.optim.SGD(param_group)
    # optimizer = op_copy(optimizer)
    # net_FC.eval()
   
    max_iter = args.max_epoch
    iter_num = 0
    text_fea = prompt_tuning_wocg.main_worker(args,train_dataloader,clip_model_vit16,iter_num=0,max_iter=max_iter)
    args.load_prompt = osp.join(args.output_dir,"prompt_model_"+str(iter_num)+".pt")
    iter_num = 1
    interval_iter = max_iter//args.max_epoch
    best_acc = 0.0
    best_acc_iter_num = 0

    while iter_num < args.FC_update_bengin:   #只更新Prompt
        ##Prompt Learning,不要动!
        print(iter_num)
        text_fea = prompt_tuning_wocg.main_worker(args,train_dataloader,clip_model_vit16,iter_num=iter_num,max_iter=max_iter)
        args.load_prompt = osp.join(args.output_dir,"prompt_model_"+str(iter_num)+".pt")   

        #test and val
        with torch.no_grad():
            output_list = []
            label_list = []
            output_label_list = []
            correct_num = 0
            iter_test = iter(test_dataloader)
            for i in range(len(test_dataloader)):
                images, pesu_label,_ = next(iter_test) 
                images = images.cuda(int(args.gpu_id), non_blocking=True)
                images_fea = clip_model_vit16.encode_image(images)
                images_fea = images_fea/images_fea.norm(dim=1,keepdim=True)
                out = scale*images_fea.half() @ text_fea.half().T
                out_so = nn.Softmax(dim=1)(out)
                out_one_hot = torch.argmax(out_so)
                label_one_hot = torch.argmax(pesu_label)
                if out_one_hot==label_one_hot:
                    correct_num +=1
                output_list.append(out_so[:,1].cpu().numpy())
                output_label_list.append(out_one_hot.cpu().numpy())
                label_list.append(pesu_label[:,1].cpu().numpy())

            output_list_val = []
            label_list_val = []
            output_label_list_val = []
            correct_num_val = 0
            iter_test_val = iter(valid_dataloader)
            for i in range(len(valid_dataloader)):
                images, pesu_label,_ = next(iter_test_val) 
                images = images.cuda(int(args.gpu_id), non_blocking=True)
                images_fea = clip_model_vit16.encode_image(images)
                images_fea = images_fea/images_fea.norm(dim=1,keepdim=True)
                out = scale*images_fea.half() @ text_fea.half().T
                out_so = nn.Softmax(dim=1)(out)
                out_one_hot = torch.argmax(out_so)
                label_one_hot = torch.argmax(pesu_label)
                if out_one_hot==label_one_hot:
                    correct_num_val +=1
                output_list_val.append(out_so[:,1].cpu().numpy())
                output_label_list_val.append(out_one_hot.cpu().numpy())
                label_list_val.append(pesu_label[:,1].cpu().numpy())
                
                    
            test_auc = roc_auc_score(label_list,output_list)
            test_acc = correct_num/len(test_dataloader)
            test_f1 = f1_score(label_list,output_label_list,average='weighted') #
                
            val_auc = roc_auc_score(label_list_val,output_list_val)
            val_acc = correct_num_val/len(valid_dataloader)
            val_f1 = f1_score(label_list_val,output_label_list_val,average='weighted') #
                             
            writer.add_scalar('Test_ACC',test_acc,iter_num)
            writer.add_scalar('Test_AUC',test_auc,iter_num)
            writer.add_scalar('Test_F1',test_f1,iter_num)
            writer.add_scalar('Val_ACC',val_acc,iter_num)
            writer.add_scalar('Val_AUC',val_auc,iter_num)
            writer.add_scalar('Val_F1',val_f1,iter_num)

                #记录prompt过程中best acc和对应的iter_num
            if test_acc>best_acc:
                best_acc = test_acc
                best_acc_iter_num = iter_num
            print("Test_ACC: %0.4f, Test_AUC: %0.4f,Test_F1-score: %0.4f,Best_ACC: %0.4f,best_acc_iter_num: %d" %(test_acc,test_auc,test_f1,best_acc,best_acc_iter_num))
        iter_num += 1
        

            
    
    
    # optimizer = torch.optim.AdamW(trainable_param, 3e-5, weight_decay=1e-5)
    # optimizer = op_copy(optimizer)
    # scheduler = CosineAnnealingLR(optimizer,T_max=10,eta_min=1e-5)  #余弦退火
    # print(optimizer.param_groups[0]['lr'])

    while iter_num >= args.FC_update_bengin and iter_num <= max_iter:
        print("Begin fine-tuning!")
        # print(optimizer.param_groups[0]['lr'])
        print(iter_num)
        #加载best_acc对应的prompt! 
        #best_acc_iter_num = 138   #Prompt Learning过程中得到的,如果从中途开始运行,需要手动设置best_acc_iter_num  
        # if iter_num >= args.FC_update_bengin:  #开始微调Image_encoder         
        # if iter_num == 150:#上一阶段best_acc_iter_num对应的模型
        #     print("best model loading")
        #     args.load_prompt = osp.join(args.output_dir,"prompt_model_"+str(best_acc_iter_num)+".pt") # 
        text_fea = prompt_tuning_wocg.main_worker(args,train_dataloader,clip_model_vit16,iter_num=iter_num,max_iter=max_iter)
        args.load_prompt = osp.join(args.output_dir,"prompt_model_"+str(iter_num)+".pt")  

        # for name, param in clip_model_vit16.named_parameters():
        #     if "visual.ln_post" in name:
        #         param.requires_grad_(True)
        #     else:
        #         param.requires_grad_(False)
            # optimizer = torch.optim.SGD(trainable_param, args.lr_src*0.05, weight_decay=1e-4,momentum=0.9,nesterov=False)
        # trainable_param = clip_model_vit16.visual.ln_post.parameters()
        # optimizer = torch.optim.SGD(trainable_param, args.lr_ft, weight_decay=1e-4,momentum=0.9,nesterov=False)

    
        # clip_model_vit16.train()

        # iter_train = iter(train_dataloader)
        # for i in range(len(train_dataloader)):
                
        #     clip_model_vit16.train()
        #     images, pesu_label,_ = next(iter_train) 
        #     pesu_label = pesu_label.cuda()
        #     images = images.cuda(int(args.gpu_id), non_blocking=True)
        #     images_fea = clip_model_vit16.encode_image(images)
        #     images_fea = images_fea/images_fea.norm(dim=1,keepdim=True)   
        #     optimizer.zero_grad()   
        #     output_train = scale*images_fea.half() @ text_fea.half().T
        #     out_so_train = nn.Softmax(dim=1)(output_train)
        #         # out_one_hot_train = torch.argmax(out_so_train)

        #     # loss2 =cri(out_so_train,pesu_label.cuda().half())
        #     # loss = loss2#+0.2*loss1


        #     # cri_focal = BCEFocalLoss(gamma=2, alpha=0.65)
        #     loss = cri(out_so_train,torch.argmax(pesu_label,dim=1))
            
        # # loss = cri(output,pesu_label.half()) + 0.2*loss_context_guided
        #     # loss = cri_focal(out_so_train,pesu_label.half())


        #     optimizer.zero_grad()
        #     loss.backward(retain_graph=True) 
        #     optimizer.step()

            # scheduler.step()

                # for name, param in clip_model_vit16.named_parameters():  #检查梯度
                #     if "ln_post" in name:
                #         print(param.grad)
                                
            #不更新
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
            if test_acc>best_acc:
                best_acc = test_acc
                best_acc_iter_num = iter_num
            # torch.save(clip_model_vit16.state_dict(),osp.join(args.output_dir,"Image_encoder_tuning"+str(iter_num)+".pt"))
                
            output_list_val = []
            label_list_val = []
            output_label_list_val = []
            correct_num_val = 0
            iter_test_val = iter(valid_dataloader)
            for i in range(len(valid_dataloader)):
                images, pesu_label,_ = next(iter_test_val) 
                images = images.cuda(int(args.gpu_id), non_blocking=True)
                images_fea = clip_model_vit16.encode_image(images)
                images_fea = images_fea/images_fea.norm(dim=1,keepdim=True)
                out = scale*images_fea.half() @ text_fea.half().T
                out_so = nn.Softmax(dim=1)(out)
                out_one_hot = torch.argmax(out_so)
                label_one_hot = torch.argmax(pesu_label)
                if out_one_hot==label_one_hot:
                    correct_num_val +=1
                output_list_val.append(out_so[:,1].cpu().numpy())
                output_label_list_val.append(out_one_hot.cpu().numpy())
                label_list_val.append(pesu_label[:,1].cpu().numpy())
                
                    
            
            val_auc = roc_auc_score(label_list_val,output_list_val)
            val_acc = correct_num_val/len(valid_dataloader)
            val_f1 = f1_score(label_list_val,output_label_list_val,average='weighted') #
            

            writer.add_scalar('Val_ACC',val_acc,iter_num)
            writer.add_scalar('Val_AUC',val_auc,iter_num)
            writer.add_scalar('Val_F1',val_f1,iter_num)
              
            writer.add_scalar('Test_ACC',test_acc,iter_num)
            writer.add_scalar('Test_AUC',test_auc,iter_num)
            writer.add_scalar('Test_F1',test_f1,iter_num)
            print("Test_ACC: %0.4f, Test_AUC: %0.4f,Test_F1-score: %0.4f,Best_ACC: %0.4f,best_acc_iter_num: %d" %(test_acc,test_auc,test_f1,best_acc,best_acc_iter_num))
        iter_num += 1
            







        

        #hard-prompt
        # text = get_text(args.categories,device)
        # text_fea = clip_model.encode_text(text)
        # text_fea = text_fea/text_fea.norm(dim=1,keepdim=True)
        
    
        # output_all=  torch.cat(output_list,dim=0)
        # label_all=  torch.cat(label_list,dim=0)
        
        # writer.add_scalar('ACC',correct_num/len(test_dataloader),iter_num)
        # writer.add_scalar('AUC',auc,iter_num)
        
    



    # iter_test = iter(test_dataloader)
    # # 对每个图像样本，和其对应的标签
    # output_list = [] 
    # for i in range(len(test_dataloader)):
    #     images, pesu_label,_ = next(iter_test)  #从数据加载器中加载当前批次的数据
    #     if len(images.size()) > 4:
    #         assert images.size()[0] == 1
    #         images = images.squeeze(0)
    #     with torch.no_grad():
    #         images = images.cuda(int(args.gpu_id), non_blocking=True)
    #         images_fea = clip_model.encode_image(images)
    #         images_fea = images_fea/images_fea.norm(dim=1,keepdim=True)
    #         out = images_fea.half() @ text_fea.T
    #         out_so = nn.Softmax(dim=1)(out)
    #         output_list.append(out)
    
    print('end')


    







        
