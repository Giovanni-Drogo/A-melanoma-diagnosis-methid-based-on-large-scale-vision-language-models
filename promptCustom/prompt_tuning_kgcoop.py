from copy import deepcopy
from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn as nn
from torch.nn import functional as F

from .iid_loss import IID_loss
import os.path as osp
import clip

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from .cocoop import get_cocoop
from .custom_clip import get_coop
from .context_guided_coop import get_context_coop
# from data.datautils_domain import  build_dataset
# from data.cls_to_names import *
# from data.domain_datasets import domain_datasets

def discrepancy(out1, out2):
    # return torch.mean(torch.abs(nn.functional.softmax(out1, dim=1) - nn.functional.softmax(out2, dim=1)))
    return -torch.mean(torch.sum(out1*out2, dim=1))

# class BCEFocalLoss(torch.nn.Module):
#     def __init__(self, gamma=0, alpha=2, reduction='elementwise_mean'):
#         super().__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = reduction

#     def forward(self, _input, target):
#         # pt = torch.sigmoid(_input)
#         pt = _input
#         alpha = self.alpha
#         # loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
#         loss = - alpha ** (1 - pt)  * target * torch.log(pt+1e-5) -  (1 - target) * torch.log(1 - pt+1e-5)

#         if self.reduction == 'elementwise_mean':
#             loss = torch.mean(loss)
#         elif self.reduction == 'sum':
#             loss = torch.sum(loss)
        # return loss
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

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = False
    return optimizer

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def get_text(categories, device):
    # (31,77) 即每一个类别都是一个77维的向量
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}.") for c in categories]).to(device)
    return text_inputs

def test_time_tuning(model, inputs, pesu_label,context_feature, optimizer, args,ori_text_fea):
    for j in range(args.tta_steps):
        output,text_features = model(inputs)
        text_feature_learnable_list = []
        for i in range(len(pesu_label)):
            if pesu_label[i] == [1,0]:
               text_feature_learnable_list.append(text_features[0:1,:])
            else:
                text_feature_learnable_list.append(text_features[1:2,:])
        text_feature_learnable = torch.cat(text_feature_learnable_list,dim=0)
            

        # print(output.size(), pesu_label.size())
        pesu_label = pesu_label.cuda()
        output = nn.Softmax(dim=1)(output)
        # loss = discrepancy(output, pesu_label)   #原始loss
        # print(pesu_label)
        # print('********************************************')
        # print(output)
        cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
        loss_context_guided = 1-cos(text_features,ori_text_fea).mean()
        # print(loss_context_guided)
        # loss2 = F.cross_entropy(output,pesu_label) #+ loss_context_guided*0.2
        # cri = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10])).cuda()
        # cri_focal = BCEFocalLoss(gamma=2, alpha=0.65)
        weight = torch.tensor([1.0,3.0]).half().cuda()  #2.5
        cri = nn.CrossEntropyLoss(weight=weight)

        # loss = cri(output,pesu_label.half()) + 0.2*loss_context_guided
        loss = cri(output,torch.argmax(pesu_label,dim=1)) + 0.2*loss_context_guided
        # print(loss)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    return 

# def prompt_main(args,dataloader,all_output, iter_num, model):
#     # This codebase has only been tested under the single GPU setting
#     # assert int(args.gpu_id) is not None
#     text_features, prompt_model = main_worker(args,dataloader,all_output, iter_num, model=model)
#     text_features = text_features.detach()
#     return text_features, prompt_model

def main_worker(args, dataloader,clip_model, iter_num=0, max_iter=0,ori_text_fea=None):  #在main_worker内执行PromptLearner

    #dataloader是数据，all_output是伪标签————>改为真实label

    classnames = args.categories

    # model = get_cocoop(args.arch, classnames, int(args.gpu_id), args.num_ctx)
    """
    get_coop:coop
    """
    #修改后的模型，在Text_encoder上
    #初始化model
    if args.prompt_type == 'concept_coop':
        model = get_context_coop(args.arch, classnames, int(args.gpu_id), args.num_ctx, args.ctx)  
    elif args.prompt_type == 'coop':
        model = get_coop(args.arch, classnames, int(args.gpu_id), args.num_ctx, args.ctx)  #初始化prompt，返回logits_new,text_features。没用到dataloader，不用改
    model = model.cuda()
    
    # clip_model,preprocess = clip.load("ViT-B/16",device=args.gpu_id)
     

    if args.load_prompt is not None:
        # print("loading prompt")
        pretrained_ctx = torch.load(args.load_prompt)['ctx']
        assert pretrained_ctx.size()[0] == args.num_ctx
        with torch.no_grad():
            model.prompt_learner.ctx.copy_(pretrained_ctx)
            model.prompt_learner.ctx_init_state = pretrained_ctx

    for name, param in model.named_parameters():
        # print(name)
        if "prompt_learner" not in name:
            param.requires_grad_(False)


    model.reset_classnames(classnames, args.arch)
    trainable_param = model.prompt_learner.parameters()
    
    
    optimizer = torch.optim.SGD(trainable_param, args.lr_src, weight_decay=1e-3,momentum=0.9,nesterov=False)
    # 存个学习率， 方便以后来计算
    optimizer = op_copy(optimizer)

# #保存的模型是有delay的
#     if iter_num >= args.begin_lr_delay:
#         iter_num_ind =iter_num- args.begin_lr_delay     
#         # if args.dset == 'VisDA-2017':
#             # lr_scheduler(optimizer, iter_num=int(iter_num), max_iter=max_iter, power=2.25)   #学习率调度器
#         # else:
#         lr_scheduler(optimizer, iter_num=int(iter_num_ind), max_iter=max_iter, power=1.5)


    optim_state = deepcopy(optimizer.state_dict())
    print(optimizer.param_groups[0]['lr'])
    cudnn.benchmark = True
    # context_feature = context_emb(dataloader,clip_model,args)

    text_features = test_time_adapt_eval(dataloader, model,clip_model, optimizer, args,iter_num,ori_text_fea)  #更新Prompt，返回更新后text_encoder输出的text_feature 


    return text_features

    

def test_time_adapt_eval(dataloader, model,clip_model, optimizer, args,iter_num,ori_text_fea):
    with torch.no_grad():
        model.train()



    iter_test = iter(dataloader)
    # 对每个图像样本，和其对应的标签
    for i in range(len(dataloader)):
        images, pesu_label,context_prompt = next(iter_test)  #从数据加载器中加载当前批次的数据
        # pesu_label = all_output[tar_idx]

        if len(images.size()) > 4:
            assert images.size()[0] == 1
            images = images.squeeze(0)
        images = images.cuda(int(args.gpu_id), non_blocking=True)

        # ori_prompt = 

        context_prompt_tok = torch.cat([clip.tokenize(p) for p in context_prompt]).cuda()   #len(context_prompt)=4
        with torch.no_grad():
            context_feature = clip_model.encode_text(context_prompt_tok)
            context_feature = context_feature / context_feature.norm(dim=-1, keepdim=True)  #[batch,512]
            
        with torch.no_grad():
            model.train()

        # optimizer.load_state_dict(optim_state)
        test_time_tuning(model,images,pesu_label,context_feature,optimizer, args,ori_text_fea)
        with torch.no_grad():
            # with torch.cuda.amp.autocast():
            model.eval()
            output,text_features = model(images)
            
    #torch.save(model.prompt_learner.state_dict(), "./prompt_model.pt")
    filename = "prompt_model_"+str(iter_num)+".pt"
    torch.save(model.prompt_learner.state_dict(), osp.join(args.output_dir, filename))
    return text_features