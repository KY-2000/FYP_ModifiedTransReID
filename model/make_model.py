import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from model.feature_rearrange.FRM import FRM
from model.feature_dropblock.Batch_DropBlock import BatchDrop
from model.patch_preprocess.patchdropout import PatchDropout

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1) # last dimension
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim) # 重组tensor
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous() # 第2和第3dimension调转
    x = x.view(batchsize, -1, dim)

    return x

##############################################################################
# weight initialization
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
# end of weight initialization
##############################################################################


##############################################################################
# resnet-50 backbone module
class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048 # 输入通道数
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False) # 分类
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x) # ResNet网络
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))
# End of initialization of resnet-50
###########################################################################################


###########################################################################################
# transformer without JPM
class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, 
                                                        stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT, PFDE_enable=cfg.MODEL.PFDE_enable,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE, patch_keep_rate=cfg.MODEL.PATCH_KEEP_RATE,
                                                        batch_drop_rate=cfg.MODEL.BATCH_DROP_RATE, batch_drop_layer=cfg.MODEL.BATCH_DROP_LAYER)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        # NEW MODULE, FRM Module
        self.FRM_enable = cfg.MODEL.FRM_enable
        print('FRM module: {}'.format(self.FRM_enable))
        if(self.FRM_enable):
            self.FRM_scale = cfg.MODEL.FRM_scale
            self.FRM = FRM(self.FRM_scale)
            print('using FRM scale is : {}'.format(self.FRM_scale))

        # NEW MODULE, Batch_DropBlock Module
        self.BDB_enable = cfg.MODEL.BDB_enable
        print('BDB module: {}'.format(self.BDB_enable))
        if(self.BDB_enable):
            self.num_x = self.base.patch_embed.num_x
            self.num_y = self.base.patch_embed.num_y
            print("num_x: ", self.num_x)
            print("num_y: ", self.num_y)
            self.batch_drop = BatchDrop(0.3, self.num_x, self.num_y)

        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            if(self.FRM_enable):
                self.classifier_FRM = nn.Linear(self.in_planes, self.num_classes, bias=False)
                self.classifier_FRM.apply(weights_init_classifier)
            if(self.BDB_enable):
                self.classifier_BDB = nn.Linear(self.in_planes, self.num_classes, bias=False)
                self.classifier_BDB.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        if(self.FRM_enable):
            self.bottleneck_FRM = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_FRM.bias.requires_grad_(False)
            self.bottleneck_FRM.apply(weights_init_kaiming)
        if(self.BDB_enable):
            self.bottleneck_BDB = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_BDB.bias.requires_grad_(False)
            self.bottleneck_BDB.apply(weights_init_kaiming)
        
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH # patches are shuffled into 'divide_length' groups
        print('using divide_length size:{}'.format(self.divide_length))

    def forward(self, x, label=None, cam_label= None, view_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)
        feat = self.bottleneck(global_feat)
        """
        features = self.base(x, cam_label=cam_label, view_label=view_label)
        cls_score_arr = []
        feat_arr = []
        feat_bn = []
        token = features[:, 0:1] # take all elements in "embed_dim" dimension
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        """

        """
        # FRM branch
        # Use same parameter as JPM
        if self.FRM_enable:
            b1_FRM_feat = self.b1(token, FRM(features[:, 1:], patch_length, self.FRM_scale))
            FRM_feat = b1_FRM_feat[:, 0]
            feat_arr.append(FRM_feat)
            FRM_feat_bn = self.bottleneck_FRM(FRM_feat)
            feat_bn.append(FRM_feat_bn)
        
        # BDB branch
        if self.BDB_enable:
            b1_BDB_feat = self.b1(self.batch_drop(global_feat))
            BDB_feat = b1_BDB_feat[:, 0]
            feat_arr.append(BDB_feat)
            BDB_feat_bn = self.bottleneck_BDB(BDB_feat)
            feat_bn.append(BDB_feat_bn)
        """

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
                # cls_score_BDB = self.classifier2(BDB_feat_fn, label)
            else:
                cls_score = self.classifier(feat)
                # cls_score_BDB = self.classifier2(BDB_feat_fn)

            return cls_score, global_feat  # global feature for triplet loss
            # return [cls_score, cls_score_BDB], [feat, BDB_feat_fn] 

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
                # return [feat, BDB_feat_fn]
            else:
                # print("Test with feature before BN")
                return global_feat
                # return [global_feat, BDB_feat]

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))
# end of initailzation of transformer without JPM 
###########################################################################################

class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH, 
                                                        patch_keep_rate=cfg.MODEL.PATCH_KEEP_RATE, PFDE_enable=cfg.MODEL.PFDE_enable,
                                                        SC_enable=cfg.MODEL.SC_enable, BDB_enable=cfg.MODEL.BDB_enable)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        # block_L11 = self.base.blocks[-2]
        block = self.base.blocks[-1]
        layer_norm = self.base.norm

        # NEW MODULE, FRM Module
        self.FRM_enable = cfg.MODEL.FRM_enable
        print('FRM module: {}'.format(self.FRM_enable))
        if(self.FRM_enable):
            # self.FRM_scale = nn.Parameter(torch.tensor([cfg.MODEL.FRM_scale]))
            self.FRM_scale = cfg.MODEL.FRM_scale
            # self.upper_FRM_scale = cfg.MODEL.FRM_scale * 2
            self.FRM = FRM(self.FRM_scale)
            print('FRM scale is : {}'.format(self.FRM_scale))
            # print('upper FRM scale is : {}'.format(self.upper_FRM_scale))
            self.b3 = nn.Sequential(
                copy.deepcopy(block),
                copy.deepcopy(layer_norm)
            )
        
        # NEW MODULE, Batch_DropBlock Module
        self.BDB_enable = cfg.MODEL.BDB_enable
        print('BDB module: {}'.format(self.BDB_enable))

        if(self.BDB_enable):
            self.num_x = self.base.patch_embed.num_x
            self.num_y = self.base.patch_embed.num_y
            self.batch_drop_rate = cfg.MODEL.BATCH_DROP_RATE
            self.batch_drop = BatchDrop(self.batch_drop_rate, self.num_x, self.num_y)
            print('using batch_drop_rate is : {}'.format(self.batch_drop_rate))
            self.b4 = nn.Sequential(
                copy.deepcopy(block),
                copy.deepcopy(layer_norm)
            )
            
        self.SC_enable = cfg.MODEL.SC_enable
        """
        print('SC module: {}'.format(self.SC_enable))
        if self.SC_enable:
            self.b5_L3 = nn.Sequential(
                copy.deepcopy(block),
                copy.deepcopy(layer_norm)
            )
            self.b5_L6 = nn.Sequential(
                copy.deepcopy(block),
                copy.deepcopy(layer_norm)
            )
            self.b5_L9 = nn.Sequential(
                copy.deepcopy(block),
                copy.deepcopy(layer_norm)
            )
        """
        

        """
        # PatchDropout using in feature learning
        self.feature_PD_enable = cfg.MODEL.FeatureDrop_enable
        print('Feature_Drop module: {}'.format(self.feature_PD_enable))
        if(self.feature_PD_enable):
            self.feature_PD = PatchDropout(keep_rate = cfg.MODEL.FEATURE_KEEP_RATE)
            print('using feature_keep_rate is : {}'.format(cfg.MODEL.FEATURE_KEEP_RATE))
            self.b5 = nn.Sequential(
                copy.deepcopy(block),
                copy.deepcopy(layer_norm)
            )
        """

        # global branch
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        # local feat. branch
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        """
        self.b3_head = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b3_tail = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        """ 
        
        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)
            if(self.FRM_enable):
                self.classifier_FRM = nn.Linear(self.in_planes, self.num_classes, bias=False)
                self.classifier_FRM.apply(weights_init_classifier)
            if(self.BDB_enable):
                self.classifier_BDB = nn.Linear(self.in_planes, self.num_classes, bias=False)
                self.classifier_BDB.apply(weights_init_classifier)
            """
            if(self.SC_enable):
                self.classifier_SC_L3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
                self.classifier_SC_L3.apply(weights_init_classifier)
                self.classifier_SC_L6 = nn.Linear(self.in_planes, self.num_classes, bias=False)
                self.classifier_SC_L6.apply(weights_init_classifier)
                self.classifier_SC_L9 = nn.Linear(self.in_planes, self.num_classes, bias=False)
                self.classifier_SC_L9.apply(weights_init_classifier)
            if(self.feature_PD_enable):
                self.classifier_FPD = nn.Linear(self.in_planes, self.num_classes, bias=False)
                self.classifier_FPD.apply(weights_init_classifier)
            """

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)
        if(self.FRM_enable):
            self.bottleneck_FRM = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_FRM.bias.requires_grad_(False)
            self.bottleneck_FRM.apply(weights_init_kaiming)
        if(self.BDB_enable):
            self.bottleneck_BDB = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_BDB.bias.requires_grad_(False)
            self.bottleneck_BDB.apply(weights_init_kaiming)
        """
        if(self.SC_enable):
            self.bottleneck_SC_L3 = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_SC_L3.bias.requires_grad_(False)
            self.bottleneck_SC_L3.apply(weights_init_kaiming)
            self.bottleneck_SC_L6 = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_SC_L6.bias.requires_grad_(False)
            self.bottleneck_SC_L6.apply(weights_init_kaiming)
            self.bottleneck_SC_L9 = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_SC_L9.bias.requires_grad_(False)
            self.bottleneck_SC_L9.apply(weights_init_kaiming)
        if(self.feature_PD_enable):
            self.bottleneck_FPD = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_FPD.bias.requires_grad_(False)
            self.bottleneck_FPD.apply(weights_init_kaiming)
        """
        # bottleneck 瓶颈层: https://blog.csdn.net/u011304078/article/details/80683985

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM # move 'shift_num' of patches to the end
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH # patches are shuffled into 'divide_length' groups
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange
            

    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'

        features = self.base(x, cam_label=cam_label, view_label=view_label)
        # print("Shape of feature is : ", features.shape) # batch_size, cls_token + patch, embed_dim
        cls_score_arr = []
        feat_arr = []
        feat_bn = []

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1] # [batch_size, 1, embed_dim], cls_token

        # FRM branch
        # Use same parameter as JPM
        """
        if self.FRM_enable:
            features = self.FRM(token, features[:, 1:], patch_length)
        """
        
        # global branch
        b1_feat = self.b1(features) # [batch_size, patch_num, embed_dim]
        global_feat = b1_feat[:, 0] # [batch_size, embed_dim], 取batch里每张照片的class token
        feat_arr.append(global_feat)

        feat = self.bottleneck(global_feat)
        feat_bn.append(feat)

        """
        if self.BDB_enable:
            features = self.batch_drop(features)
        """
        
        if self.BDB_enable:
            BDB_feat = self.batch_drop(features)
            b4_BDB_feat = self.b4(BDB_feat) # [batch_num, cls_token + patch_num, embed_dim]
            BDB_feat = b4_BDB_feat[:, 0]
            feat_arr.append(BDB_feat)
            BDB_feat_bn = self.bottleneck_BDB(BDB_feat) # [batch_num, embed_dim]
            feat_bn.append(BDB_feat_bn)
        
        # FRM branch
        # Use same parameter as JPM
        if self.FRM_enable:
            b3_FRM_feat = self.b3(self.FRM(token, features[:, 1:], patch_length))
            FRM_feat = b3_FRM_feat[:, 0]
            feat_arr.append(FRM_feat)
            FRM_feat_bn = self.bottleneck_FRM(FRM_feat)
            feat_bn.append(FRM_feat_bn)
        
        """
        # SC branch
        if self.SC_enable and SC_feat:
            b5_SC_L3_feat = self.b5_L3(SC_feat[0])
            b5_SC_L6_feat = self.b5_L6(SC_feat[1])
            b5_SC_L9_feat = self.b5_L9(SC_feat[2])

            SC_L3_feat = b5_SC_L3_feat[:, 0]
            SC_L6_feat = b5_SC_L6_feat[:, 0]
            SC_L9_feat = b5_SC_L9_feat[:, 0]

            feat_arr.append(SC_L3_feat)
            feat_arr.append(SC_L6_feat)
            feat_arr.append(SC_L9_feat)

            SC_L3_feat_bn = self.bottleneck_SC_L3(SC_L3_feat)
            SC_L6_feat_bn = self.bottleneck_SC_L6(SC_L6_feat)
            SC_L9_feat_bn = self.bottleneck_SC_L9(SC_L9_feat)

            feat_bn.append(SC_L3_feat_bn)
            feat_bn.append(SC_L6_feat_bn)
            feat_bn.append(SC_L9_feat_bn)
        
        # FPD branch
        if self.feature_PD_enable and self.training:
            FPD_feature = self.feature_PD(features)
            b5_FPD_feat = self.b5(FPD_feature)
            FPD_feat = b5_FPD_feat[:, 0]
            feat_arr.append(FPD_feat)
            FPD_feat_bn = self.bottleneck_FPD(FPD_feat)
            feat_bn.append(FPD_feat_bn)
        """
        
        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]
        feat_arr.append(local_feat_1)

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]
        feat_arr.append(local_feat_2)

        # lf_3
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]
        feat_arr.append(local_feat_3)

        # lf_4
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]
        feat_arr.append(local_feat_4)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        feat_bn.append(local_feat_1_bn / 4)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        feat_bn.append(local_feat_2_bn / 4)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        feat_bn.append(local_feat_3_bn / 4)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)
        feat_bn.append(local_feat_4_bn / 4)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_arr.append(cls_score)
                if(self.FRM_enable):
                    cls_score_FRM = self.classifier_FRM(FRM_feat_bn)
                    cls_score_arr.append(cls_score_FRM)
                if(self.BDB_enable):
                    cls_score_BDB = self.classifier_BDB(BDB_feat_bn)
                    cls_score_arr.append(cls_score_BDB)
                """
                if(self.SC_enable and SC_feat):
                    cls_score_SC_L3 = self.classifier_SC_L3(SC_L3_feat_bn)
                    cls_score_arr.append(cls_score_SC_L3)
                    cls_score_SC_L6 = self.classifier_SC_L6(SC_L6_feat_bn)
                    cls_score_arr.append(cls_score_SC_L6)
                    cls_score_SC_L9 = self.classifier_SC_L9(SC_L9_feat_bn)
                    cls_score_arr.append(cls_score_SC_L9)
                if(self.feature_PD_enable):
                    cls_score_FPD = self.classifier_FPD(FPD_feat_bn)
                    cls_score_arr.append(cls_score_FPD)
                """
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_arr.append(cls_score_1)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_arr.append(cls_score_2)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_arr.append(cls_score_3)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
                cls_score_arr.append(cls_score_4)
                
            return cls_score_arr, feat_arr
        else:
            if self.neck_feat == 'after':
                return feat_bn
                # [feat, BDB_feat_bn, FRM_feat_1_bn, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                feat[-4:] = feat[-4:] / 4
                return feat
                    # [global_feat, FRM_feat_1, BDB_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        # print(param_dict)
        for i in param_dict:
            print(i)
            try:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            except Exception as error:
                print('Caught this error: ' + repr(error))

        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

__factory_T_type = {
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID
}

def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model

"""
'vit_base_patch16_384_TransReID': vit_base_patch16_384_TransReID,
'vit_base_patch32_384_TransReID': vit_base_patch32_384_TransReID,
'vit_large_patch16_224_TransReID': vit_large_patch16_224_TransReID,
'vit_large_patch16_384_TransReID': vit_large_patch16_384_TransReID,
'vit_large_patch32_384_TransReID': vit_large_patch32_384_TransReID,
'vit_large_patch16_224_TransReID': vit_huge_patch16_224_TransReID,
'vit_huge_patch16_384_TransReID': vit_huge_patch16_384_TransReID,
'vit_huge_patch32_384_TransReID': vit_huge_patch32_384_TransReID
"""