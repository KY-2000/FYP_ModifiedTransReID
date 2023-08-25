import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class BatchDrop(nn.Module):
    """
    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        assert 0 < h_ratio <= 1.0
        assert 0 < w_ratio <= 1.0
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio
    
    def forward(self, x):
        # for CNN
        
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h-rh)
            sy = random.randint(0, w-rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx+rh, sy:sy+rw] = 0
            x = x * mask
        return x
    """
    
    def __init__(self, batch_drop_rate, num_x, num_y):
        super(BatchDrop, self).__init__()
        assert 0 < batch_drop_rate <= 1.0
        self.batch_drop_rate = batch_drop_rate
        self.num_x = num_x
        self.num_y = num_y
    
    def forward(self, x):
        # for transformer
        if self.batch_drop_rate < 1.0:
            # batch, length, dim
            N, L, D = x.shape

            # making cls mask (assumes that CLS is always the 1st element)
            cls_mask = torch.ones(N, 1, dtype=torch.int64, device=x.device) # keep class_token # [batch_size, 1]
            # generating patch mask
            patch_mask = self.get_mask(x) # [batch_size, patch_num]
            patch_mask = torch.cat([cls_mask, patch_mask], dim=1) # [batch_size, patch_num + cls_token]
            patch_mask = patch_mask.unsqueeze(-1).repeat(1,1,D) # [batch_size, patch_num + cls_token, embed_dim]
            x = torch.mul(x, patch_mask)
            # x = torch.gather(x, dim=1, index=patch_mask.unsqueeze(-1).repeat(1, 1, D))
            return x
        return x
    
    def get_mask(self, x):
        N, L, D = x.shape
        _L = L - 1 # patch length (without cls_token)
        mask_y = int(self.num_y * self.batch_drop_rate) # number of patch row that is needed to be masked
        patch_mask = torch.ones(N, _L, dtype=torch.int64, device=x.device)
        mask_start = random.randint(0, self.num_y - mask_y - 1) # starting patch row no.
        # print("mask_start: ", mask_start)
        patch_mask[:, mask_start * self.num_x : (mask_start + mask_y) * self.num_x] = 0.0
        return patch_mask

    """
    def forward(self, x):
        # for transformer
        if self.y_ratio < 1.0:
            # batch, length, dim
            N, L, D = x.shape

            # making cls mask (assumes that CLS is always the 1st element)
            cls_mask = torch.ones(1, D, dtype=torch.int64, device=x.device) # keep class_token # [1, embed_dim]
            # print("cls_mask: ", cls_mask.shape)
            # generating patch mask
            patch_mask = self.get_mask(x) # [patch_num, embed_dim]
            # print("patch_mask: ", patch_mask.shape)
            patch_mask = torch.cat([cls_mask, patch_mask], dim=0) # [patch_num + cls_token, embed_dim]
            # print("patch_mask_concat: ", patch_mask.shape)            
            patch_mask = patch_mask.unsqueeze(0).repeat(N,1,1) # [batch_size, patch_num + cls_token, embed_dim]
            # print("patch_mask_unsqueeze: ", patch_mask.shape)
            # x = x * patch_mask
            x = torch.mul(x, patch_mask)
            # x = torch.gather(x, dim=1, index=patch_mask.unsqueeze(-1).repeat(1, 1, D))
            return x
        return x
    
    def get_mask(self, x):
        N, L, D = x.shape
        _L = L - 1 # patch length (without cls_token)
        mask_y = int(self.num_y * self.y_ratio) # patch row no. that is needed to be masked
        patch_mask = torch.ones(_L, D, dtype=torch.int64, device=x.device)
        mask_start = random.randint(0, self.num_y - mask_y - 1)
        print("start: ", mask_start * self.num_x)
        print("end: ",(mask_start + mask_y) * self.num_x)
        patch_mask[mask_start * self.num_x : (mask_start + mask_y) * self.num_x, :] = 0
        # print(patch_mask[mask_start * self.num_x - 1: (mask_start + mask_y) * self.num_x + 1, :])
        return patch_mask
    """