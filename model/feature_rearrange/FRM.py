import torch
import torch.nn as nn
import torch.nn.functional as F

class FRM(nn.Module):
    def __init__(self, FRM_scale):
        super().__init__()
        self.FRM_scale = FRM_scale

    def forward(self, token, features, patch_length):
        # head feature patch
        head_feat = features[:, :patch_length] # [batch_size, patch_num/4, embed_dim]
        # second feature patch
        body1_feat = features[:, patch_length:patch_length*2] # [batch_size, patch_num/4, embed_dim]
        # third feature patch
        body2_feat = features[:, patch_length*2:patch_length*3] # [batch_size, patch_num/4, embed_dim]
        # tail feature patch
        tail_feat = features[:, patch_length*3:patch_length*4] # [batch_size, patch_num/4, embed_dim]

        # fused_head_feat = torch.cat((head_feat, body1_feat), dim=1) # [batch_size, patch_num/2, embed_dim]
        # fused_tail_feat = torch.cat((body2_feat, tail_feat), dim=1) # [batch_size, patch_num/2, embed_dim]

        """
        fused_head_feat = gaussian_weight(fused_head_feat,)
        fused_tail_feat = gaussian_weight(fused_tail_feat,)
        
        head_feat = head_block(head_feat * self.FRM_scale + body1_feat)
        tail_feat = tail_block(tail_feat * self.FRM_scale + body2_feat)
        print("Current FRM_scale: ", self.FRM_scale)
        """

        # 新想法，丢弃head和tail特征，并直接对body1_feat和body2_feat进行特征学习，抓取更具辨识度的特征
        """
        body1_feat = head_block(body1_feat)
        body2_feat = tail_block(body2_feat)
        """

        # before global
        # method 1
        head_feat = head_feat * self.FRM_scale + body1_feat
        tail_feat = tail_feat * self.FRM_scale + body2_feat

        # method 2
        # head_feat = head_block(head_feat * self.FRM_scale + body1_feat)
        # tail_feat = tail_block(tail_feat * self.FRM_scale + body2_feat)

        
        # method 3
        """
        head_feat = head_feat * self.FRM_scale
        tail_feat = tail_feat * self.FRM_scale
        """
    
        # method 4
        """
        head_feat = head_feat * self.FRM_scale
        tail_feat - tail_feat * self.FRM_scale
        head_feat = torch.mean(torch.stack((head_feat, body1_feat)), dim=0)
        tail_feat = torch.mean(torch.stack((tail_feat, body2_feat)), dim=0)
        """
       
        # method 5
        # head_feat = head_feat * self.lower_FRM_scale + body1_feat * self.upper_FRM_scale
        # tail_feat = tail_feat * self.lower_FRM_scale + body2_feat * self.upper_FRM_scale
        
        FRM_feat = torch.cat((token, head_feat, body1_feat, body2_feat, tail_feat), dim=1)

        return FRM_feat

    """
    def gaussian_weight(image, cx, cy, sigma):
        height, width = image.size()[-2:]
        x, y = torch.meshgrid(torch.arange(width), torch.arange(height))
        x, y = x.float(), y.float()
        gaussian = torch.exp(-((x-cx)**2 + (y-cy)**2)/(2*sigma**2))
        weighted_image = image * gaussian
        spatial_correlation = torch.sum(weighted_image)
        return weighted_image, spatial_correlation
    """
    