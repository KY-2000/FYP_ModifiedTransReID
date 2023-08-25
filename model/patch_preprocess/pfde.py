import torch
import torch.nn as nn

#########################################################################################
# New Module
# Short range correlation transformer for occluded person reidentification

class PFDE(nn.Module):
    def __init__(self, patch_num, embed_dim):
        super(PFDE, self).__init__()
        self.patch_num = patch_num
        self.embed_dim = embed_dim
        self.LPDE = nn.Parameter(torch.ones(self.patch_num, self.embed_dim))
        # self.LPDE = None
        
    def forward(self, x):
        # if self.LPDE is None:
            N, L, D = x.shape
            """
            print("before: ",self.LPDE.shape)
            if(self.LPDE.dim == 2):
                self.LPDE = self.LPDE.unsqueeze(0).repeat(N,1,1)
                print("unsqueezed!")
            elif(self.LPDE.dim == 3 and self.LPDE.shape[0] != N):
                self.LPDE = self.LPDE.repeat(int(N/self.LPDE.shape[0]),1,1)
                print("testing started!")
            print("after: ",self.LPDE.shape)
            """
            enhanced_patch = torch.mul(self.LPDE, x)
            return enhanced_patch
            """
            if not isinstance(self.LPDE, nn.Parameter):
                LPDE_size = (batch_size, self.patch_num, self.embed_dim)
                self.LPDE = nn.Parameter(torch.ones(LPDE_size, device=x.device))
                model.register_parameter('LPDE', self.LPDE)

            
            patch_list = []
            for i in range(batch_size):
                # enhanced_patch = torch.stack((enhanced_patch, torch.mul(self.LPDE, x[i])), 0)
                patch_list.append(torch.mul(self.LPDE, x[i]))
                # enhanced_patch = (torch.mul(self.LPDE,x[i]))
                print("###############################################")
                print(i)
                print(patch_list[i])
            enhanced_patch = torch.stack(patch_list)
            """
