import torch
import torch.nn as nn
import torch.nn.functional as F
from .update import BasicMultiUpdateBlock_GRU1 , BasicMultiUpdateBlock_GRU2
from .extractor import MultiBasicEncoder, Feature, Convnext_Encoder_stereo, Convnext_Encoder_stereo_cnet
from .geometry import Combined_Geo_Encoding_Volume
from .submodule import *
import time
import pdb
from .corr import CorrBlock1D_Cost_Volume
from .utils.utils import updisp4,Map



class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 8, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))



        self.feature_att_8 = FeatureAtt(in_channels*2, 64)
        self.feature_att_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_32 = FeatureAtt(in_channels*6, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels*2, 64)

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)

        return conv


class IGEVStereo(nn.Module): # new model 
    def __init__(self, args):
        super().__init__()
        args = Map(args)
        self.args = args    
        #print("-----------",args)
        # self.args.K_value = 3
        context_dims = args.hidden_dims # [120,120,120] 
        # ---------------------------------------------------------------------------------
        # for concat based volume
        self.feature_concat = Feature()

        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn="batch", downsample=args.n_downsample)

        # GRU 2 update block
        self.update_block2_0 = BasicMultiUpdateBlock_GRU2(self.args, hidden_dims=args.hidden_dims, cor_dim=12,concat_dim=144, K_value=self.args.K_value)
        self.update_block2_1 = BasicMultiUpdateBlock_GRU2(self.args, hidden_dims=args.hidden_dims, cor_dim=4, concat_dim=144,K_value=self.args.K_value)
        self.update_block2_2 = BasicMultiUpdateBlock_GRU2(self.args, hidden_dims=args.hidden_dims, cor_dim=2,concat_dim=144,K_value=1)
        # convert cnet to zqr
        self.context_zqr_convs_2 = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])
        
        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
            )

        
        # ---------------------------------------------------------------------------------
        # For similarity based volume 
        self.feature_corr= Convnext_Encoder_stereo()
        self.cnet_convnext= Convnext_Encoder_stereo_cnet()
        self.uniform_sampler = UniformSampler()
        
        # GRU 1 update block
        self.update_block1_0 = BasicMultiUpdateBlock_GRU1(self.args, hidden_dims=args.hidden_dims, cor_dim=12,
                                                    disp_dim=self.args.K_value+1, K_value=self.args.K_value)
        self.update_block1_1 = BasicMultiUpdateBlock_GRU1(self.args, hidden_dims=args.hidden_dims, cor_dim=4,
                                                    disp_dim=self.args.K_value+1, K_value=self.args.K_value)
        self.update_block1_2 = BasicMultiUpdateBlock_GRU1(self.args, hidden_dims=args.hidden_dims, cor_dim=2, disp_dim=1+1,
                                                    K_value=1)
        self.context_zqr_convs_1 = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        
        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)

        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)
      
        self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = FeatureAtt(8, 96)
        self.cost_agg = hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)
        

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    def upsample_disp_GRU1(self, disp, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = disp.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)
        up_disp = F.unfold(factor * disp, [3, 3], padding=1)
        up_disp = up_disp.view(N, D, 9, 1, 1, H, W)
        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, D, factor * H, factor * W)

    def upsample_disp_GRU2(self, disp, mask_feat_4, stem_2x):

        xspx = self.spx_2_gru(mask_feat_4, stem_2x)
        spx_pred = self.spx_gru(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)

        return up_disp
    
    
    def forward(self, image1, image2, image3, flow_init=None):
        """ Estimate disparity between pair of frames """
        test_mode = not self.training
        #iters=12
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous() #RGB HighRes
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous() #RGB to Gray LowRes
        image3 = (2 * (image3 / 255.0) - 1.0).contiguous() #Gray LowRes     
        
        
        features_concat1 = self.feature_concat(image1)   # return [x4, x8, x16, x32]  
        # x4torch.Size([1, 48, 80, 184]) x8:torch.Size([1, 64, 40, 92]) x16"torch.Size([1, 192, 20, 46]) x32:torch.Size([1, 160, 10, 23])
        features_concat2 = self.feature_concat(image3)

        features_corr1 = self.feature_corr(image2)
        features_corr2 = self.feature_corr(image3)

        # build the concat based volume (GWC_Volume) 
        stem_2x = self.stem_2(image1) # c3-32,r1/2
        stem_4x = self.stem_4(stem_2x) # c32-48,r1/4
        stem_2y = self.stem_2(image3)
        stem_4y = self.stem_4(stem_2y)
        
        features_concat1[0] = torch.cat((features_concat1[0], stem_4x), 1) # x_4,stem
        features_concat2[0] = torch.cat((features_concat2[0], stem_4y), 1)
    
        match_concat1 = self.desc(self.conv(features_concat1[0]))
        match_concat2 = self.desc(self.conv(features_concat2[0]))
        
        gwc_volume = build_gwc_volume(match_concat1, match_concat2, self.args.max_disp//4, 8)
        gwc_volume = self.corr_stem(gwc_volume)
        gwc_volume = self.corr_feature_att(gwc_volume, features_concat1[0])
        geo_encoding_volume = self.cost_agg(gwc_volume, features_concat1)
        
        # Init disp from geometry encoding volume
        prob_concat = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
        init_disp = disparity_regression(prob_concat, self.args.max_disp//4)

        del prob_concat, gwc_volume

        if not test_mode:
            xspx = self.spx_4(features_concat1[0])
            xspx = self.spx_2(xspx, stem_2x)
            spx_pred = self.spx(xspx)
            spx_pred = F.softmax(spx_pred, 1)
        
        
        # shared context feature , channel size [128, 128, 128]
        cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
        

        net_list1 = [torch.tanh(x[0]) for x in cnet_list]
        inp_list1 = [torch.relu(x[1]) for x in cnet_list]
        inp_list1 = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list1, self.context_zqr_convs_1)]

        net_list2 = [torch.tanh(x[0]) for x in cnet_list]
        inp_list2 = [torch.relu(x[1]) for x in cnet_list]
        inp_list2 = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list2, self.context_zqr_convs_2)]
            
            
        # correlation volume 
        corr_block = CorrBlock1D_Cost_Volume
        features_corr1, features_corr2 = features_corr1.float(), features_corr2.float()
        init_corr_volume = build_correlation_volume(features_corr1, features_corr2, self.args.max_disp // 4) #([1, 1, 48, 80, 184])
        corr_fn = corr_block(init_corr_volume, num_levels=2) 
        corr_volume = init_corr_volume.squeeze(1)
        
        prob_corr = F.softmax(corr_volume, dim=1)
        
        value, disp_topk = torch.topk(prob_corr, dim=1, k=self.args.K_value)
       

        geo_block = Combined_Geo_Encoding_Volume
        geo_fn = geo_block(geo_encoding_volume.float(), radius=self.args.corr_radius, num_levels=self.args.corr_levels)

        concat_disp = init_disp

        disp_pred_list_GRU1 = []
        disp_pred_list_GRU2 = []
        
        context = net_list1[0] # context for GRU1 
        

        # GRUs iterations to update disparity
        iters = self.args.valid_iters if test_mode else self.args.train_iters
        for itr in range(iters):
            # itr disp index for sim volume 
            disp_topk = disp_topk.detach().float()
            local_cost_list = []
            corr_list = []

            # itr disp for concat volume (value)
            concat_disp=concat_disp.detach()


            # Multi-peak  Lookup in simialrity based volume 
            # Search Radius
            if itr < 6:
                rt = 12
            elif itr < 16:
                rt = 4
            else:
                rt = 2
            
            for k in range(disp_topk.shape[1]):
                corr_disp = disp_topk[:, k, :, :].unsqueeze(dim=1) # the disp here is index 
                corr_pyramid = corr_fn(corr_disp, rt)
                local_cost_list.append(corr_pyramid[0])
                corr = torch.cat(corr_pyramid, dim=1)
                corr_list.append(corr)  
            local_cost = torch.cat(local_cost_list, dim=1)
            corr = torch.cat(corr_list, dim=1)
            geo_feat_1= geo_fn(concat_disp)

            corr_combined=torch.cat([corr,geo_feat_1],dim=1)
            disp_combined=torch.cat((disp_topk,concat_disp),dim=1)
            

            if itr < 6:
                net_list1, up_mask, delta_local_cost = self.update_block1_0(net_list1, inp_list1, corr_combined, disp_combined,
                                                                            context,
                                                                            iter32=self.args.n_gru_layers == 3,
                                                                            iter16=self.args.n_gru_layers >= 2)   
            elif itr < 16:
                net_list1, up_mask, delta_local_cost = self.update_block1_1(net_list1, inp_list1, corr_combined, disp_combined,
                                                                            context,
                                                                            iter32=self.args.n_gru_layers == 3,
                                                                            iter16=self.args.n_gru_layers >= 2)
            else:
                net_list1, up_mask, delta_local_cost = self.update_block1_2(net_list1, inp_list1, corr_combined, disp_combined,
                                                                            context,
                                                                            iter32=self.args.n_gru_layers == 3,
                                                                            iter16=self.args.n_gru_layers >= 2)
            k_value = 1 if itr >= 15 else self.args.K_value
            
            
            local_cost = local_cost + delta_local_cost # the local ref corr volume 
            prob_corr = F.softmax(local_cost, dim=1)
            
            disparity_samples_list = []

            for dk in range(disp_topk.shape[1]):
                disp_t = disp_topk[:, dk, :, :].unsqueeze(dim=1)
                min_disparity = disp_t - rt
                max_disparity = disp_t + rt
                d_samples = self.uniform_sampler(min_disparity, max_disparity, 2 * rt + 1)  #(B,2r+1,H/4,W/4)
                disparity_samples_list.append(d_samples) #concat d_samples 
            disparity_samples = torch.cat(disparity_samples_list, dim=1)
            corr_disp = torch.sum(prob_corr * disparity_samples, dim=1, keepdim=True) # disp here is value 
            _, disp_topk_index = torch.topk(prob_corr, dim=1, k=k_value)
            disp_topk = torch.gather(disparity_samples, 1, disp_topk_index)
            

            # concat volume lookup
            geo_feat = geo_fn(concat_disp)
            
       
            net_list2 = self.update_block2_0(net_list2, inp_list2, iter16=True, iter08=False, iter04=False, update=False)
            if itr < 6:
                net_list2, mask_feat_4, delta_disp = self.update_block2_0(net_list2, inp_list2,local_cost ,geo_feat, concat_disp, 
                                                                            iter16=self.args.n_gru_layers==3,
                                                                            iter08=self.args.n_gru_layers>=2)   
            elif itr < 16:
                net_list2, mask_feat_4, delta_disp = self.update_block2_1(net_list2, inp_list2,local_cost , geo_feat, concat_disp, 
                                                                            iter16=self.args.n_gru_layers==3,
                                                                            iter08=self.args.n_gru_layers>=2)
            else:
                net_list2, mask_feat_4, delta_disp= self.update_block2_2(net_list2, inp_list2, local_cost , geo_feat, concat_disp, 
                                                                            iter16=self.args.n_gru_layers==3,
                                                                            iter08=self.args.n_gru_layers>=2)

            concat_disp = concat_disp + delta_disp
            


            if test_mode and itr < iters-1:
                continue
            
            # upsample disp predictions from GRU1
            if up_mask is None:
                disp_up_GRU1=updisp4(corr_disp)
            else:
                disp_up_GRU1 = self.upsample_disp_GRU1(corr_disp, up_mask)
            disp_pred_list_GRU1.append(disp_up_GRU1)

             # upsample disp predictions from GRU2
            disp_up_GRU2 = self.upsample_disp_GRU2(concat_disp, mask_feat_4, stem_2x)
            disp_pred_list_GRU2.append(disp_up_GRU2)
           

        if test_mode:
            return disp_up_GRU2
        init_disp = context_upsample(init_disp*4., spx_pred.float()).unsqueeze(1)
        
        return init_disp,disp_pred_list_GRU1, disp_pred_list_GRU2
    
