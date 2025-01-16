import torch
import torch.nn as nn
import torch.nn.functional as nnf
from .modelio import LoadableModel, store_config_args
from .MambaBlock import mambatran


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode
        # print('size', size)

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        self.register_buffer('grid', grid)

    def forward(self, src, flow):

        # new locations
        # print('flow', flow.shape)
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class ResizeTransform(nn.Module):
    """
    调整变换的大小，这涉及调整矢量场的大小并重新缩放它。
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class ConvResBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = x + out
        out = self.activation(out)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channel=1, first_channel=8):
        super(Encoder, self).__init__()
        c = first_channel
        self.block1 = ConvBlock(in_channel, c)
        self.block2 = ConvBlock(c, c * 2)
        self.block3 = ConvBlock(c *2, c * 4)
        self.block4 = ConvBlock(c *4, c * 4)

    def forward(self, x):
        out1 = self.block1(x)
        x_conv = nn.AvgPool3d(2)(out1)
        out2 = self.block2(x_conv)
        x_conv = nn.AvgPool3d(2)(out2)
        out3 = self.block3(x_conv)

        mamba_trans = mambatran(x)  
        out1_mamba = self.block1(mamba_trans)
        x = nn.AvgPool3d(2)(out1_mamba)
        out2_mamba = self.block2(x) 
        x = nn.AvgPool3d(2)(out2_mamba)
        out3_mamba = self.block3(x)
        x = nn.AvgPool3d(2)(out3_mamba)
        out4_mamba = self.block4(x)
        return out1, out2, out3, out4_mamba
        

class DecoderBlock(nn.Module):
    def __init__(self, x_channel, y_channel, out_channel, c_channel = None):
        super(DecoderBlock, self).__init__()
        if c_channel ==None:
            self.Conv1 = ConvBlock(x_channel+y_channel, out_channel)
        else:
            self.Conv1 = ConvBlock(x_channel+y_channel+c_channel, out_channel)
        self.Conv2 = ConvResBlock(out_channel, out_channel)
        self.Conv3 = ConvResBlock(out_channel, out_channel)
        self.Conv4 = nn.Conv3d(out_channel, out_channel//2, 3, padding=1)
        self.Conv5 = nn.Conv3d(out_channel//2, 3, 3, padding=1)


    def forward(self, x, y, c = None):
        if c == None:
             concat = torch.cat([x, y], dim=1)
        else:
             concat = torch.cat([x, y, c], dim=1)
        cost_vol = self.Conv1(concat)
      
        cost_vol = self.Conv2(cost_vol)
       
        cost_vol = self.Conv3(cost_vol)
        
        cost_vol = self.Conv4(cost_vol)
        flow = self.Conv5(cost_vol)

        return flow

class ConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class CConv(nn.Module):
    def __init__(self, channel):
        super(CConv, self).__init__()

        c = channel

        self.conv = nn.Sequential(
            ConvInsBlock(c, c, 3, 1),
            ConvInsBlock(c, c, 3, 1)
        )

    def forward(self, float_fm, fixed_fm, d_fm=None):
        if d_fm == None:
            concat_fm = torch.cat([float_fm, fixed_fm], dim=1)
        else:
            concat_fm = torch.cat([float_fm, fixed_fm, d_fm], dim=1)
        x = self.conv(concat_fm)
        return x

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, alpha=0.1):
        super(UpConvBlock, self).__init__()

        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)

        self.actout = nn.Sequential(
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(alpha)
        )
    def forward(self, x):
        x = self.upconv(x)
        return self.actout(x)

        
class EMambaBIRnet(LoadableModel):
    @store_config_args
    def __init__(self, size=(80, 96, 80), in_channel=1, first_channel=8):
        super(EMambaBIRnet, self).__init__()
        c = first_channel
        self.encoder = Encoder(in_channel, c)
        self.decoder4 = DecoderBlock(x_channel = 32, y_channel = 32, out_channel = 32)
        self.decoder3 = DecoderBlock(x_channel = 32, y_channel = 32, c_channel = 32, out_channel = 32)
        self.decoder2 = DecoderBlock(x_channel = 16, y_channel = 16, c_channel = 16, out_channel = 32)
        self.decoder1 = DecoderBlock(x_channel = 8, y_channel = 8, c_channel = 8 ,out_channel = 16)
        self.size = size

        self.cconv_1 = CConv(2*4*c)
        self.cconv_2 = CConv(3*4*c)
        self.cconv_3 = CConv(3*2*c)
        
        self.upconv1 = UpConvBlock(2*4*c, 4*c, 4, 2)
        self.upconv2 = UpConvBlock(8*c+12*c, 2*c, 4, 2)
        self.upconv3 = UpConvBlock(6*c+8*c+12*c, c, 4, 2)
       

        self.transformer = nn.ModuleList()
        for i in range(4):
            self.transformer.append(SpatialTransformer([s // 2**i for s in size]))
            
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x, y):
        fx1, fx2, fx3, fx4 = self.encoder(x)
        fy1, fy2, fy3, fy4 = self.encoder(y)

        ar = 1
        br = 1
        cr = 1
        dr = 1

        wx4 = fx4

        flowall =  None
        for aa in range(ar):
            flow = self.decoder4(wx4, fy4)
            c1 = self.cconv_1(wx4,fy4)
            if aa == 0:
                flowall = flow
            else:
                flowall = self.transformer[3](flowall, flow)+flow

        flowall = self.up(2*flowall)
        
        for bb in range(br):
            D1 = self.upconv1(c1)
            wx3 = self.transformer[2](fx3, flowall)
            
            flow = self.decoder3(wx3, fy3, D1)
            c2 = self.cconv_2(wx3,fy3,D1)
            
            flowall = self.transformer[2](flowall, flow) + flow

        flowall = self.up(2 * flowall)
        
        for cc in range(cr):
            c1 = self.up(c1)  
            D2 = torch.cat([c1,c2],dim = 1)
            D2 = self.upconv2(D2)
            wx2 = self.transformer[1](fx2, flowall)
            
            flow = self.decoder2(wx2, fy2, D2)
            c3 = self.cconv_3(wx2, fy2, D2)
            flowall = self.transformer[1](flowall, flow) + flow

        flowall = self.up(2 * flowall)
        
        for dd in range(dr):
            c1 = self.up(c1)
            c2 = self.up(c2)
            D1 = torch.cat([c1,c2,c3],dim = 1)
            D1 = self.upconv3(D1)

            wx1 = self.transformer[0](fx1, flowall)
            flow = self.decoder1(wx1, fy1, D1)
            flowall = self.transformer[0](flowall, flow) + flow
        warped_x = self.transformer[0](x, flowall)

        return warped_x, flowall

