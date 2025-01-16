import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from MambaBlock_test import MambaBlock

class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.LeakyReLU(inplace=True)
        if not use_batchnorm:
            nm = nn.InstanceNorm3d(out_channels)
        else:
            nm = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, nm, relu)

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

# 读取 .nii.gz 文件
def load_nii_gz(file_path):
    nii = nib.load(file_path)
    data = nii.get_fdata()  # 获取图像数据
    affine = nii.affine  # 获取仿射矩阵（用于空间变换）
    header = nii.header  # 获取头文件信息
    return data, affine, header


# 将 NumPy 数组转换为 PyTorch 张量
def numpy_to_torch_tensor(data, dtype=torch.float32):
    tensor = torch.tensor(data, dtype=dtype)
    # 如果数据是多通道的（例如RGB图像），可能需要调整张量的形状
    # tensor = tensor.permute(3, 0, 1, 2)  # 例如从 (H, W, D, C) 转换为 (C, H, W, D)
    return tensor


# 显示图像（使用 matplotlib）
def plot_slice(data, slice_index, ax=None, cmap='gray'):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    rotated_data = np.rot90(data[:, :, slice_index], k=1)
    ax.imshow(rotated_data, cmap=cmap)
    ax.axis('off')
    plt.savefig('f.png')
    plt.show()



# 示例代码
file_path = '/root/autodl-fs/Mindboggle/NKI-RS-22-1.nii.gz'  # 替换为你的 .nii.gz 文件路径

# 读取数据
data, affine, header = load_nii_gz(file_path)

#显示原始图像
data_tensor = numpy_to_torch_tensor(data)
print(data_tensor.shape)
slice_index = data.shape[2] // 2  # 选择中间的切片
plot_slice(data_tensor.numpy().squeeze(), slice_index)  # 需要将张量转换回 NumPy 数组以进行显示

#显示mamba模块处理后的图像

#data_tensor = data_tensor.unsqueeze(0)
#data_tensor_pre_mamba = torch.cat((data_tensor,data_tensor),dim = 0).unsqueeze(0)
data_tensor = data_tensor.unsqueeze(0).unsqueeze(0)
print("234234",data_tensor.shape)
transformer = MambaBlock(          patch_size=4,
                                      in_chans=1,
                                      embed_dim=64,
                                      depths=[2,2,6,2],
                                      drop_rate=0,
                                      ape=False,
                                      spe=False,
                                      rpe=True,
                                      patch_norm=True,
                                      out_indices=(0,1,2,3),
                                      d_state=16,
                                      d_conv=4,
                                      expand=2,
                                      )

conv = Conv3dReLU(
            64,
            1,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x = data_tensor.to(device)
transformer = transformer.to(device)
up1 = DecoderBlock(1, 1, skip_channels=1, use_batchnorm=False)  # 384, 20, 20, 64


up1 = up1.to(device)
avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
conv = conv.to(device)
avg_pool = avg_pool.to(device)




x_avg = avg_pool(x)
out_feats = transformer(x)
x_conv = conv(out_feats[0])
     

x1 = up1(x_conv,x_avg)
x = up1(x1,x)

min_val = x.min()
max_val = x.max()
x = (x - min_val) / (max_val - min_val)


after_mamba = x.squeeze(0).squeeze(0)

mamba_pre_show = after_mamba.cpu()
plot_slice(mamba_pre_show.detach().numpy().squeeze(), slice_index)



