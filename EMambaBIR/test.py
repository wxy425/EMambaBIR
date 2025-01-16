import os
import argparse
import numpy as np
import nibabel as nib
import torch
import time


os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm

# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--scansdir',help='pytorch model for nonlinear registration')
parser.add_argument('--labelsdir',help='test scan npz directory')
parser.add_argument('--model', help='pytorch model for nonlinear registration')
parser.add_argument('--labels' ,help='label lookup file in npz format')
parser.add_argument('--dataset', help='dataset')

parser.add_argument('--gpu',help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--multichannel',action='store_true', help='specify that data has multiple channels')
args = parser.parse_args()


# device handling
if args.gpu and (args.gpu != '-1'):
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

else:
    device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

scansdir = args.scansdir      #vol图像目录
labelsdir = args.labelsdir    #seg图像目录
test_data = os.listdir(scansdir)
test_data = sorted(test_data, key=str.lower)  #scansdir目录下所以文件和目录，全部转换为小写后进行排序

# load moving and fixed images
add_feat_axis = not args.multichannel

if args.dataset == 'mind':
    img_size = (160, 192, 160)
else:
    img_size = (128, 128, 96)

model = vxm.EMambaBIR.EMambaBIRnet(img_size)
best_model = torch.load(args.model)
model.load_state_dict(best_model,False)


model.to(device)
total_params = sum([param.numel() for param in model.parameters()])
model.eval()

# Use this to warp segments
trf = vxm.layers.SpatialTransformer(img_size, mode='nearest')
trf.to(device)
da = jnum = 0
dice_total = []
HD95_total = []
MSE_total = []
infer_time = []
repeat_times = 0

for i in range(len(test_data)-1):
    atlas_dir = scansdir + '/' + test_data[i]    #测试vol图像
    labels_dir = labelsdir + '/' + test_data[i]  #测试vol图像对应的seg图像
    atlas_vol = vxm.py.utils.load_volfile(atlas_dir, np_var='vol', add_batch_axis=True, add_feat_axis=add_feat_axis)
    atlas_vol1 = vxm.py.utils.load_volfile(atlas_dir, np_var='vol')
    atlas_seg = vxm.py.utils.load_volfile(labels_dir, np_var='seg')
    if i == 0:
        if args.dataset == 'mind':
            labels = np.load(args.labels)['labels']
            print('mind_label')
        else:
            labels = np.unique(atlas_seg)
            labels = labels[1:]
            print('flare_label')
        print(len(labels))
    for j in range(i+1, len(test_data)):
        repeat_times += 1
        moving_dir = scansdir + '/' + test_data[j]
        labels_dir = labelsdir + '/' + test_data[j]
        moving_vol = vxm.py.utils.load_volfile(moving_dir, np_var='vol', add_batch_axis=True, add_feat_axis=add_feat_axis)
        moving_seg = vxm.py.utils.load_volfile(labels_dir, np_var='seg', add_batch_axis=True, add_feat_axis=add_feat_axis)
        moving_seg1 = vxm.py.utils.load_volfile(labels_dir, np_var='seg')
        moving_vol1 = vxm.py.utils.load_volfile(moving_dir, np_var='vol')
 
        input_moving = torch.from_numpy(moving_vol).to(device).float().permute(0, 4, 1, 2, 3)
        input_fixed = torch.from_numpy(atlas_vol).to(device).float().permute(0, 4, 1, 2, 3)
        # predict and apply transform
        with torch.no_grad():
            start_time = time.time()
            #warped_mov, warp1,ct = model(input_moving, input_fixed)
            warped_mov, warp1 = model(input_moving, input_fixed)
            end_time = time.time()
            duration = end_time - start_time
       #print(ct)
        input_seg = torch.from_numpy(moving_seg).to(device).float().permute(0, 4, 1, 2, 3)
        warped_seg = trf(input_seg, warp1)
        warped_seg = warped_seg.detach().cpu().numpy().squeeze()
        overlap = vxm.py.utils.dice(warped_seg, atlas_seg, labels=labels)
        print(np.mean(overlap))
        MSE = vxm.py.utils.MSE(warped_seg,atlas_seg)
        HD95 = vxm.py.utils.hausdorff_distance(warped_seg,atlas_seg,percentage=95)
        
     
        dice_total.append(np.mean(overlap))

        df = warp1.detach().cpu().numpy().squeeze()
        df1 = df.transpose(1,2,3,0)
        jb = vxm.py.utils.jacobian_determinant(df1)
        jnum += np.sum(jb <= 0)
        
        MSE_total.append(np.mean(MSE))

        HD95_total.append(np.mean(HD95))

        if j > 0:
                infer_time.append(duration)

dice_total = np.array(dice_total)
jnum = jnum / repeat_times
HD95_total = np.array(HD95_total)
MSE_total = np.array(MSE_total)
infer_time = np.array(infer_time)
print('Avg Dice:       %6.4f +/- %6.4f' % (np.mean(dice_total), np.std(dice_total)))
print('|js|<=0:        %6.4f'            % (jnum/(160*192*160)))
print('HD95:           %6.4f'           % (HD95_total.mean()))
print('MSE(10-3):      %6.4f'           % (MSE_total.mean()/1000))
print('infer_time:     %6.4f'          % (infer_time.mean()))
print(f'Total params:   {total_params / 1e6} Mb\n')