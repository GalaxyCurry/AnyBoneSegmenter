from models.sam import SamPredictor, sam_model_registry
from models.sam.modeling.prompt_encoder import attention_fusion
import numpy as np
import os
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from dsc import dice_coeff
import torchio as tio
import nrrd
import PIL
import cfg
from funcs import *
from predict_funs import *

args = cfg.parse_args()
from monai.networks.nets import VNet

args.if_mask_decoder_adapter = True
args.if_encoder_adapter = True
args.decoder_adapt_depth = 2


def predictVolume(image_name, lower_percentile, upper_percentile):
    dsc_gt = 0
    image1_vol = tio.ScalarImage(os.path.join(img_folder, image_name))
    print('vol shape: %s vol spacing %s' % (image1_vol.shape, image1_vol.spacing))

    # Define the percentiles
    image_tensor = image1_vol.data
    lower_bound = torch_percentile(image_tensor, lower_percentile)
    upper_bound = torch_percentile(image_tensor, upper_percentile)

    # Clip the data
    image_tensor = torch.clamp(image_tensor, lower_bound, upper_bound)
    # Normalize the data to [0, 1] 
    image_tensor = (image_tensor - lower_bound) / (upper_bound - lower_bound)
    image1_vol.set_data(image_tensor)

    mask_vol_numpy = np.zeros(image1_vol.shape)
    id_list = list(range(image1_vol.shape[3]))
    for id in id_list:
        atten_map = pred_attention(image1_vol, vnet, id, device)
        atten_map = torch.unsqueeze(torch.tensor(atten_map), 0).float().to(device)

        ori_img, pred_1, voxel_spacing1, Pil_img1, slice_id1 = evaluate_1_volume_withattention(image1_vol,
                                                                                               sam_fine_tune, device,
                                                                                               slice_id=id,
                                                                                               atten_map=atten_map)
        img1_size = Pil_img1.size
        mask_pred = ((pred_1 > 0) == cls).float().cpu()
        pil_mask1 = Image.fromarray(np.array(mask_pred[0], dtype=np.uint8), 'L').resize(img1_size,
                                                                                        resample=PIL.Image.NEAREST)
        mask_vol_numpy[0, :, :, id] = np.asarray(pil_mask1)

    mask_vol = tio.LabelMap(tensor=torch.tensor(mask_vol_numpy, dtype=torch.int), affine=image1_vol.affine)
    mask_save_folder = os.path.join(predicted_msk_folder, '/'.join(image_name.split('/')[:-1]))
    Path(mask_save_folder).mkdir(parents=True, exist_ok=True)
    mask_vol.save(
        os.path.join(mask_save_folder, image_name.split('/')[-1].replace('.nii.gz', '_predicted_SAMatten_paired.nrrd')))
    return mask_vol


def predictAndEvaluateVolume(image_name, mask_name, lower_percentile, upper_percentile):
    dsc_gt = 0
    image1_vol = tio.ScalarImage(os.path.join(img_folder, image_name))
    print('vol shape: %s vol spacing %s' % (image1_vol.shape, image1_vol.spacing))

    # Define the percentiles
    image_tensor = image1_vol.data
    lower_bound = torch_percentile(image_tensor, lower_percentile)
    upper_bound = torch_percentile(image_tensor, upper_percentile)

    # Clip the data
    image_tensor = torch.clamp(image_tensor, lower_bound, upper_bound)
    # Normalize the data to [0, 1]
    image_tensor = (image_tensor - lower_bound) / (upper_bound - lower_bound)
    image1_vol.set_data(image_tensor)

    voxels, header = nrrd.read(os.path.join(gt_msk_folder, mask_name))
    mask_gt = voxels
    mask_vol_numpy = np.zeros(image1_vol.shape)
    id_list = list(range(image1_vol.shape[3]))
    for id in id_list:
        atten_map = pred_attention(image1_vol, vnet, id, device)
        atten_map = torch.unsqueeze(torch.tensor(atten_map), 0).float().to(device)

        ori_img, pred_1, voxel_spacing1, Pil_img1, slice_id1 = evaluate_1_volume_withattention(image1_vol,
                                                                                               sam_fine_tune, device,
                                                                                               slice_id=id,
                                                                                               atten_map=atten_map)
        img1_size = Pil_img1.size

        mask_pred = ((pred_1 > 0) == cls).float().cpu()
        msk = Image.fromarray(mask_gt[:, :, id].astype(np.uint8), 'L')
        msk = transforms.Resize((256, 256))(msk)
        msk_gt = (transforms.ToTensor()(msk) > 0).float().cpu()
        dsc_gt += dice_coeff(mask_pred.cpu(), msk_gt).item()
        pil_mask1 = Image.fromarray(np.array(mask_pred[0], dtype=np.uint8), 'L').resize(img1_size,
                                                                                        resample=PIL.Image.NEAREST)
        mask_vol_numpy[0, :, :, id] = np.asarray(pil_mask1)

    mask_vol = tio.LabelMap(tensor=torch.tensor(mask_vol_numpy, dtype=torch.int), affine=image1_vol.affine)
    mask_save_folder = os.path.join(predicted_msk_folder, '/'.join(image_name.split('/')[:-1]))
    Path(mask_save_folder).mkdir(parents=True, exist_ok=True)
    mask_vol.save(
        os.path.join(mask_save_folder, image_name.split('/')[-1].replace('.nii.gz', '_predicted_SAMatten_paired.nrrd')))
    dsc_gt /= len(id_list)
    gt_vol = tio.LabelMap(tensor=torch.unsqueeze(torch.Tensor(mask_gt > 0), 0), affine=image1_vol.affine)
    dsc_vol = dice_coeff(mask_vol.data.float().cpu(), gt_vol.data).item()
    print('volume %s: slice_wise_dsc %.2f; vol_wise_dsc %.2f' % (image_name, dsc_gt, dsc_vol))


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
checkpoint_directory = './checkpoint'  # path to your checkpoint
img_folder = os.path.join('images')
gt_msk_folder = os.path.join('masks')
predicted_msk_folder = os.path.join('predicted_masks')
cls = 1

sam_fine_tune = sam_model_registry["vit_t"](args, checkpoint=os.path.join(checkpoint_directory, 'mobile_sam.pt'),
                                            num_classes=2)
sam_fine_tune.attention_fusion = attention_fusion()
sam_fine_tune.load_state_dict(
    torch.load(os.path.join(checkpoint_directory, 'bone_sam.pth'), map_location=torch.device(device)), strict=True)
sam_fine_tune = sam_fine_tune.to(device).eval()

vnet = VNet().to(device)
model_directory = './checkpoint'
vnet.load_state_dict(torch.load(os.path.join(model_directory, 'atten.pth'), map_location=torch.device(device)))

mask = predictVolume(
    image_name='liver_0.nii.gz',
    lower_percentile=1,
    upper_percentile=99
)

predictAndEvaluateVolume(
    image_name='liver_0.nii.gz',
    mask_name='liver_0_seg.nrrd',
    lower_percentile=1,
    upper_percentile=99
)
