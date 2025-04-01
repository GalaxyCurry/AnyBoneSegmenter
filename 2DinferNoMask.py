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
args.if_mask_decoder_adapter=True
args.if_encoder_adapter = True
args.decoder_adapt_depth = 2


def predictSlice(image_name, lower_percentile, upper_percentile, slice_id, attention_enabled):
    
    image1_vol = tio.ScalarImage(os.path.join(img_folder, image_name))
    print('vol shape: %s vol spacing %s' %(image1_vol.shape,image1_vol.spacing))

    image_tensor = image1_vol.data
    lower_bound = torch_percentile(image_tensor, lower_percentile)
    upper_bound = torch_percentile(image_tensor, upper_percentile)

    # Clip the data
    image_tensor = torch.clamp(image_tensor, lower_bound, upper_bound)

    # Normalize the data to [0, 1] 
    image_tensor = (image_tensor - lower_bound) / (upper_bound - lower_bound)

    image1_vol.set_data(image_tensor)
    atten_map= pred_attention(image1_vol,vnet,slice_id,device)
    
    atten_map = torch.unsqueeze(torch.tensor(atten_map),0).float().to(device)
    print(atten_map.device)
    if attention_enabled:
        ori_img,pred_1,voxel_spacing1,Pil_img1,slice_id1 = evaluate_1_volume_withattention(image1_vol,sam_fine_tune,device,slice_id=slice_id,atten_map=atten_map)
    else:
        ori_img,pred_1,voxel_spacing1,Pil_img1,slice_id1 = evaluate_1_volume_withattention(image1_vol,sam_fine_tune,device,slice_id=slice_id)
        
    mask_pred = ((pred_1>0)==cls).float().cpu()

    return ori_img, mask_pred, atten_map


def visualizeSlicePrediction(ori_img, image_name, atten_map, mask_pred):
    image = np.rot90(torchvision.transforms.Resize((args.out_size,args.out_size))(ori_img)[0])
    image_3d = np.repeat(np.array(image*255,dtype=np.uint8).copy()[:, :, np.newaxis], 3, axis=2)

    pred_mask_auto = (mask_pred[0])*255

    target_prediction =  [103,169,237]   
    image_pred_auto = drawContour(image_3d.copy(), np.rot90(pred_mask_auto),target_prediction,size=-1,a=0.6)

    fig, a = plt.subplots(1,4, figsize=(20,15))

    a[0].imshow(image,cmap='gray',vmin=0, vmax=1)
    a[0].set_title(image_name)
    a[0].axis(False)

    a[1].imshow(image_pred_auto,cmap='gray',vmin=0, vmax=255)
    a[1].set_title('pre_mask',fontsize=10)
    a[1].axis(False)

    a[2].imshow(np.rot90(atten_map.cpu()[0]),vmin=0, vmax=1,cmap='coolwarm')
    a[2].set_title('atten_map',fontsize=10)
    a[2].axis(False)

    plt.tight_layout()
    plt.show()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
checkpoint_directory = './checkpoint' # path to your checkpoint
img_folder = os.path.join('images')
gt_msk_folder = os.path.join('masks')
predicted_msk_folder = os.path.join('predicted_masks')
cls = 1

sam_fine_tune = sam_model_registry["vit_t"](args,checkpoint=os.path.join(checkpoint_directory,'mobile_sam.pt'),num_classes=2)
sam_fine_tune.attention_fusion = attention_fusion()  
sam_fine_tune.load_state_dict(torch.load(os.path.join(checkpoint_directory,'bone_sam.pth'),map_location=torch.device(device)), strict = True)
sam_fine_tune = sam_fine_tune.to(device).eval()

vnet = VNet().to(device)
model_directory = './checkpoint'
vnet.load_state_dict(torch.load(os.path.join(model_directory,'atten.pth'),map_location=torch.device(device)))



ori_img, predictedSliceMask, atten_map = predictSlice(
    image_name = 'demo.nii.gz',
    lower_percentile = 1,
    upper_percentile = 99,
    slice_id = 3, # slice number
    attention_enabled = True, # if you want to use the depth attention
)


visualizeSlicePrediction(
    ori_img=ori_img, 
    image_name='demo.nii.gz',
    atten_map=atten_map,
    mask_pred=predictedSliceMask,
)

