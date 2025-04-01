import SimpleITK as sitk


def nii2nii(oripath,savepath):
    data = sitk.ReadImage(oripath)
    img = sitk.GetArrayFromImage(data)
    out = sitk.GetImageFromArray(img)
    sitk.WriteImage(out,savepath)


nii2nii('./masks/liver_0_seg.nii.gz', 'liver_0_seg.nrrd')
