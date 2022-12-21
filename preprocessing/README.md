# Light Estimation for Captured Data

We use the pretrained [SDPS-Net](https://github.com/guanyingc/SDPS-Net) to estimate the light direction of the handheld cellphone flashlights from single-view multi-light input images. The output of predicted light directions will be defaultly saved under each dataset folder (named `light_direction_sdps.npy`). The light positions are then obtained by combining the estimated light direction and the roughly measured light-object distance (we assume point light). After initialization, the position of lights are jointly optimized with the shape and BRDF during training.


## Get the Initial Light Direction Estimation
```bash
## replace `GPU_ID` and `OBJ_NAME` with your choices.
CUDA_VISIBLE_DEVICES=GPU_ID \
python test.py \
    --retrain data/models/LCNet_CVPR2019.pth.tar \
    --retrain_s2 data/models/NENet_CVPR2019.pth.tar \
    --benchmark UPS_Custom_Dataset \
    --bm_dir ../dataset/OBJ_NAME
```

## Citation
This submodule is adapted from **[SDPS-Net: Self-calibrating Deep Photometric Stereo Networks, CVPR 2019 (Oral)](http://guanyingc.github.io/SDPS-Net/)**, which addresses the problem of learning-based _uncalibrated_ photometric stereo for non-Lambertian surface. If you find this code or the provided models useful in your research, please consider cite: 
```
@inproceedings{chen2019SDPS_Net,
  title={SDPS-Net: Self-calibrating Deep Photometric Stereo Networks},
  author={Chen, Guanying and Han, Kai and Shi, Boxin and Matsushita, Yasuyuki and Wong, Kwan-Yee K.},
  booktitle={CVPR},
  year={2019}
}
```
