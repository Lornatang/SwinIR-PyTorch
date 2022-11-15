# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os

import cv2
import torch
from natsort import natsorted

import swinirnet_config
import imgproc
import model
from image_quality_assessment import PSNR, SSIM
from utils import make_directory


def main() -> None:
    # Initialize the super-resolution bsrgan_model
    sr_model = model.__dict__[swinirnet_config.g_arch_name](
        in_channels=swinirnet_config.g_in_channels,
        out_channels=swinirnet_config.g_out_channels,
        channels=swinirnet_config.g_channels)
    sr_model = sr_model.to(device=swinirnet_config.device)
    print(f"Build `{swinirnet_config.g_arch_name}` model successfully.")

    # Load the super-resolution bsrgan_model weights
    checkpoint = torch.load(swinirnet_config.g_model_weights_path, map_location=lambda storage, loc: storage)
    sr_model.load_state_dict(checkpoint["state_dict"])
    print(f"Load `{swinirnet_config.g_arch_name}` model weights "
          f"`{os.path.abspath(swinirnet_config.g_model_weights_path)}` successfully.")

    # Create a folder of super-resolution experiment results
    make_directory(swinirnet_config.sr_dir)

    # Start the verification mode of the bsrgan_model.
    sr_model.eval()

    # Initialize the sharpness evaluation function
    psnr = PSNR(swinirnet_config.upscale_factor, swinirnet_config.only_test_y_channel)
    ssim = SSIM(swinirnet_config.upscale_factor, swinirnet_config.only_test_y_channel)

    # Set the sharpness evaluation function calculation device to the specified model
    psnr = psnr.to(device=swinirnet_config.device, non_blocking=True)
    ssim = ssim.to(device=swinirnet_config.device, non_blocking=True)

    # Initialize IQA metrics
    psnr_metrics = 0.0
    ssim_metrics = 0.0

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(swinirnet_config.lr_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        lr_image_path = os.path.join(swinirnet_config.lr_dir, file_names[index])
        sr_image_path = os.path.join(swinirnet_config.sr_dir, file_names[index])
        gt_image_path = os.path.join(swinirnet_config.gt_dir, file_names[index])

        print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        lr_tensor = imgproc.preprocess_one_image(lr_image_path, swinirnet_config.device)
        gt_tensor = imgproc.preprocess_one_image(gt_image_path, swinirnet_config.device)

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_tensor = sr_model(lr_tensor)

        # Save image
        sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(sr_image_path, sr_image)

        # Cal IQA metrics
        psnr_metrics += psnr(sr_tensor, gt_tensor).item()
        ssim_metrics += ssim(sr_tensor, gt_tensor).item()

    # Calculate the average value of the sharpness evaluation index,
    # and all index range values are cut according to the following values
    # PSNR range value is 0~100
    # SSIM range value is 0~1
    avg_psnr = 100 if psnr_metrics / total_files > 100 else psnr_metrics / total_files
    avg_ssim = 1 if ssim_metrics / total_files > 1 else ssim_metrics / total_files

    print(f"PSNR: {avg_psnr:4.2f} [dB]\n"
          f"SSIM: {avg_ssim:4.4f} [u]")


if __name__ == "__main__":
    main()
