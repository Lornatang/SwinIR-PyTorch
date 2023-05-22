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
import time
from typing import Any

import cv2
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

import model
from dataset import CUDAPrefetcher, PairedImageDataset
from imgproc import tensor_to_image
from utils import build_iqa_model, load_pretrained_state_dict, make_directory, AverageMeter, ProgressMeter, Summary

# Read parameters from configuration file
with open("configs/test/SWINIRNet_X4.yaml", "r") as f:
    config = yaml.full_load(f)


def load_dataset(device: torch.device) -> CUDAPrefetcher:
    test_datasets = PairedImageDataset(config["DATASET"]["PAIRED_TEST_GT_IMAGES_DIR"],
                                       config["DATASET"]["PAIRED_TEST_LR_IMAGES_DIR"])
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=config["HYP"]["IMGS_PER_BATCH"],
                                 shuffle=config["HYP"]["SHUFFLE"],
                                 num_workers=config["HYP"]["NUM_WORKERS"],
                                 pin_memory=config["HYP"]["PIN_MEMORY"],
                                 drop_last=False,
                                 persistent_workers=config["HYP"]["PERSISTENT_WORKERS"])
    test_data_prefetcher = CUDAPrefetcher(test_dataloader, device)

    return test_data_prefetcher


def build_model(device: torch.device) -> nn.Module | Any:
    g_model = model.__dict__[config["MODEL"]["NAME"]](in_channels=config["MODEL"]["IN_CHANNELS"],
                                                      out_channels=config["MODEL"]["OUT_CHANNELS"],
                                                      channels=config["MODEL"]["CHANNELS"])
    g_model = g_model.to(device)

    # compile model
    if config["MODEL"]["COMPILED"]:
        g_model = torch.compile(g_model)

    return g_model


def test(
        g_model: nn.Module,
        data_prefetcher: CUDAPrefetcher,
        psnr_model: nn.Module,
        ssim_model: nn.Module,
        device: torch.device,
        print_frequency: int,
        save_image: bool,
        save_dir_path: Any,
) -> [float, float]:
    if save_image and save_dir_path is None:
        raise ValueError("Image save location cannot be empty!")

    if save_image and not os.path.exists(save_dir_path):
        raise ValueError("The image save location does not exist!")

    # The information printed by the progress bar
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    psnres = AverageMeter("PSNR", ":4.2f", Summary.AVERAGE)
    ssimes = AverageMeter("SSIM", ":4.4f", Summary.AVERAGE)
    progress = ProgressMeter(len(data_prefetcher),
                             [batch_time, psnres, ssimes],
                             prefix=f"Test: ")

    # set the model as validation model
    g_model.eval()

    with torch.no_grad():
        # Initialize data batches
        batch_index = 0

        # Set the data set iterator pointer to 0 and load the first batch of data
        data_prefetcher.reset()
        batch_data = data_prefetcher.next()

        # Record the start time of verifying a batch
        end = time.time()

        while batch_data is not None:
            # Load batches of data
            gt = batch_data["gt"].to(device, non_blocking=True)
            lr = batch_data["lr"].to(device, non_blocking=True)

            # Reasoning
            sr = g_model(lr)

            # Calculate the image sharpness evaluation index
            psnr = psnr_model(sr, gt)
            ssim = ssim_model(sr, gt)

            # record current metrics
            psnres.update(psnr.item(), sr.size(0))
            ssimes.update(ssim.item(), ssim.size(0))

            # Record the total time to verify a batch
            batch_time.update(time.time() - end)
            end = time.time()

            # Output a verification log information
            if batch_index % print_frequency == 0:
                progress.display(batch_index)

            # Save the processed image after super-resolution
            if save_image and batch_data["image_name"] is None:
                raise ValueError("The image_name is None, please check the dataset.")
            if save_image:
                image_name = os.path.basename(batch_data["image_name"][0])
                sr_image = tensor_to_image(sr, False, False)
                sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_dir_path, image_name), sr_image)

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # Add 1 to the number of data batches
            batch_index += 1

    # Print the performance index of the model at the current Epoch
    progress.display_summary()

    return psnres.avg, ssimes.avg


def main() -> None:
    device = torch.device("cuda", config["DEVICE_ID"])
    test_data_prefetcher = load_dataset(device)
    g_model = build_model(device)
    psnr_model, ssim_model = build_iqa_model(
        config["SCALE"],
        config["ONLY_TEST_Y_CHANNEL"],
        device,
    )

    # Load model weights
    g_model = load_pretrained_state_dict(g_model, config["MODEL"]["COMPILED"], config["MODEL_PATH"])

    # Create a directory for saving test results
    save_dir_path = os.path.join(config["SAVE_DIR_PATH"], config["EXP_NAME"])
    if config["SAVE_IMAGE"]:
        make_directory(save_dir_path)

    psnr, ssim = test(g_model,
                      test_data_prefetcher,
                      psnr_model,
                      ssim_model,
                      device,
                      config["PRINT_FREQ"],
                      config["SAVE_IMAGE"],
                      save_dir_path)


if __name__ == "__main__":
    main()
