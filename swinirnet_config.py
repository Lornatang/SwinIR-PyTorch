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
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = True
# Model architecture name
g_arch_name = "swinir_default_sr_x4"
# Model arch config
g_in_channels = 3
g_out_channels = 3
g_channels = 64
upscale_factor = 4
# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = "SwinIRNet_default_sr_x4-DIV2K-DIV2K"

if mode == "train":
    # Dataset address
    train_gt_images_dir = f"./data/DIV2K/SwinIR/train"

    test_gt_images_dir = f"./data/Set5/GTmod12"
    test_lr_images_dir = f"./data/Set5/LRbicx{upscale_factor}"

    gt_image_size = int(48 * upscale_factor)
    batch_size = 16
    num_workers = 4

    # The address to load the pretrained model
    pretrained_g_model_weights_path = ""

    # Incremental training and migration training
    resume_g_model_weights_path = f""

    # Total num epochs (1,000,000 iters)
    epochs = 1000

    # loss function weights
    loss_weight = [1.0]

    # Optimizer parameter
    model_lr = 2e-4
    model_betas = (0.9, 0.999)
    model_eps = 1e-8
    model_weight_decay = 0.0

    # EMA parameter
    model_ema_decay = 0.999

    # Dynamically adjust the learning rate policy
    lr_scheduler_milestones = [int(epochs * 0.5), int(epochs * 0.8), int(epochs * 0.9), int(epochs * 0.95), epochs]
    lr_scheduler_gamma = 0.5

    # How many iterations to print the training result
    train_print_frequency = 100
    test_print_frequency = 1

if mode == "test":
    # Test data address
    lr_dir = f"./data/Set5/LRbicx{upscale_factor}"
    sr_dir = f"./results/test/{exp_name}"
    gt_dir = "./data/Set5/GTmod12"

    g_model_weights_path = "./"
