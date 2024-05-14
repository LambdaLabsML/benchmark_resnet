from __future__ import print_function

import argparse
import os
import subprocess

from torchvision import datasets, transforms, models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

import ray.train.torch
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer


# Training settings
parser = argparse.ArgumentParser(description="PyTorch CIFAR10 ResNet152 Training")
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size for training (default: 128)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=1000,
    metavar="N",
    help="input batch size for testing (default: 1000)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.1,
    metavar="LR",
    help="learning rate (default: 0.1)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="SGD momentum (default: 0.9)",
)
parser.add_argument(
    "--no-cuda",
    action="store_true",
    default=False,
    help="disables CUDA training",
)
parser.add_argument(
    "--seed",
    type=int,
    default=1,
    metavar="S",
    help="random seed (default: 1)",
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--repeat",
    type=int,
    default=10,
    metavar="R",
    help="how many times we repeat each batch (for inflating the dataset size)",
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=10,
    help="number of workers for dataloader",
)
parser.add_argument(
    "--save-model",
    action="store_true",
    default=False,
    help="For Saving the current Model",
)
parser.add_argument(
    "--pin-memory",
    action="store_true",
    default=False,
    help="For automatically put the fetched data Tensors in pinned memory",
)
parser.add_argument(
    "--persistent-workers",
    action="store_true",
    default=False,
    help="The data loader will not shutdown the worker processes after a dataset has been consumed once",
)
parser.add_argument(
    "--prefetch-factor",
    type=int,
    default=2,
    help="Number of batches loaded in advance by each worker.",
)
parser.add_argument(
    "--dir",
    default="logs",
    metavar="L",
    help="directory where summary logs are stored",
)

parser.add_argument(
    "--use-syn",
    action="store_true",
    default=False,
    help="Use synthetic data for training",
)
parser.add_argument(
    "--use-transform",
    action="store_true",
    default=False,
    help="Use data transformation for training",
)
parser.add_argument(
    "--num-syn-batches",
    type=int,
    default=25,
    help="number steps for benchmark using synthetic data",
)
parser.add_argument(
    "--storage-path",
    type=str,
    help="Path to dataset",
    default="/mnt/cluster_storage",
)
parser.add_argument(
    "--dataset-path",
    type=str,
    help="Path to dataset",
    default="/mnt/cluster_storage/tiny-224",
)
parser.add_argument(
    "--dataset-url",
    type=str,
    help="URL to download imagenet style dataset",
    default="https://lambdaml.s3.us-west-1.amazonaws.com/tiny-224.zip",
)
parser.add_argument(
    "--num-gpu-workers",
    type=int,
    default=1,
    help="number of workers for dataloader",
)
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")


def train_func():
    # ----------------------------------
    # Dataset
    # ----------------------------------
    # Downlaod tiny-imagenet if needed
    if ray.train.get_context().get_world_rank() == 0:
        if os.path.isdir(args.dataset_path):
            print("Dataset exist. Skip download")
        else:
            print("Download dataset.")
            command_download = [
                "wget",
                "-O",
                args.dataset_path + ".zip",
                args.dataset_url,
            ]
            subprocess.run(command_download)
            command_unzip = [
                "unzip",
                args.dataset_path + ".zip",
                "-d",
                os.path.dirname(args.dataset_path),
            ]
            subprocess.run(command_unzip)

    if args.use_transform:
        data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
                    ),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
                    ),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
                    ),
                ]
            ),
        }
    else:
        data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
        }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(args.dataset_path, x), data_transforms[x])
        for x in ["train", "val", "test"]
    }

    print(f"Number of workers: {args.num_workers}")
    print(f"Pin memory: {args.pin_memory}")
    print(f"Persistent memory: {args.pin_memory}")
    print(f"Prefetch workers: {args.persistent_workers}")
    print(f"Use transformes: {args.use_transform}")

    train_loader = torch.utils.data.DataLoader(
        image_datasets["train"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )

    # train_loader = ray.train.torch.prepare_data_loader(train_loader)

    train_loader = ray.train.torch.prepare_data_loader(
        train_loader, add_dist_sampler=False
    )

    # ----------------------------------
    # Model
    # ----------------------------------
    # Load the resnet152 model
    model = models.resnet152(pretrained=False).to(device)
    model = ray.train.torch.prepare_model(model)

    model.train()

    # ----------------------------------
    # Others
    # ----------------------------------
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        # if ray.train.get_context().get_world_size() > 1:
        #     train_loader.sampler.set_epoch(epoch)
        total_samples = 0
        start_time = time.time()  # Start timing

        if args.use_syn:
            print("Use synthetic data ...")
            data = torch.randn(args.batch_size, 3, 224, 224).to(device)
            target = torch.randint(0, 1000, (args.batch_size,)).to(device)
            for batch_idx in range(args.num_syn_batches):
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

                total_samples += len(data)  # Update total samples processed
                if ray.train.get_context().get_world_rank() == 0:
                    if batch_idx % args.log_interval == 0:
                        print(
                            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss={:.6f}".format(
                                epoch,
                                batch_idx * len(data),
                                len(train_loader.dataset),
                                100.0 * batch_idx / len(train_loader),
                                loss.item(),
                            )
                        )
                        niter = epoch * len(train_loader) + batch_idx
        else:
            print("Use real data ...")
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device, non_blocking=args.pin_memory), target.to(
                    device, non_blocking=args.pin_memory
                )
                # Attach tensors to the device.
                for r in range(args.repeat):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()

                    total_samples += len(data)  # Update total samples processed

                    if ray.train.get_context().get_world_rank() == 0:
                        if batch_idx % args.log_interval == 0:
                            print(
                                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss={:.6f}".format(
                                    epoch,
                                    batch_idx * len(data),
                                    len(train_loader.dataset),
                                    100.0 * batch_idx / len(train_loader),
                                    loss.item(),
                                )
                            )
                            niter = epoch * len(train_loader) + batch_idx

        elapsed_time = time.time() - start_time
        throughput = total_samples / elapsed_time

        if ray.train.get_context().get_world_rank() == 0:
            print(f"Epoch {epoch}: Throughput is {throughput:.2f} samples/sec")


scaling_config = ScalingConfig(num_workers=args.num_gpu_workers, use_gpu=True)
run_config = ray.train.RunConfig(storage_path=args.storage_path)

ray_trainer = TorchTrainer(
    train_func, scaling_config=scaling_config, run_config=run_config
)
result = ray_trainer.fit()
