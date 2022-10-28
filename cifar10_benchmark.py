
import os
import time
import pytorch_lightning as pl
import torch
from ssl_models import *
import torchvision
from pytorch_lightning.loggers import WandbLogger
from lightly.data import LightlyDataset
import wandb

logs_root_dir = os.path.join(os.getcwd(), 'benchmark_logs')

# set max_epochs to 800 for long run (takes around 10h on a single V100)
max_epochs = 800
num_workers = 12
knn_k = 200
knn_t = 0.1
classes = 10
dataset = "cifar10"

# Set to True to enable Distributed Data Parallel training.
distributed = False

# Set to True to enable Synchronized Batch Norm (requires distributed=True). 
# If enabled the batch norm is calculated over all gpus, otherwise the batch
# norm is only calculated from samples on the same gpu.
sync_batchnorm = False

# Set to True to gather features from all gpus before calculating 
# the loss (requires distributed=True).
# If enabled then the loss on every gpu is calculated with features from all 
# gpus, otherwise only features from the same gpu are used.
gather_distributed = False 

# benchmark

batch_size = 256
lr_factor = batch_size / 128 # scales the learning rate linearly with batch size

# use a GPU if available
gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

if distributed:
    distributed_backend = 'ddp'
    # reduce batch size for distributed training
    batch_size = batch_size // gpus
else:
    distributed_backend = None
    # limit to single gpu if not using distributed training
    gpus = min(gpus, 1)

# Adapted from our MoCo Tutorial on CIFAR-10
#
# Replace the path with the location of your CIFAR-10 dataset.
# We assume we have a train folder with subfolders
# for each class and .png images inside.
#
# You can download `CIFAR-10 in folders from kaggle 
# <https://www.kaggle.com/swaroopkml/cifar10-pngs-in-folders>`_.

# The dataset structure should be like this:
# cifar10/train/
#  L airplane/
#    L 10008_airplane.png
#    L ...
#  L automobile/
#  L bird/
#  L cat/
#  L deer/
#  L dog/
#  L frog/
#  L horse/
#  L ship/
#  L truck/

path_to_train = f'/datasets/{dataset}/train/'
path_to_test = f'/datasets/{dataset}/test/'

# Use SimCLR augmentations, additionally, disable blur for cifar10
collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=32,
    gaussian_blur=0.,
)

# Multi crop augmentation for SwAV, additionally, disable blur for cifar10
swav_collate_fn = lightly.data.SwaVCollateFunction(
    crop_sizes=[32],
    crop_counts=[2], # 2 crops @ 32x32px
    crop_min_scales=[0.14],
    gaussian_blur=0,
)

# Multi crop augmentation for DINO, additionally, disable blur for cifar10
dino_collate_fn = lightly.data.DINOCollateFunction(
    global_crop_size=32,
    n_local_views=0,
    gaussian_blur=(0, 0, 0),
)

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

# cifar100 = torchvision.datasets.CIFAR100("/home/ido/datasets/cifar100", download=True)
# dataset_train_ssl = LightlyDataset.from_torch_dataset(cifar100)
#
# cifar100 = torchvision.datasets.CIFAR100("/home/ido/datasets/cifar100", download=True)
# dataset_train_kNN = LightlyDataset.from_torch_dataset(cifar100, transform=test_transforms)
#
# cifar100_test = torchvision.datasets.CIFAR100("/home/ido/datasets/cifar100", train=False, download=True)
# dataset_test = LightlyDataset.from_torch_dataset(cifar100_test, transform=test_transforms)

cifar10 = torchvision.datasets.CIFAR10("/home/ido/datasets/cifar10", download=True)
dataset_train_ssl = LightlyDataset.from_torch_dataset(cifar10)

cifar10 = torchvision.datasets.CIFAR10("/home/ido/datasets/cifar10", download=True)
dataset_train_kNN = LightlyDataset.from_torch_dataset(cifar10, transform=test_transforms)

cifar10_test = torchvision.datasets.CIFAR10("/home/ido/datasets/cifar10", train=False, download=True)
dataset_test = LightlyDataset.from_torch_dataset(cifar10_test, transform=test_transforms)

#
# dataset_train_ssl = lightly.data.LightlyDataset(
#     input_dir=path_to_train
# )
#
# # we use test transformations for getting the feature for kNN on train data
# dataset_train_kNN = lightly.data.LightlyDataset(
#     input_dir=path_to_train,
#     transform=test_transforms
# )
#
# dataset_test = lightly.data.LightlyDataset(
#     input_dir=path_to_test,
#     transform=test_transforms
# )

def get_data_loaders(batch_size: int, model):
    """Helper method to create dataloaders for ssl, kNN train and kNN test

    Args:
        batch_size: Desired batch size for all dataloaders
    """
    col_fn = collate_fn
    if model == SwaVModel:
        col_fn = swav_collate_fn
    elif model == DINOModel:
        col_fn = dino_collate_fn
    dataloader_train_ssl = torch.utils.data.DataLoader(
        dataset_train_ssl,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=col_fn,
        drop_last=True,
        num_workers=num_workers
    )

    dataloader_train_kNN = torch.utils.data.DataLoader(
        dataset_train_kNN,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    return dataloader_train_ssl, dataloader_train_kNN, dataloader_test




models = [
    BarlowTwinsModel
]
# models = [
#     BarlowTwinsModel,
#     BYOLModel,
#     DCL,
#     DCLW,
#     DINOModel,
#     MocoModel,
#     NNCLRModel,
#     SimCLRModel,
#     SimSiamModel,
#     SwaVModel,
# ]
bench_results = dict()

experiment_version = None
# loop through configurations and train models
seeds = [2]
for BenchmarkModel in models:
    # for depth in [2, 4, 6]:
    for resnet_type in ['resnet-34', 'resnet-18', 'resnet-50']:
    # for depth in [2]:
        wandb.init(project='SelfSupervised', entity='ibenshaul', mode="online",
                   sync_tensorboard=True, reinit=True, tags=['sanity', dataset])

        runs = []
        model_name = BenchmarkModel.__name__.replace('Model', '')
        conf = {'resnet-type': resnet_type, 'model_name': model_name, 'batch_size': batch_size, }
        wandb.config.update(conf)
        for seed in seeds:
            wandb.config.update({"seed": seed})
            pl.seed_everything(seed)
            dataloader_train_ssl, dataloader_train_kNN, dataloader_test = get_data_loaders(
                batch_size=batch_size,
                model=BenchmarkModel,
            )
            benchmark_model = BenchmarkModel(dataloader_train_kNN, classes, resnet_type=resnet_type)

            # Save logs to: {CWD}/benchmark_logs/cifar10/{experiment_version}/{model_name}/
            # If multiple runs are specified a subdirectory for each run is created.
            # sub_dir = model_name if n_runs <= 1 else f'{model_name}/run{seed}'
            # logger = TensorBoardLogger(
            #     save_dir=os.path.join(logs_root_dir, 'cifar10'),
            #     name='',
            #     sub_dir=sub_dir,
            #     version=experiment_version,
            # )
            logger = WandbLogger(project="SelfSupervised")

            if experiment_version is None:
                # Save results of all models under same version directory
                experiment_version = logger.version
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(os.path.join(logs_root_dir, 'cifar10'), 'checkpoints')
            )
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                gpus=gpus,
                default_root_dir=logs_root_dir,
                strategy=distributed_backend,
                sync_batchnorm=sync_batchnorm,
                logger=logger,
                callbacks=[checkpoint_callback],
                check_val_every_n_epoch=10
                # val_check_interval=0.05
            )
            start = time.time()
            trainer.fit(
                benchmark_model,
                train_dataloaders=dataloader_train_ssl,
                val_dataloaders=dataloader_test
            )
            end = time.time()
            run = {
                'model': model_name,
                'batch_size': batch_size,
                'epochs': max_epochs,
                'max_accuracy': benchmark_model.max_knn_accuracy,
                'runtime': end - start,
                'gpu_memory_usage': torch.cuda.max_memory_allocated(),
                'seed': seed,
            }

            runs.append(run)
            print(run)

            # delete model and trainer + free up cuda memory
            del benchmark_model
            del trainer
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        bench_results[model_name] = runs

# print results table
# header = (
#     f"| {'Model':<13} | {'Batch Size':>10} | {'Epochs':>6} "
#     f"| {'KNN Test Accuracy':>18} | {'Time':>10} | {'Peak GPU Usage':>14} |"
# )
# print('-' * len(header))
# print(header)
# print('-' * len(header))
# for model, results in bench_results.items():
#     runtime = np.array([result['runtime'] for result in results])
#     runtime = runtime.mean() / 60 # convert to min
#     accuracy = np.array([result['max_accuracy'] for result in results])
#     gpu_memory_usage = np.array([result['gpu_memory_usage'] for result in results])
#     gpu_memory_usage = gpu_memory_usage.max() / (1024**3) # convert to gbyte
#
#     if len(accuracy) > 1:
#         accuracy_msg = f"{accuracy.mean():>8.3f} +- {accuracy.std():>4.3f}"
#     else:
#         accuracy_msg = f"{accuracy.mean():>18.3f}"
#
#     print(
#         f"| {model:<13} | {batch_size:>10} | {max_epochs:>6} "
#         f"| {accuracy_msg} | {runtime:>6.1f} Min "
#         f"| {gpu_memory_usage:>8.1f} GByte |",
#         flush=True
#     )
# print('-' * len(header))
