import copy
import lightly
import torch
import torch.nn as nn
from lightly.models import modules
from lightly.models.modules import heads
from lightly.models import utils
from benchamarking import BenchmarkModule
import wandb
from models.ConvArch import ConvArch

# set max_epochs to 800 for long run (takes around 10h on a single V100)
max_epochs = 400
num_workers = 0
knn_k = 200
knn_t = 0.1
classes = 10

#  Set to True to enable Distributed Data Parallel training.
distributed = False

# Set to True to enable Synchronized Batch Norm (requires distributed=True).
# If enabled the batch norm is calculated over all gpus, otherwise the batch
# norm is only calculated from samples on the same gpu.
sync_batchnorm = False

# Set to True to gather features from all gpus before calculating
# the loss (requires distributed=True).
#  If enabled then the loss on every gpu is calculated with features from all
# gpus, otherwise only features from the same gpu are used.
gather_distributed = False

# benchmark

batch_size = 512
lr_factor = batch_size / 128  #  scales the learning rate linearly with batch size


class MocoModel(BenchmarkModule):
	def __init__(self, dataloader_kNN, num_classes):
		super().__init__(dataloader_kNN, num_classes)

		# create a ResNet backbone and remove the classification head
		num_splits = 0 if sync_batchnorm else 8
		resnet = lightly.models.ResNetGenerator('resnet-18', num_splits=num_splits)
		self.backbone = nn.Sequential(
			*list(resnet.children())[:-1],
			nn.AdaptiveAvgPool2d(1)
		)

		# create a moco model based on ResNet
		self.projection_head = heads.MoCoProjectionHead(512, 512, 128)
		self.backbone_momentum = copy.deepcopy(self.backbone)
		self.projection_head_momentum = copy.deepcopy(self.projection_head)
		utils.deactivate_requires_grad(self.backbone_momentum)
		utils.deactivate_requires_grad(self.projection_head_momentum)

		# create our loss with the optional memory bank
		self.criterion = lightly.loss.NTXentLoss(
			temperature=0.1,
			memory_bank_size=4096,
		)

	def forward(self, x):
		x = self.backbone(x).flatten(start_dim=1)
		return self.projection_head(x)

	def training_step(self, batch, batch_idx):
		(x0, x1), _, _ = batch

		# update momentum
		utils.update_momentum(self.backbone, self.backbone_momentum, 0.99)
		utils.update_momentum(self.projection_head, self.projection_head_momentum, 0.99)

		def step(x0_, x1_):
			x1_, shuffle = utils.batch_shuffle(x1_, distributed=distributed)
			x0_ = self.backbone(x0_).flatten(start_dim=1)
			x0_ = self.projection_head(x0_)

			x1_ = self.backbone_momentum(x1_).flatten(start_dim=1)
			x1_ = self.projection_head_momentum(x1_)
			x1_ = utils.batch_unshuffle(x1_, shuffle, distributed=distributed)
			return x0_, x1_

		# We use a symmetric loss (model trains faster at little compute overhead)
		# https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
		loss_1 = self.criterion(*step(x0, x1))
		loss_2 = self.criterion(*step(x1, x0))

		loss = 0.5 * (loss_1 + loss_2)
		self.log('train_loss_ssl', loss)
		return loss

	def configure_optimizers(self):
		params = list(self.backbone.parameters()) + list(self.projection_head.parameters())
		optim = torch.optim.SGD(
			params,
			lr=6e-2 * lr_factor,
			momentum=0.9,
			weight_decay=5e-4,
		)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
		return [optim], [scheduler]


class SimCLRModel(BenchmarkModule):
	def __init__(self, dataloader_kNN, num_classes):
		super().__init__(dataloader_kNN, num_classes)
		# create a ResNet backbone and remove the classification head
		resnet = lightly.models.ResNetGenerator('resnet-18')
		self.backbone = nn.Sequential(
			*list(resnet.children())[:-1],
			nn.AdaptiveAvgPool2d(1)
		)
		self.projection_head = heads.SimCLRProjectionHead(512, 512, 128)
		self.criterion = lightly.loss.NTXentLoss()

	def forward(self, x):
		x = self.backbone(x).flatten(start_dim=1)
		z = self.projection_head(x)
		return z

	def training_step(self, batch, batch_index):
		(x0, x1), _, _ = batch
		z0 = self.forward(x0)
		z1 = self.forward(x1)
		loss = self.criterion(z0, z1)
		self.log('train_loss_ssl', loss)
		return loss

	def configure_optimizers(self):
		optim = torch.optim.SGD(
			self.parameters(),
			lr=6e-2 * lr_factor,
			momentum=0.9,
			weight_decay=5e-4
		)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
		return [optim], [scheduler]


class SimSiamModel(BenchmarkModule):
	def __init__(self, dataloader_kNN, num_classes):
		super().__init__(dataloader_kNN, num_classes)
		# create a ResNet backbone and remove the classification head
		resnet = lightly.models.ResNetGenerator('resnet-18')
		self.backbone = nn.Sequential(
			*list(resnet.children())[:-1],
			nn.AdaptiveAvgPool2d(1)
		)
		self.prediction_head = heads.SimSiamPredictionHead(2048, 512, 2048)
		# use a 2-layer projection head for cifar10 as described in the paper
		self.projection_head = heads.ProjectionHead([
			(
				512,
				2048,
				nn.BatchNorm1d(2048),
				nn.ReLU(inplace=True)
			),
			(
				2048,
				2048,
				nn.BatchNorm1d(2048),
				None
			)
		])
		self.criterion = lightly.loss.NegativeCosineSimilarity()

	def forward(self, x):
		f = self.backbone(x).flatten(start_dim=1)
		z = self.projection_head(f)
		p = self.prediction_head(z)
		z = z.detach()
		return z, p

	def training_step(self, batch, batch_idx):
		(x0, x1), _, _ = batch
		z0, p0 = self.forward(x0)
		z1, p1 = self.forward(x1)
		loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
		self.log('train_loss_ssl', loss)
		return loss

	def configure_optimizers(self):
		optim = torch.optim.SGD(
			self.parameters(),
			lr=6e-2,  #  no lr-scaling, results in better training stability
			momentum=0.9,
			weight_decay=5e-4
		)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
		return [optim], [scheduler]


class BarlowTwinsModel(BenchmarkModule):
	def __init__(self, dataloader_kNN, num_classes, depth=10, width=100, resnet_type='resnet-18'):
		super().__init__(dataloader_kNN, num_classes)
		# create a ResNet backbone and remove the classification head
		if resnet_type is not None:
			# ResNet version from resnet -{9, 18, 34, 50, 101, 152}.
			resnet = lightly.models.ResNetGenerator(resnet_type)
			self.backbone = nn.Sequential(
				*list(resnet.children())[:-1],
				nn.AdaptiveAvgPool2d(1),
			)
		else:
			input_image_width = self.dataloader_kNN.dataset[0][0].shape[1]
			self.backbone = ConvArch(input_channel_number=3, input_image_width=input_image_width, output_size=512,
									 depth=depth, width=width)

		self.projection_head = heads.ProjectionHead([
			(
				512,
				2048,
				nn.BatchNorm1d(2048),
				nn.ReLU(inplace=True)
			),
			(
				2048,
				2048,
				None,
				None
			)
		])

		self.criterion = lightly.loss.BarlowTwinsLoss(gather_distributed=gather_distributed)

	def forward(self, x):
		x = self.backbone(x).flatten(start_dim=1)
		z = self.projection_head(x)
		return z

	def training_step(self, batch, batch_index):
		(x0, x1), _, _ = batch
		z0 = self.forward(x0)
		z1 = self.forward(x1)
		loss = self.criterion(z0, z1)
		self.log('train_loss_ssl', loss)
		return loss

	def configure_optimizers(self):
		wandb.config.update({'optimizer': 'SGD'})
		optim = torch.optim.SGD(
			self.parameters(),
			lr=6e-2 * lr_factor,
			momentum=0.9,
			weight_decay=5e-4
		)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
		return [optim], [scheduler]


class BYOLModel(BenchmarkModule):
	def __init__(self, dataloader_kNN, num_classes):
		super().__init__(dataloader_kNN, num_classes)
		# create a ResNet backbone and remove the classification head
		resnet = lightly.models.ResNetGenerator('resnet-18')
		self.backbone = nn.Sequential(
			*list(resnet.children())[:-1],
			nn.AdaptiveAvgPool2d(1)
		)

		# create a byol model based on ResNet
		self.projection_head = heads.BYOLProjectionHead(512, 1024, 256)
		self.prediction_head = heads.BYOLPredictionHead(256, 1024, 256)

		self.backbone_momentum = copy.deepcopy(self.backbone)
		self.projection_head_momentum = copy.deepcopy(self.projection_head)

		utils.deactivate_requires_grad(self.backbone_momentum)
		utils.deactivate_requires_grad(self.projection_head_momentum)

		self.criterion = lightly.loss.NegativeCosineSimilarity()

	def forward(self, x):
		y = self.backbone(x).flatten(start_dim=1)
		z = self.projection_head(y)
		p = self.prediction_head(z)
		return p

	def forward_momentum(self, x):
		y = self.backbone_momentum(x).flatten(start_dim=1)
		z = self.projection_head_momentum(y)
		z = z.detach()
		return z

	def training_step(self, batch, batch_idx):
		utils.update_momentum(self.backbone, self.backbone_momentum, m=0.99)
		utils.update_momentum(self.projection_head, self.projection_head_momentum, m=0.99)
		(x0, x1), _, _ = batch
		p0 = self.forward(x0)
		z0 = self.forward_momentum(x0)
		p1 = self.forward(x1)
		z1 = self.forward_momentum(x1)
		loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
		self.log('train_loss_ssl', loss)
		return loss

	def configure_optimizers(self):
		params = list(self.backbone.parameters()) \
				 + list(self.projection_head.parameters()) \
				 + list(self.prediction_head.parameters())
		optim = torch.optim.SGD(
			params,
			lr=6e-2 * lr_factor,
			momentum=0.9,
			weight_decay=5e-4,
		)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
		return [optim], [scheduler]


class SwaVModel(BenchmarkModule):
	def __init__(self, dataloader_kNN, num_classes):
		super().__init__(dataloader_kNN, num_classes)
		# create a ResNet backbone and remove the classification head
		resnet = lightly.models.ResNetGenerator('resnet-18')
		self.backbone = nn.Sequential(
			*list(resnet.children())[:-1],
			nn.AdaptiveAvgPool2d(1)
		)

		self.projection_head = heads.SwaVProjectionHead(512, 512, 128)
		self.prototypes = heads.SwaVPrototypes(128, 512)  # use 512 prototypes

		self.criterion = lightly.loss.SwaVLoss(sinkhorn_gather_distributed=gather_distributed)

	def forward(self, x):
		x = self.backbone(x).flatten(start_dim=1)
		x = self.projection_head(x)
		x = nn.functional.normalize(x, dim=1, p=2)
		return self.prototypes(x)

	def training_step(self, batch, batch_idx):
		# normalize the prototypes so they are on the unit sphere
		self.prototypes.normalize()

		# the multi-crop dataloader returns a list of image crops where the
		# first two items are the high resolution crops and the rest are low
		# resolution crops
		multi_crops, _, _ = batch
		multi_crop_features = [self.forward(x) for x in multi_crops]

		# split list of crop features into high and low resolution
		high_resolution_features = multi_crop_features[:2]
		low_resolution_features = multi_crop_features[2:]

		# calculate the SwaV loss
		loss = self.criterion(
			high_resolution_features,
			low_resolution_features
		)

		self.log('train_loss_ssl', loss)
		return loss

	def configure_optimizers(self):
		optim = torch.optim.Adam(
			self.parameters(),
			lr=1e-3 * lr_factor,
			weight_decay=1e-6,
		)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
		return [optim], [scheduler]


class NNCLRModel(BenchmarkModule):
	def __init__(self, dataloader_kNN, num_classes):
		super().__init__(dataloader_kNN, num_classes)
		# create a ResNet backbone and remove the classification head
		resnet = lightly.models.ResNetGenerator('resnet-18')
		self.backbone = nn.Sequential(
			*list(resnet.children())[:-1],
			nn.AdaptiveAvgPool2d(1)
		)
		self.prediction_head = heads.NNCLRPredictionHead(256, 4096, 256)
		# use only a 2-layer projection head for cifar10
		self.projection_head = heads.ProjectionHead([
			(
				512,
				2048,
				nn.BatchNorm1d(2048),
				nn.ReLU(inplace=True)
			),
			(
				2048,
				256,
				nn.BatchNorm1d(256),
				None
			)
		])

		self.criterion = lightly.loss.NTXentLoss()
		self.memory_bank = modules.NNMemoryBankModule(size=4096)

	def forward(self, x):
		y = self.backbone(x).flatten(start_dim=1)
		z = self.projection_head(y)
		p = self.prediction_head(z)
		z = z.detach()
		return z, p

	def training_step(self, batch, batch_idx):
		(x0, x1), _, _ = batch
		z0, p0 = self.forward(x0)
		z1, p1 = self.forward(x1)
		z0 = self.memory_bank(z0, update=False)
		z1 = self.memory_bank(z1, update=True)
		loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
		return loss

	def configure_optimizers(self):
		optim = torch.optim.SGD(
			self.parameters(),
			lr=6e-2 * lr_factor,
			momentum=0.9,
			weight_decay=5e-4,
		)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
		return [optim], [scheduler]


class DINOModel(BenchmarkModule):
	def __init__(self, dataloader_kNN, num_classes):
		super().__init__(dataloader_kNN, num_classes)
		# create a ResNet backbone and remove the classification head
		resnet = lightly.models.ResNetGenerator('resnet-18')
		self.backbone = nn.Sequential(
			*list(resnet.children())[:-1],
			nn.AdaptiveAvgPool2d(1)
		)
		self.head = self._build_projection_head()
		self.teacher_backbone = copy.deepcopy(self.backbone)
		self.teacher_head = self._build_projection_head()

		utils.deactivate_requires_grad(self.teacher_backbone)
		utils.deactivate_requires_grad(self.teacher_head)

		self.criterion = lightly.loss.DINOLoss(output_dim=2048)

	def _build_projection_head(self):
		head = heads.DINOProjectionHead(512, 2048, 256, 2048, batch_norm=True)
		# use only 2 layers for cifar10
		head.layers = heads.ProjectionHead([
			(512, 2048, nn.BatchNorm1d(2048), nn.GELU()),
			(2048, 256, None, None),
		]).layers
		return head

	def forward(self, x):
		y = self.backbone(x).flatten(start_dim=1)
		z = self.head(y)
		return z

	def forward_teacher(self, x):
		y = self.teacher_backbone(x).flatten(start_dim=1)
		z = self.teacher_head(y)
		return z

	def training_step(self, batch, batch_idx):
		utils.update_momentum(self.backbone, self.teacher_backbone, m=0.99)
		utils.update_momentum(self.head, self.teacher_head, m=0.99)
		views, _, _ = batch
		views = [view.to(self.device) for view in views]
		global_views = views[:2]
		teacher_out = [self.forward_teacher(view) for view in global_views]
		student_out = [self.forward(view) for view in views]
		loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
		self.log('train_loss_ssl', loss)
		return loss

	def configure_optimizers(self):
		param = list(self.backbone.parameters()) \
				+ list(self.head.parameters())
		optim = torch.optim.SGD(
			param,
			lr=6e-2 * lr_factor,
			momentum=0.9,
			weight_decay=5e-4,
		)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
		return [optim], [scheduler]


class DCL(BenchmarkModule):
	def __init__(self, dataloader_kNN, num_classes):
		super().__init__(dataloader_kNN, num_classes)
		# create a ResNet backbone and remove the classification head
		resnet = lightly.models.ResNetGenerator('resnet-18')
		self.backbone = nn.Sequential(
			*list(resnet.children())[:-1],
			nn.AdaptiveAvgPool2d(1)
		)
		self.projection_head = heads.SimCLRProjectionHead(512, 512, 128)
		self.criterion = lightly.loss.DCLLoss()

	def forward(self, x):
		x = self.backbone(x).flatten(start_dim=1)
		z = self.projection_head(x)
		return z

	def training_step(self, batch, batch_index):
		(x0, x1), _, _ = batch
		z0 = self.forward(x0)
		z1 = self.forward(x1)
		loss = self.criterion(z0, z1)
		self.log('train_loss_ssl', loss)
		return loss

	def configure_optimizers(self):
		optim = torch.optim.SGD(
			self.parameters(),
			lr=6e-2 * lr_factor,
			momentum=0.9,
			weight_decay=5e-4
		)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
		return [optim], [scheduler]


class DCLW(BenchmarkModule):
	def __init__(self, dataloader_kNN, num_classes):
		super().__init__(dataloader_kNN, num_classes)
		# create a ResNet backbone and remove the classification head
		resnet = lightly.models.ResNetGenerator('resnet-18')
		self.backbone = nn.Sequential(
			*list(resnet.children())[:-1],
			nn.AdaptiveAvgPool2d(1)
		)
		self.projection_head = heads.SimCLRProjectionHead(512, 512, 128)
		self.criterion = lightly.loss.DCLWLoss()

	def forward(self, x):
		x = self.backbone(x).flatten(start_dim=1)
		z = self.projection_head(x)
		return z

	def training_step(self, batch, batch_index):
		(x0, x1), _, _ = batch
		z0 = self.forward(x0)
		z1 = self.forward(x1)
		loss = self.criterion(z0, z1)
		self.log('train_loss_ssl', loss)
		return loss

	def configure_optimizers(self):
		optim = torch.optim.SGD(
			self.parameters(),
			lr=6e-2 * lr_factor,
			momentum=0.9,
			weight_decay=5e-4
		)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
		return [optim], [scheduler]
