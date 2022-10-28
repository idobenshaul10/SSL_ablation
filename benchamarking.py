""" Helper modules for benchmarking SSL models """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
import numpy as np
from numpy.linalg import norm

from cifar100_superclass import CIFAR100_mapper
from kmeans_pytorch import kmeans
from sklearn.cluster import KMeans

# code for kNN prediction from here:
# https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb

mapper = CIFAR100_mapper()


def compute_cdnv(feature, num_clusters, cluster_ids_x):
	N = num_clusters * [0]
	mean = num_clusters * [0]
	mean_s = num_clusters * [0]

	for c in range(num_clusters):
		idxs = (cluster_ids_x == c).nonzero(as_tuple=True)[0]
		if len(idxs) == 0:  # If no class-c in this batch
			continue

		h_c = feature[idxs, :]
		mean[c] += torch.sum(h_c, dim=0)
		N[c] += h_c.shape[0]
		mean_s[c] += torch.sum(torch.square(h_c))

	for c in range(num_clusters):
		mean[c] /= N[c]
		mean_s[c] /= N[c]

	avg_cdnv = 0
	total_num_pairs = num_clusters * (num_clusters - 1) / 2
	for class1 in range(num_clusters):
		for class2 in range(class1 + 1, num_clusters):
			variance1 = abs(mean_s[class1].item() - torch.sum(torch.square(mean[class1])).item())
			variance2 = abs(mean_s[class2].item() - torch.sum(torch.square(mean[class2])).item())
			variance_avg = (variance1 + variance2) / 2
			dist = torch.norm((mean[class1]) - (mean[class2])) ** 2
			dist = dist.item()
			cdnv = variance_avg / dist
			avg_cdnv += cdnv / total_num_pairs
	return avg_cdnv
def ncc_predict(feature: torch.Tensor,
				feature_bank: torch.Tensor,
				feature_labels: torch.Tensor,
				num_classes: int) -> torch.Tensor:

	feature_bank = feature_bank.cpu()
	feature = feature.T.cpu()
	class_means_bank = torch.zeros((num_classes, feature_bank.shape[0]))
	for c in range(num_classes):
		class_means_bank[c] = feature_bank[:, (feature_labels == c)].mean(dim=1)


	NCC_scores = [torch.norm(feature[i, :] - class_means_bank, dim=1) for i in range(feature.shape[0])]
	NCC_scores = torch.stack(NCC_scores)
	NCC_pred = torch.argmin(NCC_scores, dim=1)

	ncc_acc = (NCC_pred == feature_labels.cpu()).sum() / len(NCC_pred)
	cdnv = compute_cdnv(feature, num_classes, feature_labels)

	return ncc_acc.cpu().item(), cdnv

def predict_k_cluster_ncc(feature: torch.Tensor,
				feature_bank: torch.Tensor,
				feature_labels: torch.Tensor,
				num_clusters: int) -> torch.Tensor:
	feature_bank = feature_bank.T#.cpu()
	feature = feature.T

	# feature = feature.cpu()
	# cluster_ids_x, cluster_centers = kmeans(
	# 	X=feature_bank, num_clusters=num_clusters, distance='euclidean', device=torch.device('cpu')
	# )

	kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(feature_bank.cpu())
	cluster_ids_x = torch.tensor(kmeans.labels_)
	cluster_centers = torch.tensor(kmeans.cluster_centers_)
	feature = feature.cpu()

	# cluster_centers = cluster_centers.cuda()
	NCC_scores = [torch.norm(feature[i, :] - cluster_centers, dim=1) for i in range(feature.shape[0])]
	NCC_scores = torch.stack(NCC_scores)
	NCC_pred = torch.argmin(NCC_scores, dim=1)
	# NCC_acc = ((NCC_pred == cluster_ids_x.cuda()).sum() / NCC_pred.shape[0]).cpu()
	NCC_acc = ((NCC_pred == cluster_ids_x).sum() / NCC_pred.shape[0])

	N = num_clusters * [0]
	mean = num_clusters * [0]
	mean_s = num_clusters * [0]

	# COMPUTE CDNV
	for c in range(num_clusters):
		idxs = (cluster_ids_x == c).nonzero(as_tuple=True)[0]
		if len(idxs) == 0:
			continue

		h_c = feature[idxs, :]
		mean[c] += torch.sum(h_c, dim=0)
		N[c] += h_c.shape[0]
		mean_s[c] += torch.sum(torch.square(h_c))

	for c in range(num_clusters):
		idxs = (cluster_ids_x == c).nonzero(as_tuple=True)[0]
		if len(idxs) == 0:  # If no class-c in this batch
			continue

		h_c = feature[idxs, :]
		mean[c] += torch.sum(h_c, dim=0)
		N[c] += h_c.shape[0]
		mean_s[c] += torch.sum(torch.square(h_c))

	for c in range(num_clusters):
		mean[c] /= N[c]
		mean_s[c] /= N[c]

	avg_cdnv = 0
	total_num_pairs = num_clusters * (num_clusters - 1) / 2
	for class1 in range(num_clusters):
		for class2 in range(class1 + 1, num_clusters):
			variance1 = abs(mean_s[class1].item() - torch.sum(torch.square(mean[class1])).item())
			variance2 = abs(mean_s[class2].item() - torch.sum(torch.square(mean[class2])).item())
			variance_avg = (variance1 + variance2) / 2
			dist = torch.norm((mean[class1]) - (mean[class2])) ** 2
			dist = dist.item()
			cdnv = variance_avg / dist
			avg_cdnv += cdnv / total_num_pairs

	return NCC_acc, cluster_centers, avg_cdnv

def ncc_predict_superclass(feature: torch.Tensor,
						   feature_bank: torch.Tensor,
						   feature_labels: torch.Tensor,
						   num_classes: int) -> torch.Tensor:

	feature_bank = feature_bank.cpu()
	feature = feature.cpu()
	class_means_bank = torch.zeros((20, feature_bank.shape[0]))
	for superclass in range(20):
		superclass_instances = mapper(superclass)
		superclass_num = 0.
		for instance in superclass_instances:
			instance_features = feature_bank[:, (feature_labels == instance)]
			class_means_bank[superclass] += instance_features.sum(dim=1)
			superclass_num += instance_features.shape[1]

		class_means_bank[superclass] /= superclass_num

	NCC_scores = [torch.norm(feature[i, :] - class_means_bank, dim=1) for i in range(feature.shape[0])]
	NCC_scores = torch.stack(NCC_scores)
	NCC_pred = torch.argmin(NCC_scores, dim=1)
	return NCC_pred.cuda()


def knn_predict(feature: torch.Tensor,
				feature_bank: torch.Tensor,
				feature_labels: torch.Tensor,
				num_classes: int,
				knn_k: int = 200,
				knn_t: float = 0.1) -> torch.Tensor:
	"""Run kNN predictions on features based on a feature bank

	This method is commonly used to monitor performance of self-supervised
	learning methods.

	The default parameters are the ones
	used in https://arxiv.org/pdf/1805.01978v1.pdf.

	Args:
		feature:
			Tensor of shape [N, D] for which you want predictions
		feature_bank:
			Tensor of a database of features used for kNN
		feature_labels:
			Labels for the features in our feature_bank
		num_classes:
			Number of classes (e.g. `10` for CIFAR-10)
		knn_k:
			Number of k neighbors used for kNN
		knn_t:
			Temperature parameter to reweights similarities for kNN

	Returns:
		A tensor containing the kNN predictions

	Examples:
		>>> images, targets, _ = batch
		>>> feature = backbone(images).squeeze()
		>>> # we recommend to normalize the features
		>>> feature = F.normalize(feature, dim=1)
		>>> pred_labels = knn_predict(
		>>>     feature,
		>>>     feature_bank,
		>>>     targets_bank,
		>>>     num_classes=10,
		>>> )
	"""

	# compute cos similarity between each feature vector and feature bank ---> [B, N]
	sim_matrix = torch.mm(feature, feature_bank)
	# [B, K]
	sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
	# [B, K]
	sim_labels = torch.gather(feature_labels.expand(
		feature.size(0), -1), dim=-1, index=sim_indices)
	# we do a reweighting of the similarities
	sim_weight = (sim_weight / knn_t).exp()
	# counts for each class
	one_hot_label = torch.zeros(feature.size(
		0) * knn_k, num_classes, device=sim_labels.device)
	# [B*K, C]
	one_hot_label = one_hot_label.scatter(
		dim=-1, index=sim_labels.view(-1, 1), value=1.0)
	# weighted score ---> [B, C]
	pred_scores = torch.sum(one_hot_label.view(feature.size(
		0), -1, num_classes) * sim_weight.unsqueeze(dim=-1), dim=1)
	pred_labels = pred_scores.argsort(dim=-1, descending=True)
	return pred_labels


class BenchmarkModule(pl.LightningModule):
	"""A PyTorch Lightning Module for automated kNN callback

	At the end of every training epoch we create a feature bank by feeding the
	`dataloader_kNN` passed to the module through the backbone.
	At every validation step we predict features on the validation data.
	After all predictions on validation data (validation_epoch_end) we evaluate
	the predictions on a kNN classifier on the validation data using the
	feature_bank features from the train data.

	We can access the highest test accuracy during a kNN prediction
	using the `max_accuracy` attribute.

	Attributes:
		backbone:
			The backbone model used for kNN validation. Make sure that you set the
			backbone when inheriting from `BenchmarkModule`.
		max_accuracy:
			Floating point number between 0.0 and 1.0 representing the maximum
			test accuracy the benchmarked model has achieved.
		dataloader_kNN:
			Dataloader to be used after each training epoch to create feature bank.
		num_classes:
			Number of classes. E.g. for cifar10 we have 10 classes. (default: 10)
		knn_k:
			Number of nearest neighbors for kNN
		knn_t:
			Temperature parameter for kNN

	Examples:
		>>> class SimSiamModel(BenchmarkingModule):
		>>>     def __init__(dataloader_kNN, num_classes):
		>>>         super().__init__(dataloader_kNN, num_classes)
		>>>         resnet = lightly.models.ResNetGenerator('resnet-18')
		>>>         self.backbone = nn.Sequential(
		>>>             *list(resnet.children())[:-1],
		>>>             nn.AdaptiveAvgPool2d(1),
		>>>         )
		>>>         self.resnet_simsiam =
		>>>             lightly.models.SimSiam(self.backbone, num_ftrs=512)
		>>>         self.criterion = lightly.loss.SymNegCosineSimilarityLoss()
		>>>
		>>>     def forward(self, x):
		>>>         self.resnet_simsiam(x)
		>>>
		>>>     def training_step(self, batch, batch_idx):
		>>>         (x0, x1), _, _ = batch
		>>>         x0, x1 = self.resnet_simsiam(x0, x1)
		>>>         loss = self.criterion(x0, x1)
		>>>         return loss
		>>>     def configure_optimizers(self):
		>>>         optim = torch.optim.SGD(
		>>>             self.resnet_simsiam.parameters(), lr=6e-2, momentum=0.9
		>>>         )
		>>>         return [optim]
		>>>
		>>> model = SimSiamModel(dataloader_train_kNN)
		>>> trainer = pl.Trainer()
		>>> trainer.fit(
		>>>     model,
		>>>     train_dataloader=dataloader_train_ssl,
		>>>     val_dataloaders=dataloader_test
		>>> )
		>>> # you can get the peak accuracy using
		>>> print(model.max_accuracy)

	"""

	def __init__(self,
				 dataloader_kNN: DataLoader,
				 num_classes: int,
				 knn_k: int = 200,
				 knn_t: float = 0.1):
		super().__init__()
		self.backbone = nn.Module()
		self.max_knn_accuracy = 0.0
		self.max_nnc_accuracy = 0.0
		self.dataloader_kNN = dataloader_kNN
		self.num_classes = num_classes
		self.knn_k = knn_k
		self.knn_t = knn_t

		# create dummy param to keep track of the device the model is using
		self.dummy_param = nn.Parameter(torch.empty(0))

	def training_epoch_end(self, outputs):
		# update feature bank at the end of each training epoch
		self.backbone.eval()
		self.feature_bank = []
		self.targets_bank = []
		with torch.no_grad():
			for data in self.dataloader_kNN:
				img, target, _ = data
				img = img.to(self.dummy_param.device)
				target = target.to(self.dummy_param.device)
				feature = self.backbone(img).squeeze()
				feature = F.normalize(feature, dim=1)
				self.feature_bank.append(feature)
				self.targets_bank.append(target)
		self.feature_bank = torch.cat(
			self.feature_bank, dim=0).t().contiguous()
		self.targets_bank = torch.cat(
			self.targets_bank, dim=0).t().contiguous()
		self.backbone.train()

	def validation_step(self, batch, batch_idx):
		# we can only do kNN predictions once we have a feature bank
		if hasattr(self, 'feature_bank') and hasattr(self, 'targets_bank'):
			if batch_idx == 0:
				results = {}
				# for k in [3, 10, 20, 50, 100, 500]:
				#
				# 	k_NCC_acc, cluster_centers, cdnv = predict_k_cluster_ncc(self.feature_bank,
				# 									   self.feature_bank,
				# 									   self.targets_bank,
				# 									   k)
				# 	print(f"NCC Acc for k={k}: {k_NCC_acc}, cdnv: {cdnv}")
				# 	results[k] = (k_NCC_acc, cluster_centers, cdnv)

				ncc, cdnv = ncc_predict(self.feature_bank,
													   self.feature_bank,
													   self.targets_bank,
													   self.num_classes)

				results['original'] = (ncc, None, cdnv)
				# results['ORIGINAL_ncc_acc'], results['ORIGINAL_cdnv'] = ncc_predict(self.feature_bank,
				# 																	self.feature_bank,
				# 																	self.targets_bank,
				# 																	self.num_classes)
				return results

			# import pdb; pdb.set_trace()
			#
			# images, targets, _ = batch
			# targets_superclass = mapper.coarse_labels[targets.cpu().numpy()]
			#
			# feature = self.backbone(images).squeeze()
			# feature = F.normalize(feature, dim=1)
			# # pred_labels = knn_predict(
			# # 	feature,
			# # 	self.feature_bank,
			# # 	self.targets_bank,
			# # 	self.num_classes,
			# # 	self.knn_k,
			# # 	self.knn_t
			# # )
			# num = images.size()
			# # top1 = (pred_labels[:, 0] == targets).float().sum()
			# # pred_labels_ncc = ncc_predict(
			# # 	feature,
			# # 	self.feature_bank,
			# # 	self.targets_bank,
			# # 	self.num_classes
			# # )
			#
			# # pred_superclass_labels_ncc = ncc_predict_superclass(
			# # 	feature,
			# # 	self.feature_bank,
			# # 	self.targets_bank,
			# # 	self.num_classes
			# # )
			#
			#
			#
			# 	# knn_predict
			#
			# num = images.size()
			# top1_ncc = (pred_labels_ncc == targets).float().sum()
			# top1_ncc_superclass = (pred_superclass_labels_ncc.cpu().numpy() == targets_superclass).sum()


	def validation_epoch_end(self, outputs):
		device = self.dummy_param.device
		if outputs:
			log_result = {}
			for key, value in outputs[0].items():
				ncc, _, cdnv = value
				log_result.update({'epoch': self.current_epoch, f'NCC_{key}_clusters': ncc, f'CDNV_{key}_clusters': cdnv})

			wandb.log(log_result)
			# total_num = torch.Tensor([0]).to(device)
			# total_top1_ncc = torch.Tensor([0.]).to(device)
			# total_top1_ncc_superclass = torch.Tensor([0.]).to(device)
			# for (num, top1_ncc, top1_ncc_superclass) in outputs:
			# 	total_num += num[0]
			# 	total_top1_ncc_superclass += top1_ncc_superclass
			# 	total_top1_ncc += top1_ncc
			#
			# if dist.is_initialized() and dist.get_world_size() > 1:
			# 	dist.all_reduce(total_num)
			# 	dist.all_reduce(total_top1_ncc)
			# 	dist.all_reduce(total_top1_ncc_superclass)
			#
			# ncc_acc = float(total_top1_ncc.item() / total_num.item())
			# ncc_superclass_acc = float(total_top1_ncc_superclass.item() / total_num.item())
			#
			# self.log('ncc_accuracy', ncc_acc, prog_bar=True)
			#
			# wandb.log({'epoch': self.current_epoch, 'ncc_acc': ncc_acc,
			# 		'ncc_superclass_acc': ncc_superclass_acc})
