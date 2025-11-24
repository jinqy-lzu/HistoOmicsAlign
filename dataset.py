import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader,ConcatDataset, random_split
import pandas
from torchvision import transforms, datasets
from PIL import Image
import pandas as pd
class GeneHistopDataset_cls(Dataset):
    def __init__(self, gene_expression_file, select_data ,image_dir, transform=None):
        self.gene_data = pd.read_csv(gene_expression_file)
        self.image_dir = image_dir
        self.transform = transform
        self.gene = None
        self.img = None
        self.label = None
        self.valid_indices = []
        for idx in range(len(self.gene_data)):
            sample_name = self.gene_data.iloc[idx, 1]
            image_path = os.path.join(self.image_dir, f"{sample_name}.pt")
            if os.path.exists(image_path):
                self.valid_indices.append(idx)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        sample_name = self.gene_data.iloc[real_idx, 1]
        gene_expression = self.gene_data.iloc[real_idx, 3:-1].values.astype(float)
        label = self.gene_data.iloc[real_idx, 2]
        image_feat_path = os.path.join(self.image_dir, f"{sample_name}.pt")
        image_feat  = torch.load(image_feat_path,weights_only=True)
        self.gene = torch.tensor(gene_expression, dtype=torch.float32)
        self.label = torch.tensor(label)
        self.img =  image_feat
        return self.gene, self.label, self.img
    def get_size(self):
        real_idx = self.valid_indices[0]
        sample_name = self.gene_data.iloc[real_idx, 1]
        gene_expression = self.gene_data.iloc[real_idx, 3:-1].values.astype(float)
        image_path = os.path.join(self.image_dir, f"{sample_name}.pt")
        image  = torch.load(image_path, weights_only=True)
        return gene_expression.shape, image.shape

def geneHistopDataloader_cls(geneFile, selectData, imgDir, batchSize = 8,train_ratio = 0.8):
    transformTrain = transforms.Compose(
            [transforms.Resize((1024, 1024)),
             transforms.RandomCrop(1024, padding=4), 
             transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])
    transformTest = transforms.Compose(
            [   transforms.Resize((1024, 1024)),
                transforms.ToTensor(), 
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])

    Dataset = GeneHistopDataset_cls(geneFile, selectData, imgDir, transformTrain)
    total_size = len(Dataset)
    print("Total data sample : " + str(total_size))
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size

    torch.manual_seed(42)
    train_dataset, test_dataset = random_split(Dataset, [train_size, test_size])
    trainDataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=1)
    testDataloader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False, num_workers=1)
    gene_size, img_feat_size = Dataset.get_size()
    return trainDataloader, testDataloader, gene_size, img_feat_size

class GeneHistopDataset_surv(Dataset):
    def __init__(self, gene_expression_file, select_data ,image_dir, transform=None):
        self.gene_data = pd.read_csv(gene_expression_file)
        self.image_dir = image_dir
        self.transform = transform
        self.gene = None
        self.img = None
        self.label = None
        self.event_time=None
        self.survival_status = None
        self.sample_id = None
        self.valid_indices = []
        for idx in range(len(self.gene_data)):
            sample_name = self.gene_data.iloc[idx, 1]
            image_path = os.path.join(self.image_dir, f"{sample_name}.pt")
            if os.path.exists(image_path):
                self.valid_indices.append(idx)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        sample_id = self.gene_data.iloc[real_idx, 0]
        self.sample_id = sample_id
        sample_name = self.gene_data.iloc[real_idx, 1]
        gene_expression = self.gene_data.iloc[real_idx, 6:-1].values.astype(float)
        time = self.gene_data.iloc[real_idx, 2] 
        self.event_time = torch.tensor(time ,dtype=torch.float32)
        self.survival_status = torch.tensor(self.gene_data.iloc[real_idx, 3])
        self.survival_status = 1-self.survival_status
        label = self.gene_data.iloc[real_idx, 4]

        image_feat_path = os.path.join(self.image_dir, f"{sample_name}.pt")
        image_feat  = torch.load(image_feat_path,weights_only=True)
        
        self.gene = torch.tensor(gene_expression, dtype=torch.float32)
        self.label = torch.tensor(label)
        self.img =  image_feat
        return self.gene, self.label, self.img,self.event_time, self.survival_status,self.sample_id
    def get_size(self):
        real_idx = self.valid_indices[0]
        sample_name = self.gene_data.iloc[real_idx, 1]
        gene_expression = self.gene_data.iloc[real_idx, 6:-1].values.astype(float)
        image_path = os.path.join(self.image_dir, f"{sample_name}.pt")
        image  = torch.load(image_path, weights_only=True)
        return gene_expression.shape, image.shape

def geneHistopDataloader_surv(geneFile, selectData, imgDir, batchSize = 8,train_ratio = 0.8):
    transformTrain = transforms.Compose(
            [transforms.Resize((1024, 1024)),
             transforms.RandomCrop(1024, padding=4), 
             transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])
    transformTest = transforms.Compose(
            [   transforms.Resize((1024, 1024)),
                transforms.ToTensor(), 
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])

    Dataset = GeneHistopDataset_surv(geneFile, selectData, imgDir, transformTrain)
    total_size = len(Dataset)
    print("Total data sample : " + str(total_size))
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size 
    torch.manual_seed(75)
    train_dataset, test_dataset = random_split(Dataset, [train_size, test_size])
    trainDataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=1)
    testDataloader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False, num_workers=1)
    gene_size, img_feat_size = Dataset.get_size()
    return trainDataloader, testDataloader, gene_size, img_feat_size

