import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from skimage import io
import numpy as np
from os import listdir
from os.path import isfile, join

from fastai.vision.all import PILImage
import matplotlib.pyplot as plt
from torchvision import transforms

# Set device

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FoodDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.files = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.files[idx])

        image = PILImage.create(img_name)

        if self.transform:
            sample = self.transform(image)
        print("processed image")
        return sample


IMG_HEIGHT = 28
IMG_WIDTH = 28
# Load data
def transform(img):
    img_resized = img.resize((IMG_HEIGHT,IMG_WIDTH))
    img_np = np.array(img_resized)
    img_np = img_np/255
    img_np = img_np.reshape(3, IMG_HEIGHT, IMG_WIDTH)
    # img_np = img_np.astype(np.double)
    print(type(img_np[0][0][0]))
    x_np = torch.from_numpy(img_np).double()
    return x_np


dataset = FoodDataset(root_dir='tripadvisor_dataset/tripadvisor_mini', transform=transform)

# print(dataset.__getitem__(0))

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

train_features= next(iter(dataloader))


img = train_features[0].squeeze().resize(IMG_WIDTH,IMG_WIDTH,3)

# print(f"Feature batch shape: {train_features.size()}")
# plt.imshow(img, cmap="gray")
# plt.show()


import torch.nn as nn

class ClusteringLayer(nn.Module):
        """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.
    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """
        # TODO in_features en out_features aanpassen naar onze dataset (hier 10 want 10 mogelijke cijfers, is ook het aantal clusters dat ze gepakt hebben)
        def __init__(self, in_features=10, out_features=10, alpha=1.0):
            super(ClusteringLayer, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.alpha = alpha
            self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
            self.weight = nn.init.xavier_uniform_(self.weight)

        def forward(self, x):
            x = x.unsqueeze(1) - self.weight
            x = torch.mul(x, x)
            x = torch.sum(x, dim=2)
            x = 1.0 + (x / self.alpha)
            x = 1.0 / x
            x = x ** ((self.alpha +1.0) / 2.0)
            x = torch.t(x) / torch.sum(x, dim=1)
            x = torch.t(x)
            return x

        def extra_repr(self):
            return 'in_features={}, out_features={}, alpha={}'.format(
                self.in_features, self.out_features, self.alpha
            )

        def set_weight(self, tensor):
            self.weight = nn.Parameter(tensor)


            

class CAE(nn.Module):
    def __init__(self, input_shape=[28,28,1], num_clusters=10, filters=[32, 64, 128], leaky=True, neg_slope=0.01, activations=False, bias=True):
        super(CAE, self).__init__()
        self.activations = activations
        # bias = True
        self.pretrained = False
        self.num_clusters = num_clusters
        self.input_shape = input_shape
        self.filters = filters
        print(input_shape, input_shape[2])
        self.conv1 = nn.Conv2d(input_shape[2], filters[0], 5, stride=2, padding=2, bias=bias)
        
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
        self.conv3 = nn.Conv2d(filters[1], filters[2], 3, stride=2, padding=0, bias=bias)

        lin_features_len = ((input_shape[0]//2//2-1) // 2) * ((input_shape[0]//2//2-1) // 2) * filters[2]
        
        print("lengte is", lin_features_len, "en", num_clusters)
        self.embedding = nn.Linear(lin_features_len, num_clusters, bias=bias)
        self.deembedding = nn.Linear(num_clusters, lin_features_len, bias=bias)
        out_pad = 1 if input_shape[0] // 2 // 2 % 2 == 0 else 0
        self.deconv3 = nn.ConvTranspose2d(filters[2], filters[1], 3, stride=2, padding=0, output_padding=out_pad, bias=bias)
        out_pad = 1 if input_shape[0] // 2 % 2 == 0 else 0
        self.deconv2 = nn.ConvTranspose2d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad, bias=bias)
        out_pad = 1 if input_shape[0] % 2 == 0 else 0
        self.deconv1 = nn.ConvTranspose2d(filters[0], input_shape[2], 5, stride=2, padding=2, output_padding=out_pad, bias=bias)
        self.clustering = ClusteringLayer(num_clusters, num_clusters)

        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()


    def forward(self, x):
        print("in forward", x.shape)
        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)
        if self.activations:
            x = self.sig(x)

        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        extra_out = x
        clustering_out = self.clustering(x)
        x = self.deembedding(x)

        x = x.view(x.size(0), self.filters[2], ((self.input_shape[0]//2//2-1) // 2), ((self.input_shape[0]//2//2-1) // 2))
        x = self.deconv3(x)

        x = self.deconv2(x)

        x = self.deconv1(x)
        if self.activations:
            x = self.tanh(x)
        return x, clustering_out, extra_out



cae = CAE()
# cae.double()


# res = cae(train_features[0])










### test op fashion
from torchvision.transforms import ToTensor
from torchvision import datasets

training_data_fashion = training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
print(next(iter(train_dataloader))[0])
res = cae(next(iter(train_dataloader))[0])
