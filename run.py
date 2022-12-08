# Import libraries for convolutional neural network and data processing
# We will use pytorch to create a convolutional neural network training on MNIST dataset.
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pickle
import argparse

# Import the MNIST dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from basic_CNN import CNN

# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# dataset = "cifar10"
# seed = 10

# CIFAR 10 stats
mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]


parser = argparse.ArgumentParser(description="Fit different DCAs.")
parser.add_argument("--cuda", type=int, help="save directory")
parser.add_argument("--experiment", type=int, default=0, help="Experiment number")
parser.add_argument("--dataset", type=str, default="mnist", help="Dataset to use")
args = parser.parse_args()
args.cuda = (args.cuda % 7)
# if args.cuda == 1:
#     args.cuda == 0
device = args.cuda
dataset = args.dataset
# args.experiment=26


def prepare_configs_loop(*all_param_grids):
    params = np.meshgrid(*all_param_grids, indexing="ij")
    params = [param.flatten() for param in params]
    return params


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class AddMask(object):
    def __init__(self, mask_size=0.5, random_mask=True):
        self.mask_size = mask_size
        self.random_mask = random_mask

    def __call__(self, tensor):
        mask = torch.ones(tensor.size())
        if self.random_mask:
            random_x_start = torch.randint(
                0, int(tensor.size()[1] * self.mask_size), size=(1,)
            ).item()
            random_y_start = torch.randint(
                0, int(tensor.size()[2] * self.mask_size), size=(1,)
            ).item()
            mask[
                :,
                random_x_start : random_x_start
                + int(self.mask_size * tensor.size()[1]),
                random_y_start : random_y_start
                + int(self.mask_size * tensor.size()[2]),
            ] = 0
        else:
            mask[
                :,
                : int(self.mask_size * tensor.size()[1]),
                : int(self.mask_size * tensor.size()[2]),
            ] = 0
        return tensor * mask

    def __repr__(self):
        return self.__class__.__name__ + "(mask_size={0})".format(self.mask_size)


def get_dataloaders(dataset):
    if dataset == "mnist":
        train_data = datasets.MNIST(
            root="data", train=True, download=True, transform=transforms.ToTensor()
        )

        test_data = datasets.MNIST(
            root="data", train=False, download=True, transform=transforms.ToTensor()
        )
        test_data_noisy = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    AddGaussianNoise(0.0, 1.0),
                ]
            ),
        )
        test_data_masked = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), AddMask(0.5, True)]),
        )
        test_data_left_masked = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), AddMask(0.5, False)]),
        )
        # test_data_elastic = datasets.MNIST(
        #     root="data",
        #     train=False,
        #     download=True,
        #     transform=transforms.Compose(
        #         [
        #             transforms.ToTensor(),
        #             transforms.Normalize((0.1307,), (0.3081,)),
        #             transforms.ElasticTransform(10, 10),
        #         ]
        #     ),
        # )
        test_data_gaussian_blur = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    transforms.GaussianBlur(3, 1),
                ]
            ),
        )

    elif dataset == "cifar10":
        train_data = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]
            ),
        )
        test_data = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]
            ),
        )

        test_data_noisy = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                    AddGaussianNoise(0.0, 1.0),
                ]
            ),
        )

        test_data_masked = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                    AddMask(0.5, True),
                ]
            ),
        )
        test_data_left_masked = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), AddMask(0.5, False)]),
        )
        # test_data_elastic = datasets.CIFAR10(
        #     root="data",
        #     train=False,
        #     download=True,
        #     transform=transforms.Compose(
        #         [
        #             transforms.ToTensor(),
        #             transforms.Normalize(mean, std),
        #             transforms.ElasticTransform(10, 10),
        #         ]
        #     ),
        # )
        test_data_gaussian_blur = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                    # transforms.G ,
                    transforms.GaussianBlur(3, 1),
                ]
            ),
        )

    return {
        "train": torch.utils.data.DataLoader(
            train_data,
            batch_size=512,
            shuffle=True,
        ),
        "test": torch.utils.data.DataLoader(
            test_data,
            batch_size=512,
            shuffle=True,
        ),
        "test_data_noisy": torch.utils.data.DataLoader(
            test_data_noisy,
            batch_size=512,
            shuffle=True,
        ),
        "test_data_masked": torch.utils.data.DataLoader(
            test_data_masked,
            batch_size=512,
            shuffle=True,
        ),
        "test_data_left_masked": torch.utils.data.DataLoader(
            test_data_left_masked,
            batch_size=512,
            shuffle=True,
        ),
        # "test_data_elastic": torch.utils.data.DataLoader(
        #     test_data_elastic,
        #     batch_size=512,
        #     shuffle=True,
        # ),
        "test_data_gaussian_blur": torch.utils.data.DataLoader(
            test_data_gaussian_blur,
            batch_size=512,
            shuffle=True,
        ),
    }


def train(num_epochs, cnn, loaders, device):
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)
    cnn.train()

    # Train the model
    total_step = len(loaders["train"])
    losses = []

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders["train"]):

            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images).to(device)  # batch x
            b_y = Variable(labels).to(device)  # batch y
            # b_x = images
            # b_y = labels
            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)

            # clear gradients for this training step
            optimizer.zero_grad()

            # backpropagation, compute gradients
            loss.backward()
            # apply gradients
            optimizer.step()
            losses.append(loss.item())

            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item()
                    )
                )

    return losses


def test(loaders, label, device):
    # Test the model
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders[label]:
            images = images.to(device)
            labels = labels.to(device)
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            correct += (pred_y == labels).sum().item()
            total += labels.size(0)
        accuracy = correct / float(total)
        print("Test Accuracy of the model on the 10000 test images: %.4f" % accuracy)
        return accuracy


num_layers = np.arange(2, 5)
num_recurrences = np.arange(3, 4)
num_channels = [8, 16, 24]
seeds = [0, 10, 100, 1000, 10000, 1000000]

params = prepare_configs_loop(num_layers, num_recurrences, num_channels, seeds)
print(len(params[0]))
print(params)
# for num_layers in range(2, 5):
#     for num_recurrence in range(3):
#         for num_channels in [8, 16, 24]:
# settings = [[3, 2, 16],

num_layer, num_recurrence, num_channel, seed = [
    params[i][args.experiment] for i in range(len(params))
]
random.seed(seed)
torch.manual_seed(seed)
cnn = CNN(
    num_layers=num_layer,
    num_recurrence=num_recurrence,
    num_channels=num_channel,
    dataset=dataset,
).to(device)
loaders = get_dataloaders(dataset)

if not os.path.exists(
    f"./models/{dataset}_seed_{seed}_model_num_layers={num_layer}_num_recurrence={num_recurrence}_num_channels={num_channel}.pkl"
):
    print("model does not exist")
    loss = train(3, cnn, loaders, device)
    test_acc = test(loaders, "test", device)
    noisy_test_acc = test(loaders, "test_data_noisy", device)
    masked_test_acc = test(loaders, "test_data_masked", device)
    left_masked_test_acc = test(loaders, "test_data_left_masked", device)
    gaussian_blur_test_acc = test(loaders, "test_data_gaussian_blur", device)
    data = {
        "cnn": cnn,
        "loss": loss,
        "test_acc": test_acc,
        "test_noisy_acc": noisy_test_acc,
        "test_masked_acc": masked_test_acc,
        "test_left_masked_acc": left_masked_test_acc,
        "test_gaussian_blur_acc": gaussian_blur_test_acc,
    }
else:
    with open(
        f"./models/{dataset}_seed_{seed}_model_num_layers={num_layer}_num_recurrence={num_recurrence}_num_channels={num_channel}.pkl",
        "rb",
    ) as f:
        data = pickle.load(f)
    if "test_left_masked_acc" in data.keys():
        print("model already tested")
        exit()
    cnn = data["cnn"].to(device)
    left_masked_test_acc = test(loaders, "test_data_left_masked", device)
    gaussian_blur_test_acc = test(loaders, "test_data_gaussian_blur", device)
    data["test_left_masked_acc"] = left_masked_test_acc
    # data["test_elastic_acc"] = elastic_test_acc
    data["test_gaussian_blur_acc"] = gaussian_blur_test_acc


with open(
    f"./models/{dataset}_seed_{seed}_model_num_layers={num_layer}_num_recurrence={num_recurrence}_num_channels={num_channel}.pkl",
    "wb",
) as f:
    pickle.dump(
        data,
        f,
    )
