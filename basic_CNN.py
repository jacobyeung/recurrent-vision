import torch.nn as nn


class CNN(nn.Module):
    def __init__(
        self, num_layers, num_recurrence, num_channels, kernel_size=3, dataset="mnist"
    ):
        super(CNN, self).__init__()
        self.num_recurrence = num_recurrence
        if dataset == "mnist":
            in_channels = 1
        elif dataset == "cifar10":
            in_channels = 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_channels,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(num_channels, num_channels, kernel_size, 1, padding),
                    nn.ReLU(),
                )
            ]
            * num_layers
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(num_channels * 16 * 16, 10)

    def forward(self, x):
        x = self.conv1(x)
        for _ in range(self.num_recurrence):
            x = self.conv(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x  # return x for visualization
