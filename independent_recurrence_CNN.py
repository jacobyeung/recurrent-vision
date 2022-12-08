import torch.nn as nn


class indRecurrenceCNN(nn.Module):
    def __init__(
        self, num_layers, num_recurrence, num_channels, kernel_size=3, dataset="mnist"
    ):
        super(indRecurrenceCNN, self).__init__()
        self.num_recurrence = num_recurrence
        padding = (kernel_size - 1) // 2

        self.conv_ind = [
            nn.Sequential(
                nn.Conv2d(num_channels, num_channels, kernel_size, 1, padding),
                nn.ReLU(),
            )
        ] * num_recurrence

        if dataset == "mnist":
            in_channels = 1
            self.out = nn.Linear(num_channels * 14 * 14, 10)
        elif dataset == "cifar10":
            in_channels = 3
            self.out = nn.Linear(num_channels * 16 * 16, 10)
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv(x)
        for i in range(self.num_recurrence):
            x = self.conv_ind[i](x)
            x = self.conv(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x  # return x for visualization
