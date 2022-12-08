import torch.nn as nn


class indRecurrenceCNN(nn.Module):
    def __init__(
        self, num_layers, num_recurrence, num_channels, kernel_size=3, dataset="mnist"
    ):
        super(indRecurrenceCNN, self).__init__()
        self.num_recurrence = num_recurrence
        padding = (kernel_size - 1) // 2

        self.conv_ind = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_channels, num_channels, kernel_size, 1, padding),
                nn.ReLU(),
            )
        ] * num_recurrence)

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
        xs = []
        x = self.conv1(x)
        x = self.conv(x)
        xs.append(x)
        for i in range(self.num_recurrence):
            xnew = self.conv_ind[i](xs[-1])
            xnew = self.conv(xnew)
            xnew = self.conv(xnew)
            xs.append(xnew+xs[-1])
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        xs[-1] = xs[-1].view(xs[-1].size(0), -1)
        output = self.out(xs[-1])
        return output, x  # return x for visualization
