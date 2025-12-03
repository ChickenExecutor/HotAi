from torch import nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # TO DO: initialize model layers and components, store as instance variables for use in forward()
        # Hint: Make sure your layer's parameters are registered with your module. For instance, if you initialize layer's and store them in an array rather than directly as member variable of your class, you may need to register them using self.register_module([name], [layer])
        # YOUR CODE HERE
        self.conv1 = nn.Conv2d(in_channels = 3, kernel_size= 11, stride = 4, out_channels=96)

        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride = 2)

        self.conv2 = nn.Conv2d(in_channels = 96, kernel_size= 5, padding=2, out_channels=256)

        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride = 2)

        self.conv3 = nn.Conv2d(in_channels = 256, kernel_size= 3, padding=1, out_channels=384)

        self.conv4 = nn.Conv2d(in_channels = 384, kernel_size= 3, padding=1, out_channels=384)

        self.conv5 = nn.Conv2d(in_channels = 384, kernel_size= 3, padding=1, out_channels=256)

        self.relu = nn.ReLU(inplace=True)


        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride = 2)


        self.fully_connected1 = nn.Linear(in_features=256, out_features=128)
        self.fully_connected2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # TO DO: implement forward pass through model
        # YOUR CODE HERe
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool1(x)
        x =  self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2(x)
        x =  self.conv3(x)
        x = self.relu(x)
        x =  self.conv4(x)
        x = self.relu(x)
        x =  self.conv5(x)
        x = self.relu(x)
        x = self.max_pool3(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fully_connected1(x)
        x = self.relu(x)
        x = self.fully_connected2(x)
        return x