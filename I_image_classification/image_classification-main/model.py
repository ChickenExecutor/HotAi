from torch import nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # TO DO: initialize model layers and components, store as instance variables for use in forward()
        # Hint: Make sure your layer's parameters are registered with your module. For instance, if you initialize layer's and store them in an array rather than directly as member variable of your class, you may need to register them using self.register_module([name], [layer])
        # YOUR CODE HERE

    def forward(self, x):
        # TO DO: implement forward pass through model
        # YOUR CODE HERE
