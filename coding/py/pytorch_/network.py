from torch import nn



class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

        self.linear_relu_stack2 = nn.Sequential(
            self.make_gen_block(),
            self.make_gen_block()
        )

    def make_gen_block(self):
        return nn.Sequential(
            nn.Linear(28*28, 512),
            # block structure
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == '__main__':
    model = NeuralNetwork()
    print(model)

    # Subclassing nn.Module automatically tracks all fields defined inside your model object, 
    # and makes all parameters accessible using your model's parameters() or named_parameters() methods.
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")