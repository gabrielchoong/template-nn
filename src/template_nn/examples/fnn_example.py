import torch.nn as nn
from template_nn import FNN


def create_fnn_model():
    # Example of creating a Feedforward Neural Network (FNN)
    # This FNN has an input size of 10, output size of 2, and two hidden layers of sizes 64 and 32.
    # ReLU activation functions are used for both hidden layers.
    fnn_model = FNN({
        "input_size": 10,
        "output_size": 2,
        "hidden_sizes": [64, 32],
        "activation_functions": [nn.ReLU(), nn.ReLU()]
    })
    print("FNN Model created successfully:")
    print(fnn_model)
    # Example of a training loop (simplified)
    # optimizer = torch.optim.Adam(fnn_model.parameters(), lr=0.001)
    # criterion = nn.CrossEntropyLoss()
    # for epoch in range(num_epochs):
    #     outputs = fnn_model(inputs)
    #     loss = criterion(outputs, labels)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    return fnn_model


if __name__ == "__main__":
    create_fnn_model()
