import torch.nn as nn
from template_nn import CNN


def create_cnn_model():
    # Example of creating a Convolutional Neural Network (CNN)
    # This CNN is configured for image input (e.g., 3 channels, 64x64 pixels).
    # It includes two convolutional layers and two fully connected layers.
    cnn_model = CNN({
        "image_size": (64, 64),
        "conv_channels":
        [3, 6, 16],  # Input channels, then output channels for each conv layer
        "conv_kernel_size": 3,
        "pool_kernel_size": 2,
        "fcn_hidden_sizes": [120,
                             80],  # Hidden layers for the fully connected part
        "activation_functions":
        [nn.ReLU(), nn.ReLU()],  # Activations for conv and fc layers
        "output_channel": 10  # Output classes
    })
    print("CNN Model created successfully:")
    print(cnn_model)
    # Example of a training loop (simplified)
    # optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
    # criterion = nn.CrossEntropyLoss()
    # for epoch in range(num_epochs):
    #     outputs = cnn_model(inputs)
    #     loss = criterion(outputs, labels)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    return cnn_model


if __name__ == "__main__":
    create_cnn_model()
