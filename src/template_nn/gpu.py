import torch


def get_gpu_device() -> torch.device:
    """
    Get the GPU device.
    Training on CPU is just not feasible currently.
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")

        return device

    if torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon) for GPU acceleration.")

        return device

    # cuda or mps not present
    device = torch.device("cpu")

    # if this line is printed but gpu device is present, there's something wrong
    print("Using CPU for training.")

    # warn if no gpu device is detected
    if not torch.cuda.is_available() and not torch.backends.mps.is_built():
        print("Warning: Neither CUDA nor MPS are available. Falling back to CPU.")

    return device
