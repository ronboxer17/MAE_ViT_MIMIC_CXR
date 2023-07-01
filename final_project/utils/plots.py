import matplotlib.pyplot as plt
import numpy as np
import torch


def display_image(image_tensor: torch.Tensor):
    # Convert tensor to numpy array
    image_np = image_tensor.numpy()

    # Check if the image tensor was on a CUDA device, and if so move it back to CPU
    if torch.cuda.is_available():
        image_np = image_tensor.cpu().numpy()

    # PyTorch tensors for images have the shape (C, H, W) so we need to transpose it to (H, W, C)
    image_np = np.transpose(image_np, (1, 2, 0))

    # # If the image's pixel values range from 0-255, normalize to 0-1
    # if np.max(image_np) > 1:
    #     image_np = image_np / 255

    # Display the image
    plt.imshow(image_np)
    plt.axis('off')  # Turn off axis numbers and labels
    plt.show()

# To use the function:
# display_image(your_image_tensor)
