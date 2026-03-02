import numpy as np

def generate_gradcam(model, input_tensor, target_layer=None):
    """
    Dummy Grad-CAM generator.
    Returns a blank heatmap (all zeros) of the same spatial size as the input.
    """
    # Assume input_tensor is a 4D tensor (batch, channels, height, width)
    # Return a dummy heatmap with shape (height, width)
    if hasattr(input_tensor, 'shape') and len(input_tensor.shape) == 4:
        h, w = input_tensor.shape[2], input_tensor.shape[3]
    else:
        h, w = 224, 224  # default
    return np.zeros((h, w), dtype=np.float32)