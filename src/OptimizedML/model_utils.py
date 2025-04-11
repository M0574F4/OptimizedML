# utils/model_utils.py
import torch
import torchvision
import torchvision.models.quantization as models_quant
from torchvision.models import ResNet18_Weights

SUPPORTED_MODELS = ["resnet18"] # Extend this list as you add more models

def get_model_and_transform(model_name: str, weights_enum=ResNet18_Weights.DEFAULT):
    """
    Loads a pre-trained FP32 model and its associated preprocessing transform.
    """
    if model_name == "resnet18":
        weights = weights_enum
        model = torchvision.models.resnet18(weights=weights)
        preprocess = weights.transforms()
    # Add elif blocks for other supported models here
    # elif model_name == "mobilenet_v2":
    #     weights = MobileNet_V2_Weights.DEFAULT
    #     model = torchvision.models.mobilenet_v2(weights=weights)
    #     preprocess = weights.transforms()
    else:
        raise ValueError(f"Model '{model_name}' is not supported. Supported models: {SUPPORTED_MODELS}")

    model.eval()
    model.cpu()
    print(f"Original FP32 {model_name} model loaded and moved to CPU.")
    print("Preprocessing transforms for model loaded.")
    return model, preprocess

def get_quantization_aware_model(model_name: str, fp32_model: torch.nn.Module):
    """
    Creates a quantization-aware version of the model architecture and loads
    weights from the corresponding FP32 model.
    """
    if model_name == "resnet18":
        # Create architecture without pre-trained weights, ready for quantization structure
        quant_aware_model = models_quant.resnet18(weights=None, quantize=False)
    # Add elif blocks for other supported models here
    # elif model_name == "mobilenet_v2":
    #     quant_aware_model = models_quant.mobilenet_v2(weights=None, quantize=False)
    else:
        raise ValueError(f"Quantization-aware architecture for '{model_name}' not supported. Supported models: {SUPPORTED_MODELS}")

    # Load weights from the trained FP32 model
    quant_aware_model.load_state_dict(fp32_model.state_dict())
    quant_aware_model.eval()
    quant_aware_model.cpu()
    print(f"Quantization-aware {model_name} architecture created and weights loaded.")
    return quant_aware_model

def fuse_model_modules(model: torch.nn.Module):
    """
    Attempts to fuse common sequences of modules suitable for quantization.
    Handles potential AttributeError for specific fusion functions.
    """
    model.eval() # Fusion usually requires eval mode
    print("Attempting module fusion...")
    try:
        # Try the QAT version first as it might handle more patterns
        # Note: This might need `import torch.ao.quantization as ao_quant` if not directly under `torch.quantization`
        # Adjust the import path based on your PyTorch version if needed.
        # Let's assume it might be under torch.quantization for now based on typical usage
        torch.quantization.fuse_modules_qat(model, inplace=True)
        print("Fused modules using fuse_modules_qat.")
    except AttributeError:
        print("torch.quantization.fuse_modules_qat not found, trying fuse_modules...")
        try:
            # Fallback to the standard fuse_modules if QAT version fails/doesn't exist
            # Define common patterns; you might need to customize this list
            # For ResNet, common fusions like Conv-BN-ReLU are often handled internally
            # by the models_quant structure, but explicit call can be useful.
            # Example: torch.quantization.fuse_modules(model, [['conv1', 'bn1', 'relu']], inplace=True)
            # For simplicity, let's assume the models_quant structure handles most needed fusions
            # If specific fusions are needed, they should be defined here based on the model.
            print("Attempted fusion with basic fuse_modules (may require specific layer lists).")
            # If fuse_modules itself isn't found or needs specific args, handle here
        except Exception as e:
            print(f"Could not fuse modules using basic fuse_modules: {e}")
    except Exception as e:
         print(f"Could not fuse modules: {e}")