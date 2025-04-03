import os
import torch
import torchvision.models as models
import torch.onnx
import onnx
import yaml
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load the trained ResNet model from .pt or .pth file.
    
    Args:
        model_path (str): Path to the model file
        device (str): Device to run the model on (cuda or cpu)
    
    Returns:
        model: Loaded PyTorch model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    if model_path.endswith('.pt'):
        try:
            model = torch.load(model_path, map_location=device, weights_only=False)
        except Exception as e:
            logger.error(f"Failed to load .pt file: {str(e)}")
            raise
    elif model_path.endswith('.pth'):
        model = models.resnet18()
        state_dict = torch.load(model_path, map_location=device)
        num_classes = state_dict['fc.weight'].shape[0]
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unsupported file format: {model_path}. Use .pt or .pth")
    
    model.to(device)
    model.eval()
    logger.info(f"Model loaded from {model_path} on {device}")
    return model

def export_to_onnx(model, tensor_input, onnx_path, opset_version=16, dynamic_axes=None):
    """
    Export the PyTorch model to ONNX format.
    
    Args:
        model: Loaded PyTorch model
        tensor_input: Dummy input tensor
        onnx_path (str): Path to save the ONNX file
        opset_version (int): ONNX opset version
        dynamic_axes (dict): Dynamic axes configuration
    """
    torch.onnx.export(
        model,
        tensor_input,
        onnx_path,
        verbose=True,
        input_names=["inputs"],
        output_names=["outputs"],
        opset_version=opset_version,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )
    
    # Kiểm tra file ONNX
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)
    logger.info(f"Model exported to ONNX at {onnx_path}")
    
    # In thông tin input của mô hình
    logger.info("Thông tin input của mô hình ONNX:")
    for input_info in model_onnx.graph.input:
        logger.info(str(input_info))

def main(config_path="config.yaml"):
    # Đọc config từ YAML
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    inference_config = config.get('inference', {})
    model_path = inference_config.get('model_path')
    onnx_path = inference_config.get('onnx_path', 'trained_resnet18.onnx')
    image_size = tuple(inference_config.get('image_size', [224, 224]))
    batch_size = inference_config.get('batch_size', 4)  # Batch size cho dummy input
    opset_version = inference_config.get('opset_version', 16)
    
    if not model_path:
        raise ValueError("Missing 'model_path' in inference config")
    
    # Thiết bị
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load mô hình
    model = load_model(model_path, device)
    
    # Tạo input tensor giả
    tensor_input = torch.randn(batch_size, 3, image_size[0], image_size[1]).to(device)
    
    # Cấu hình dynamic axes
    dynamic_axes = None
    if batch_size > 1:
        dynamic_axes = {
            "inputs": {0: "batch", 2: "height", 3: "width"},
            "outputs": {0: "batch", 1: "logits"},
        }
    
    # Export sang ONNX
    logger.info("Exporting model to ONNX...")
    export_to_onnx(model, tensor_input, onnx_path, opset_version, dynamic_axes)

if __name__ == "__main__":
    main(config_path="config/config.yaml")