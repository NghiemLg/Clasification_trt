import os
import torch
from torchvision import transforms
from PIL import Image
import yaml
import logging
import matplotlib.pyplot as plt
import numpy as np

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load the trained ResNet model from .pt file.
    
    Args:
        model_path (str): Path to the .pt file
        device (str): Device to run inference on (cuda or cpu)
    
    Returns:
        model: Loaded PyTorch model
    """
    # Load toàn bộ mô hình, không giới hạn weights_only
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    logger.info(f"Model loaded from {model_path} on {device}")
    return model

def preprocess_image(image_path, image_size=(224, 224)):
    """
    Preprocess a single image for inference.
    
    Args:
        image_path (str): Path to the image file
        image_size (tuple): Target image size (height, width)
    
    Returns:
        tensor: Preprocessed image tensor
        pil_image: Original PIL image for visualization
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    
    pil_image = Image.open(image_path).convert('RGB')
    image_tensor = transform(pil_image).unsqueeze(0)
    return image_tensor, pil_image

def inference(model, image_dir, class_names, image_size, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Perform inference on a directory of images with visualization.
    
    Args:
        model: Loaded PyTorch model
        image_dir (str): Directory containing images to predict
        class_names (list): List of class names
        image_size (tuple): Target image size
        device (str): Device to run inference on
    """
    model.eval()
    images_to_show = []
    predictions = []
    
    for image_name in os.listdir(image_dir):
        if not image_name.lower().endswith(('png', 'jpg', 'jpeg')):
            continue
        
        image_path = os.path.join(image_dir, image_name)
        try:
            image_tensor, pil_image = preprocess_image(image_path, image_size=image_size)
            image_tensor = image_tensor.to(device)
            
            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                predicted_class = class_names[predicted.item()]
            
            logger.info(f"Image: {image_name} | Predicted class: {predicted_class}")
            images_to_show.append(pil_image)
            predictions.append(predicted_class)
        
        except Exception as e:
            logger.warning(f"Error processing {image_name}: {str(e)}")
    
    # Visualization: Hiển thị tất cả ảnh trong một lưới
    num_images = len(images_to_show)
    cols = 3  # Số cột
    rows = (num_images + cols - 1) // cols  # Số hàng
    plt.figure(figsize=(cols * 4, rows * 4))
    
    for i, (img, pred) in enumerate(zip(images_to_show, predictions)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(np.array(img))
        plt.title(f"Predicted: {pred}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main(config_path="config.yaml"):
    # Đọc config từ YAML
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    inference_config = config.get('inference', {})
    model_path = inference_config.get('model_path')
    image_dir = inference_config.get('image_dir')
    image_size = tuple(inference_config.get('image_size', [224, 224]))
    
    if not model_path or not image_dir:
        raise ValueError("Missing 'model_path' or 'image_dir' in inference config")
    
    # Thiết bị
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Lấy class names từ train_dir
    preprocess_config = config.get('preprocess_data', {})
    train_dir = preprocess_config.get('train_dir')
    if not train_dir:
        raise ValueError("Missing 'train_dir' in preprocess_data config")
    class_names = sorted(os.listdir(train_dir))
    
    # Load mô hình
    model = load_model(model_path, device)
    
    # Inference với visualization
    logger.info("Starting inference with visualization...")
    inference(model, image_dir, class_names, image_size, device)

if __name__ == "__main__":
    # Điều chỉnh đường dẫn config nếu cần
    main(config_path="/media/nlg/D/2025/IVSRRRRRRRRRRRRRRRR/classification/config/config.yaml")