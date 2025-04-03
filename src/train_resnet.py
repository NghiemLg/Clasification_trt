import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import yaml
import logging
import sys
from preprocess_data import preprocess_data
import os

# Thiết lập logging không buffer
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def train_resnet(config_path="config.yaml", model_name="resnet18", num_epochs=10, learning_rate=0.001):
    # Đọc config từ file YAML
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    preprocess_config = config.get('preprocess_data', {})
    num_classes = len(os.listdir(preprocess_config.get('train_dir')))
    
    # Thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load dữ liệu
    train_loader, test_loader = preprocess_data(config_path=config_path)
    
    # Load mô hình ResNet
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # Cập nhật để tránh warning
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    
    # Loss và optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Huấn luyện
    logger.info(f"Starting training with {model_name} for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # In log mỗi 100 batch
            if (i + 1) % 100 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)}, "
                            f"Loss: {running_loss / (i + 1):.4f}, Accuracy: {100 * correct / total:.2f}%")
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
    # Đánh giá trên test set
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = 100 * test_correct / test_total
    logger.info(f"Test Accuracy: {test_acc:.2f}%")
    
    # Lưu mô hình
    # Trong hàm train_resnet
    save_path = f"trained_{model_name}.pt"  # Đổi đuôi thành .pt
    torch.save(model, save_path)  # Lưu toàn bộ mô hình thay vì state_dict
    logger.info(f"Model saved to {save_path}")
        
    return model

if __name__ == "__main__":
    train_resnet(config_path="config/config.yaml", model_name="resnet18", num_epochs=10, learning_rate=0.001)