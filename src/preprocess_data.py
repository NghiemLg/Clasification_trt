import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import yaml
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageDataset(Dataset):
    def __init__(self, data_dir, image_size, normalize=True, augment=False):
        """
        Args:
            data_dir (str): Thư mục chứa dữ liệu (train hoặc test)
            image_size (tuple): Kích thước ảnh đầu ra
            normalize (bool): Chuẩn hóa về [0, 1]
            augment (bool): Tăng cường dữ liệu
        """
        self.data_dir = data_dir
        self.image_size = image_size
        
        # Định nghĩa transform
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize(image_size),
        ]
        if augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(30),
            ])
        transform_list.append(transforms.ToTensor())  # Chuyển thành tensor, tự động [0, 1]
        if not normalize:
            transform_list.append(transforms.Normalize(mean=[0], std=[1]))  # Bỏ normalize nếu không cần
        self.transform = transforms.Compose(transform_list)
        
        # Lấy danh sách ảnh và nhãn
        self.image_paths = []
        self.labels = []
        class_names = sorted(os.listdir(data_dir))
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        for class_name in class_names:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if os.path.isfile(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
        
        logger.info(f"Loaded {len(self.image_paths)} images from {data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Tải ảnh khi cần
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot load image from {img_path}")
        
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Áp dụng transform
        if self.transform:
            image = self.transform(image)
        
        return image, label

def preprocess_data(config_path="config.yaml"):
    """
    Preprocess data using Dataset and DataLoader based on config.yaml.
    
    Args:
        config_path (str): Path to the YAML configuration file
    """
    # Đọc config từ file YAML
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    preprocess_config = config.get('preprocess_data', {})
    train_dir = preprocess_config.get('train_dir')
    test_dir = preprocess_config.get('test_dir')
    image_size = tuple(preprocess_config.get('image_size', [224, 224]))
    normalize = preprocess_config.get('normalize', True)
    augment = preprocess_config.get('augment', False)
    batch_size = preprocess_config.get('batch_size', 32)
    
    if not train_dir or not test_dir:
        raise ValueError("Missing 'train_dir' or 'test_dir' in preprocess_data config")
    
    # Tạo Dataset
    train_dataset = ImageDataset(train_dir, image_size, normalize, augment)
    test_dataset = ImageDataset(test_dir, image_size, normalize, augment=False)  # Không augment tập test
    
    # Tạo DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Ví dụ: Duyệt qua một batch để kiểm tra
    for images, labels in train_loader:
        logger.info(f"Train batch shape: {images.shape}, Labels: {labels.shape}")
        break
    
    for images, labels in test_loader:
        logger.info(f"Test batch shape: {images.shape}, Labels: {labels.shape}")
        break
    
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = preprocess_data(config_path="cconfig/config.yaml")