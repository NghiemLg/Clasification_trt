import os
import shutil
import random
import yaml
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def split_data(config_path="config.yaml"):
    """
    Split dataset into train and test sets based on config.yaml.
    
    Args:
        config_path (str): Path to the YAML configuration file
    """
    # Đọc config từ file YAML
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    split_config = config.get('split_data', {})
    root_dir = split_config.get('root_dir')
    output_dir = split_config.get('output_dir')
    train_ratio = split_config.get('train_ratio', 0.8)
    
    if not root_dir or not output_dir:
        raise ValueError("Missing 'root_dir' or 'output_dir' in split_data config")
    
    # Tạo thư mục đầu ra
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Duyệt qua từng category
    for category in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category)
        if not os.path.isdir(category_path):
            continue
        
        # Lấy danh sách ảnh
        images = [f for f in os.listdir(category_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        random.shuffle(images)
        
        # Chia train/test
        train_count = int(len(images) * train_ratio)
        train_images = images[:train_count]
        test_images = images[train_count:]
        
        # Tạo thư mục cho category trong train/test
        train_category_path = os.path.join(train_dir, category)
        test_category_path = os.path.join(test_dir, category)
        os.makedirs(train_category_path, exist_ok=True)
        os.makedirs(test_category_path, exist_ok=True)
        
        # Sao chép ảnh
        for img in train_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(train_category_path, img))
        
        for img in test_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(test_category_path, img))
        
        logger.info(f"Category '{category}': {train_count} train, {len(test_images)} test images")
    
    logger.info("Dataset splitting completed!")

if __name__ == "__main__":
    split_data(config_path="cconfig/config.yaml")

