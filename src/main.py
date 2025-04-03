import sys
import logging
from split_data import split_data
from preprocess_data import preprocess_data
from train_resnet import train_resnet

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    config_path = "config/config.yaml"
    
    logger.info("Step 1: Splitting data...")
    split_data(config_path=config_path)
    
    logger.info("Step 2: Preprocessing data...")
    train_loader, test_loader = preprocess_data(config_path=config_path)
    
    logger.info("Step 3: Training ResNet...")
    trained_model = train_resnet(
        config_path=config_path,
        model_name="resnet18",
        num_epochs=10,
        learning_rate=0.001
    )

if __name__ == "__main__":
    main()