import zipfile
import shutil
import os
from dotenv import load_dotenv
from pathlib import Path
import logging

class DataManager:
    def __init__(self, log_level=logging.INFO):

        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=log_level,
                            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        
        # Load environment variables from .env file
        load_dotenv()

        # Define data directory from environment variables
        self.data_dir = Path(os.getenv("PROJECT_DIR")) / os.getenv("DATA_DIR")
        self.train_dir = self.data_dir / "train"
        self.test_dir = self.data_dir / "test"
        self.logger.info(f"Data directory set to: {self.data_dir}")

    def extract_dataset(self, zip_filename="AI_Portfolio_Dataset.zip"):
        """Extracts the dataset from a ZIP file and organizes it into train and test folders."""
        
        zip_path = self.data_dir / zip_filename
        
        # Unzip dataset
        self.logger.info(f"Unzipping dataset from {zip_path}")

        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
        else:
            self.logger.error(f"ZIP file {zip_path} not found!")
            return
        
        # Move train and test folders to data directory
        for folder in ["train", "test"]:
            src = self.data_dir / "dataset" / folder
            dst = self.data_dir / folder

            if src.exists():
                if dst.exists():
                    self.logger.warning(f"Destination {dst} already exists. It will be overwritten.")
                    shutil.rmtree(dst)
                self.logger.info(f"Moving {src} to {dst}")
                shutil.move(src, dst)
            else:
                self.logger.error(f"Source {src} does not exist.")
        
        # Clean up by removing the extracted dataset directory
        dataset_dir = self.data_dir / "dataset"
        if dataset_dir.exists():
            self.logger.info(f"Removing temporary dataset directory {dataset_dir}")
            shutil.rmtree(dataset_dir)
        else:
            self.logger.warning(f"Dataset directory {dataset_dir} does not exist for cleanup.")

        self.logger.info("Dataset extraction complete.")

    def verify_dataset(self):
        """Verifies the integrity and structure of the dataset."""

        # TODO: Implement verification logic

    def get_train_images_list(self):
        """Helper function to get list of image file paths from the train directory."""
        images = list(self.train_dir.glob("*.jpg"))
        if not images:
            self.logger.warning(f"No images found in {self.train_dir}")
        return images
    
    def get_test_images_list(self):
        """Helper function to get list of image file paths from the test directory."""
        images = list(self.test_dir.glob("*.jpg"))
        if not images:
            self.logger.warning(f"No images found in {self.test_dir}")
        return images
    
    def get_class_from_filename(self, file_path):
        """Helper function to extract class from filename."""
        return file_path.stem.split("_")[1]
    
    def get_classes_list(self, strict=True):
        """Helper function to get list of unique classes from the training images."""

        train_images = self.get_train_images_list()
        test_images = self.get_test_images_list()

        # Extract classes from filenames
        train_set = {self.get_class_from_filename(p) for p in train_images}
        test_set = {self.get_class_from_filename(p) for p in test_images}

        # Check for consistency between train and test classes
        if train_set != test_set:
            log = f"Mismatch between training and testing classes. Training classes: {sorted(train_set)}, Testing classes: {sorted(test_set)}" 
            if strict:
                self.logger.error(log)
                raise ValueError("Training and testing classes do not match.")
            else:
                self.logger.warning(log)

        return sorted(train_set)

    def get_class_index_map(self):
        """Helper function to create a mapping from class names to indices."""
        classes = self.get_classes_list()
        class_to_index = {cls_name: idx for idx, cls_name in enumerate(classes)}
        return class_to_index