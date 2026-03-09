import os
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
from transformers import AutoImageProcessor

class FieldParser:
    def __init__(self, args):
        self.args = args
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.image_preprocessor)

    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt", size=self.args.input_size).pixel_values
        return pixel_values[0]

    def clean_report(self, report):
        if not isinstance(report, str):
            return ""
        # Basic cleaning
        report = report.strip().lower()
        return report

    def parse(self, features):
        to_return = {'id': features['id']}
        report = features.get("findings", "")
        report = self.clean_report(report)
        to_return['input_text'] = report

        # Image processing
        # Handle path that might be absolute or relative
        image_path = features['Path']
        if not os.path.isabs(image_path):
             # Try to construct full path if base_dir is set, otherwise assume relative to CWD
             if self.args.base_dir:
                 image_path = os.path.join(self.args.base_dir, image_path)
        
        images = []
        try:
            with Image.open(image_path) as pil:
                array = np.array(pil, dtype=np.uint8)
                if array.shape[-1] != 3 or len(array.shape) != 3:
                     array = np.array(pil.convert("RGB"), dtype=np.uint8)
                image = self._parse_image(array)
                images.append(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            images.append(torch.zeros((3, self.args.input_size, self.args.input_size)))

        to_return["image"] = images
        
        return to_return

    def transform_with_parse(self, inputs):
        return self.parse(inputs)

class CTDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.parser = FieldParser(args)
        
        # Load CSV
        csv_path = args.ct_processed_csv
        if not os.path.exists(csv_path):
             raise FileNotFoundError(f"Processed CT CSV not found at {csv_path}. Please run dataset/create_ct_processed_classes.py first.")
             
        df = pd.read_csv(csv_path)
        
        # Filter by split
        self.df = df[df['split'] == split].reset_index(drop=True)
        print(f"Loaded {len(self.df)} records for split '{split}'")

        # Identify label columns
        metadata_cols = ['id', 'Path', 'findings', 'split']
        self.label_cols = [c for c in df.columns if c not in metadata_cols]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        features = row.to_dict()
        
        parsed = self.parser.transform_with_parse(features)
        
        # Add labels tensor
        labels = [row[col] for col in self.label_cols]
        parsed['disease_labels'] = torch.tensor(labels, dtype=torch.float)
        
        return parsed

def create_datasets(args):
    train_dataset = CTDataset(args, 'train')
    val_dataset = CTDataset(args, 'val')
    test_dataset = CTDataset(args, 'test')
    return train_dataset, val_dataset, test_dataset
