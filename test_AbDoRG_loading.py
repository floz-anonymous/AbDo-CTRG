import sys
import os
import torch
import argparse
from dataset.data_module import DataModule

# Mock args
class MockArgs:
    def __init__(self):
        self.dataset = 'ct'
        self.image_preprocessor = 'microsoft/swin-base-patch4-window7-224'
        self.vision_model = 'None'
        self.input_size = 224
        self.batch_size = 2
        self.val_batch_size = 2
        self.test_batch_size = 2
        self.num_workers = 0
        self.prefetch_factor = None # Must be None if num_workers 0
        self.base_dir = ''
        self.ct_processed_csv = 'dataset/ct_processed_classes.csv'
        self.ct_reports_dir = 'dataset/reports'
        self.ct_images_dir = 'dataset/ct_dataset'

def test_data_loading():
    print("Testing data loading...")
    
    # Check if CSV exists
    if not os.path.exists('dataset/ct_processed_classes.csv'):
        print("CSV not found. Please run dataset/create_ct_processed_classes.py first.")
        return

    args = MockArgs()

    # Mock args for dynamic labels
    import pandas as pd
    if os.path.exists('dataset/ct_processed_classes.csv'):
        df = pd.read_csv('dataset/ct_processed_classes.csv')
        metadata_cols = ['id', 'Path', 'findings', 'split']
        args.condition_names = [c for c in df.columns if c not in metadata_cols]
        args.num_classes = len(args.condition_names)
        print(f"Test loaded {args.num_classes} classes: {args.condition_names[:5]}...")
    else:
        args.condition_names = None

    dm = DataModule(args)
    dm.setup('fit')
    
    print(f"Train set size: {len(dm.dataset['train'])}")
    print(f"Val set size: {len(dm.dataset['validation'])}")
    
    # Get a batch
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    
    print("Batch keys:", batch.keys())
    
    images = batch['image']
    print(f"Image object type: {type(images)}")
    if isinstance(images, list):
        print(f"Image list length: {len(images)}")
        if len(images) > 0:
            print(f"First element type: {type(images[0])}")
            if isinstance(images[0], torch.Tensor):
                 print(f"First image shape: {images[0].shape}")
            elif isinstance(images[0], list):
                 print(f"First element is list of length {len(images[0])}")
                 print(f"First image shape: {images[0][0].shape}")
    elif isinstance(images, torch.Tensor):
        print(f"Image tensor shape: {images.shape}")
    
    # Check ID and Text
    print("IDs:", batch['id'])
    print("Text:", batch['input_text'])
    
    if 'disease_labels' in batch:
        print("Labels shape:", batch['disease_labels'].shape)
        print("Labels:", batch['disease_labels'])

if __name__ == '__main__':
    test_data_loading()
