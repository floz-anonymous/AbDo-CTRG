import os
from pprint import pprint
from configs.config import parser
from dataset.data_module import DataModule
from lightning_tools.callbacks import add_callbacks
from models.AbDoRG_classifier import MambaCTClassifierGen
from lightning.pytorch import seed_everything
import lightning.pytorch as pl

def train(args):
    dm = DataModule(args)
    callbacks = add_callbacks(args)

    trainer = pl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        accelerator=args.accelerator,
        precision=args.precision,
        val_check_interval = args.val_check_interval,
        limit_val_batches = args.limit_val_batches,
        max_epochs = args.max_epochs,
        num_sanity_val_steps = args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks["callbacks"], 
        logger=callbacks["loggers"]
    )

    # Load label columns from the processed CSV
    import pandas as pd
    csv_path = getattr(args, 'ct_processed_csv', None)
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Exclude metadata columns to find label columns
        metadata_cols = ['id', 'Path', 'findings', 'split']
        args.condition_names = [c for c in df.columns if c not in metadata_cols]
        args.num_classes = len(args.condition_names)
        print(f"Loaded {args.num_classes} classes from {csv_path}")
    else:
        print(f"Warning: {csv_path} not found. Model will use default classes.")

    model = MambaCTClassifierGen(args)

    if args.test:
        trainer.test(model, datamodule=dm)
    elif args.validate:
        trainer.validate(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)

def main():
    args = parser.parse_args()
    
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    seed_everything(42, workers=True)
    train(args)

if __name__ == '__main__':
    main()
