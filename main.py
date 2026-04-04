import argparse
import torch
import numpy as np
from pathlib import Path

from config import ModelConfig, TrainingConfig
from data import DatasetBuilder, DataLoader, SyntheticDataGenerator
from models import BaseModel, MultiLayerEncoder
from training import Trainer, LossFunctions, OptimizerFactory
from evaluation import Evaluator
from utils import setup_logger, set_seed, create_output_dir

def main(args):
    """Main entry point."""
    
    # Setup
    set_seed(42)
    logger = setup_logger('dsm_eqa')
    output_dir = create_output_dir(args.output_dir)
    
    logger.info("=" * 50)
    logger.info("DSM-EQA: Data-Driven Structure Modeling with Equation Quality Assessment")
    logger.info("=" * 50)
    
    # Load configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    logger.info(f"Model Config: {model_config.to_dict()}")
    logger.info(f"Training Config: {training_config.to_dict()}")
    
    # Generate synthetic data
    logger.info("Generating synthetic data...")
    data_gen = SyntheticDataGenerator(seed=42)
    X, y = data_gen.generate_nonlinear_data(n_samples=1000, n_features=10)
    logger.info(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Build datasets
    logger.info("Building datasets...")
    dataset_builder = DatasetBuilder()
    datasets = dataset_builder.build_from_arrays(X, y, split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15})
    
    # Create data loaders
    logger.info("Creating data loaders...")
    data_loader = DataLoader(batch_size=training_config.batch_size, num_workers=training_config.num_workers)
    loaders = data_loader.create_loaders(datasets)
    
    # Create model
    logger.info("Creating model...")
    model = MultiLayerEncoder(
        input_dim=model_config.input_dim,
        hidden_dims=model_config.hidden_dims,
        output_dim=model_config.output_dim,
        dropout=model_config.encoder_dropout
    )
    logger.info(f"Model parameters: {model.count_parameters()}")
    
    # Setup training
    logger.info("Setting up training...")
    optimizer = OptimizerFactory.create_optimizer(
        model.parameters(),
        optimizer_type=training_config.optimizer,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )
    
    loss_fn = LossFunctions.get_loss_function(training_config.loss_function)
    
    trainer = Trainer(model, optimizer, loss_fn, device=training_config.device, logger=logger)
    
    # Train model
    logger.info("Starting training...")
    history = trainer.fit(
        loaders['train'],
        loaders['val'],
        epochs=training_config.epochs,
        early_stopping_patience=training_config.early_stopping_patience
    )
    
    # Evaluate
    logger.info("Evaluating model...")
    evaluator = Evaluator(model, device=training_config.device)
    metrics = evaluator.full_evaluation(loaders['test'])
    
    logger.info("Test Metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    logger.info("=" * 50)
    logger.info("Training completed!")
    logger.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSM-EQA Training Script")
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    main(args)
