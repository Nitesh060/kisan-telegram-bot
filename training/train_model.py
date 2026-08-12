#!/usr/bin/env python3
"""
Train crop disease detection model using transfer learning.
Uses MobileNetV3Small for lightweight inference.

Important preprocessing rule:
MobileNetV3Small is created with include_preprocessing=True (the default),
so training/validation/test generators must NOT rescale images to [0, 1].
The model receives images in the normal [0, 255] pixel range and performs
its own MobileNetV3 preprocessing internally.
"""

import os
import json
import argparse
import logging
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_model(num_classes: int, input_shape: tuple = (224, 224, 3)) -> keras.Model:
    """Create a transfer-learning model using MobileNetV3Small."""
    logger.info(f"🏗️ Creating model with {num_classes} classes...")

    # MobileNetV3 preprocessing is kept INSIDE the pretrained model.
    # Therefore inputs must remain in the [0, 255] pixel range.
    base_model = keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        include_preprocessing=True,
    )

    # Freeze base model layers (transfer learning)
    base_model.trainable = False

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
    )

    logger.info("✅ Model created successfully")
    return model


def create_data_generators(
    augmentation: bool = True,
    target_size: tuple = (224, 224),
) -> tuple:
    """
    Create data generators.

    Do NOT use rescale=1./255 here. MobileNetV3 performs its own
    preprocessing inside the model and expects raw [0, 255] pixels.
    """
    if augmentation:
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
        )
    else:
        train_datagen = ImageDataGenerator()

    validation_datagen = ImageDataGenerator()

    return train_datagen, validation_datagen


def train_model(
    model: keras.Model,
    train_dir: str,
    val_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    target_size: tuple = (224, 224),
) -> dict:
    """Train the model."""
    logger.info("📊 Setting up data generators...")

    train_datagen, val_datagen = create_data_generators(
        augmentation=True,
        target_size=target_size,
    )

    logger.info(f"Loading training data from {train_dir}...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    logger.info(f"Loading validation data from {val_dir}...")
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    class_indices = train_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}

    logger.info(f"📚 Found {len(class_names)} classes")
    for idx, name in sorted(class_names.items()):
        logger.info(f"   {idx}: {name}")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
        ),
    ]

    logger.info(f"🚀 Starting training for {epochs} epochs...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
    )

    return history, class_names


def evaluate_model(
    model: keras.Model,
    test_dir: str,
    target_size: tuple = (224, 224),
) -> dict:
    """Evaluate model on test data."""
    logger.info(f"📊 Evaluating model on {test_dir}...")

    # Keep test inputs in [0, 255] for MobileNetV3 internal preprocessing.
    test_datagen = ImageDataGenerator()

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
    )

    eval_results = model.evaluate(test_generator)

    results = {
        "test_loss": eval_results[0],
        "test_accuracy": eval_results[1],
        "test_precision": eval_results[2] if len(eval_results) > 2 else None,
        "test_recall": eval_results[3] if len(eval_results) > 3 else None,
    }

    logger.info(f"✅ Test Accuracy: {results['test_accuracy']:.4f}")
    logger.info(f"   Test Loss: {results['test_loss']:.4f}")

    return results


def save_model(
    model: keras.Model,
    model_path: str,
    class_names: dict,
    class_names_path: str,
):
    """Save trained model and class mapping."""
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    logger.info(f"💾 Saving model to {model_path}...")
    model.save(model_path)

    logger.info(f"💾 Saving class names to {class_names_path}...")
    class_names_dir = os.path.dirname(class_names_path)
    if class_names_dir:
        os.makedirs(class_names_dir, exist_ok=True)
    with open(class_names_path, "w", encoding="utf-8") as f:
        # Save as an ordered list so inference maps output index -> class
        # unambiguously.
        ordered_names = [class_names[i] for i in sorted(class_names)]
        json.dump(ordered_names, f, indent=2, ensure_ascii=False)

    logger.info("✅ Model and class names saved successfully")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train crop disease detection model")
    parser.add_argument(
        "--train-dir",
        type=str,
        default="data/training",
        help="Path to training data directory",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default="data/validation",
        help="Path to validation data directory",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="data/test",
        help="Path to test data directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/crop_disease_model.keras",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--class-names-path",
        type=str,
        default="models/class_names.json",
        help="Path to save class names mapping",
    )

    args = parser.parse_args()

    if not os.path.exists(args.train_dir):
        logger.error(f"❌ Training directory not found: {args.train_dir}")
        logger.info("\nExpected directory structure:")
        logger.info("data/training/")
        logger.info("  ├── Tomato_Early_Blight/")
        logger.info("  ├── Tomato_Late_Blight/")
        logger.info("  └── ...")
        return

    logger.info("🌾 Kisan Crop Disease Model Training")
    logger.info("=" * 60)

    num_classes = len([
        d for d in os.listdir(args.train_dir)
        if os.path.isdir(os.path.join(args.train_dir, d))
    ])
    if num_classes == 0:
        logger.error("❌ No disease classes found in training directory")
        return

    model = create_model(num_classes)
    logger.info("\nModel architecture:")
    model.summary()

    logger.info("\n" + "=" * 60)
    history, class_names = train_model(
        model,
        args.train_dir,
        args.val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    if os.path.exists(args.test_dir):
        logger.info("\n" + "=" * 60)
        evaluate_model(model, args.test_dir)
    else:
        logger.warning(f"⚠️ Test directory not found: {args.test_dir}")

    logger.info("\n" + "=" * 60)
    save_model(model, args.model_path, class_names, args.class_names_path)

    logger.info("\n" + "=" * 60)
    logger.info("✅ Training complete!")
    logger.info(f"Model saved to: {args.model_path}")
    logger.info(f"Class names saved to: {args.class_names_path}")


if __name__ == "__main__":
    main()
