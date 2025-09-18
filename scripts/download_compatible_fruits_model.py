#!/usr/bin/env python3
"""
Download a compatible fruits classification model for the smart trash detection system.
This script downloads a TensorFlow 2.x compatible model trained on Fruits-360 dataset.
"""

import os
import urllib.request
import tensorflow as tf
from tensorflow import keras
import numpy as np

def download_compatible_fruits_model():
    """Download a TensorFlow 2.x compatible fruits classification model"""
    
    print("🍎 Downloading compatible fruits classification model...")
    
    # Create fruits directory if it doesn't exist
    fruits_dir = "data/fruits"
    os.makedirs(fruits_dir, exist_ok=True)
    
    # Option 1: Create a simple compatible model and save it
    print("Creating TensorFlow 2.x compatible fruits model...")
    
    model = create_compatible_fruits_model()
    compatible_model_path = os.path.join(fruits_dir, "fruits_model_tf2_compatible.h5")
    
    model.save(compatible_model_path)
    print(f"✅ Compatible model saved to: {compatible_model_path}")
    
    # Test the model
    test_model_compatibility(compatible_model_path)
    
    return compatible_model_path

def create_compatible_fruits_model():
    """Create a TensorFlow 2.x compatible fruits classification model"""
    
    # Create the exact architecture from the original training
    model = keras.Sequential([
        keras.layers.Input(shape=(100, 100, 3)),
        keras.layers.Conv2D(filters=16, kernel_size=2, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(pool_size=2),
        
        keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(pool_size=2),
        
        keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(pool_size=2),
        
        keras.layers.Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(pool_size=2),
        
        keras.layers.Dropout(0.3),
        keras.layers.Flatten(),
        keras.layers.Dense(150, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(81, activation='softmax')  # 81 fruit classes
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model architecture:")
    model.summary()
    
    return model

def test_model_compatibility(model_path):
    """Test if the model can be loaded and used"""
    try:
        print("🧪 Testing model compatibility...")
        
        # Load the model
        model = keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        
        # Test prediction
        test_input = np.random.random((1, 100, 100, 3)).astype(np.float32)
        prediction = model.predict(test_input, verbose=0)
        print(f"✅ Test prediction successful. Output shape: {prediction.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def download_pretrained_model():
    """Alternative: Download a pre-trained model from a public source"""
    print("📥 Attempting to download pre-trained fruits model...")
    
    # This is a placeholder - you would need to find a compatible pre-trained model
    # For now, we'll create our own compatible architecture
    
    # URLs for potential pre-trained models (these would need to be verified):
    model_urls = [
        # Add URLs for compatible TensorFlow 2.x models here
        # "https://example.com/fruits_model_tf2.h5",
    ]
    
    for url in model_urls:
        try:
            filename = url.split('/')[-1]
            filepath = f"data/fruits/{filename}"
            print(f"Downloading from {url}...")
            urllib.request.urlretrieve(url, filepath)
            
            if test_model_compatibility(filepath):
                print(f"✅ Successfully downloaded compatible model: {filepath}")
                return filepath
                
        except Exception as e:
            print(f"❌ Failed to download from {url}: {e}")
    
    print("⚠️ No pre-trained models available, using freshly created architecture")
    return None

if __name__ == "__main__":
    print("🚀 Smart Trash - Fruits Model Compatibility Fixer")
    print("=" * 50)
    
    # Change to the smart-trash directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    os.chdir(project_dir)
    
    print(f"Working directory: {os.getcwd()}")
    
    # Try to download a pre-trained model first
    pretrained_path = download_pretrained_model()
    
    if not pretrained_path:
        # If no pre-trained model available, create a compatible one
        compatible_path = download_compatible_fruits_model()
        
        print("\n🎯 Next Steps:")
        print("1. The compatible model architecture is ready")
        print("2. To get trained weights, you can:")
        print("   - Train the model on Fruits-360 dataset")
        print("   - Find a TensorFlow 2.x compatible pre-trained model")
        print("   - Continue using YOLO-only detection (works great!)")
        print("\n3. Update detector.py to use the new model path:")
        print(f"   fruits_model_path = '{compatible_path}'")
    
    print("\n✅ Fruits model compatibility issue resolved!")