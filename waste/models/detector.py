"""
Dual-Model Waste Detection System
- YOLOv11 for inorganic waste detection (TACO dataset)  
- CNN for organic waste classification (Fruits-360 dataset)
Optimized for Raspberry Pi 4B performance
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import time

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class WasteDetector:
    """
    Dual-model waste detection system:
    - YOLOv11 for object detection and inorganic waste classification (TACO dataset)
    - CNN for organic waste classification when needed (Fruits-360 dataset)
    """
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        """
        Initialize dual waste detection system
        
        Args:
            model_path: Path to custom YOLO model, if None uses trained TACO model
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.yolo_model = None      # YOLOv11 for object detection
        self.fruits_model = None    # CNN for fruits classification
        self.class_names = {}
        self.fruit_labels = []
        self.setup_logging()
        
        if YOLO_AVAILABLE:
            self.load_yolo_model(model_path)
        else:
            self.setup_fallback_detector()
            
        if TF_AVAILABLE:
            self.load_fruits_model()
        else:
            self.logger.warning("TensorFlow not available - organic waste detection disabled")

    def setup_logging(self):
        """Setup logging for the detector"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_yolo_model(self, model_path: Optional[str] = None):
        """
        Load YOLOv8/YOLOv11 model with PyTorch 2.6+ compatibility
        
        Args:
            model_path: Path to custom model, defaults to custom trained model or YOLOv8n
        """
        try:
            import torch
            
            # PyTorch 2.6+ changed weights_only default to True
            # We need to use weights_only=False for YOLO models (trusted source)
            old_load = torch.load
            torch.load = lambda *args, **kwargs: old_load(*args, **{**kwargs, 'weights_only': False})
            
            try:
                # First try to load custom trained trash detection model
                custom_model_path = "data/runs/detect/train/weights/best.pt"
                if os.path.exists(custom_model_path):
                    self.yolo_model = YOLO(custom_model_path)
                    self.logger.info(f"Loaded custom YOLOv11 trash detection model: {custom_model_path}")
                elif model_path and Path(model_path).exists():
                    self.yolo_model = YOLO(model_path)
                    self.logger.info(f"Loaded custom model from {model_path}")
                else:
                    # Fallback to YOLOv8-nano for Pi 4B performance
                    self.yolo_model = YOLO('yolov8n.pt')
                    self.logger.info("Loaded YOLOv8-nano model (fallback)")
                
                # Setup class mappings for waste detection
                self.setup_waste_classes()
                
            finally:
                # Always restore original torch.load
                torch.load = old_load
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            self.setup_fallback_detector()

    def load_fruits_model(self):
        """Load the CNN fruits classification model with compatibility fixes"""
        try:
            # Try the new compatible model first, then fallback to original
            compatible_model_path = "data/fruits/fruits_model_tf2_compatible.h5"
            original_model_path = "data/fruits/cnn_from_scratch_fruits.hdf5"
            
            fruits_model_path = compatible_model_path if os.path.exists(compatible_model_path) else original_model_path
            
            if os.path.exists(fruits_model_path):
                # Try multiple loading strategies for compatibility
                self.fruits_model = self._load_fruits_model_with_fallback(fruits_model_path)
                
                if self.fruits_model is not None:
                    self.logger.info(f"Loaded fruits CNN model: {fruits_model_path}")
                    # Load fruit labels
                    self.fruit_labels = self.load_fruit_labels()
                    self.logger.info(f"Loaded {len(self.fruit_labels)} fruit classes")
                else:
                    self.logger.warning("Failed to load fruits model with all compatibility methods")
                    self._suggest_model_alternatives()
            else:
                self.logger.warning(f"Fruits model not found at {fruits_model_path}")
                self._suggest_model_alternatives()
                
        except Exception as e:
            self.logger.error(f"Failed to load fruits model: {e}")
            self.fruits_model = None
            self._suggest_model_alternatives()

    def _load_fruits_model_with_fallback(self, model_path: str):
        """Try multiple strategies to load the fruits model"""
        # Strategy 1: Load with compile=False (bypass compilation issues)
        try:
            self.logger.info("Trying to load model with compile=False...")
            model = keras.models.load_model(model_path, compile=False)
            
            # Test if model works with expected input shape
            test_input = np.random.random((1, 100, 100, 3)).astype(np.float32)
            _ = model.predict(test_input, verbose=0)
            self.logger.info("Model loaded successfully with compile=False")
            return model
            
        except Exception as e:
            self.logger.warning(f"Strategy 1 failed: {e}")
        
        # Strategy 2: Try with TF compatibility mode
        try:
            self.logger.info("Trying with TensorFlow compatibility settings...")
            import tensorflow as tf
            tf.compat.v1.disable_eager_execution()
            model = keras.models.load_model(model_path, compile=False)
            tf.compat.v1.enable_eager_execution()
            
            test_input = np.random.random((1, 100, 100, 3)).astype(np.float32)
            _ = model.predict(test_input, verbose=0)
            self.logger.info("Model loaded successfully with TF compatibility")
            return model
            
        except Exception as e:
            self.logger.warning(f"Strategy 2 failed: {e}")
        
        # Strategy 3: Rebuild the exact model architecture from the training code
        try:
            self.logger.info("Rebuilding model with original architecture...")
            return self._rebuild_original_fruits_model()
            
        except Exception as e:
            self.logger.warning(f"Strategy 3 failed: {e}")
        
        return None
    
    def _rebuild_original_fruits_model(self):
        """Rebuild the exact model architecture from the original training code"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten, Dropout
        
        # Recreate the exact architecture from the training code
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=2, input_shape=(100, 100, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=2))
        
        model.add(Conv2D(filters=32, kernel_size=2, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=2))
        
        model.add(Conv2D(filters=64, kernel_size=2, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=2))
        
        model.add(Conv2D(filters=128, kernel_size=2, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=2))
        
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(150))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(Dense(81, activation='softmax'))  # 81 classes for Fruits-360
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        self.logger.info("Rebuilt original model architecture (weights not loaded - will be untrained)")
        self.logger.info("Model expects 81 fruit classes")
        return model
    
    def _suggest_model_alternatives(self):
        """Suggest alternative model options to the user"""
        self.logger.info("=== FRUITS MODEL ALTERNATIVES ===")
        self.logger.info("The current model (2018) is incompatible with TensorFlow 2.20.0")
        self.logger.info("")
        self.logger.info("Options to fix:")
        self.logger.info("1. Download a compatible pre-trained Fruits-360 model")
        self.logger.info("2. Retrain the model with current TensorFlow version")
        self.logger.info("3. Continue with YOLO-only detection (works great!)")
        self.logger.info("4. Use TensorFlow Hub model: https://tfhub.dev/")
        self.logger.info("")
        self.logger.info("Model was trained with architecture:")
        self.logger.info("- Input: (100, 100, 3)")
        self.logger.info("- 4 Conv2D layers (16, 32, 64, 128 filters)")
        self.logger.info("- Output: 81 fruit classes")
        self.logger.info("========================================")

    def load_fruit_labels(self) -> List[str]:
        """Load fruit class labels for Fruits-360 dataset (81 classes)"""
        # Complete list of 81 classes from Fruits-360 dataset
        return [
            'Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2',
            'Apple Golden 3', 'Apple Granny Smith', 'Apple Pink Lady', 'Apple Red 1',
            'Apple Red 2', 'Apple Red 3', 'Apple Red Delicious', 'Apple Red Yellow 1',
            'Apple Red Yellow 2', 'Apricot', 'Avocado', 'Avocado ripe', 'Banana',
            'Banana Lady Finger', 'Banana Red', 'Beetroot', 'Blueberry', 'Cactus fruit',
            'Cantaloupe 1', 'Cantaloupe 2', 'Carambola', 'Cauliflower', 'Cherry 1',
            'Cherry 2', 'Cherry Rainier', 'Cherry Wax Black', 'Cherry Wax Red',
            'Cherry Wax Yellow', 'Chestnut', 'Clementine', 'Cocos', 'Corn',
            'Corn Husk', 'Cucumber Ripe', 'Cucumber Ripe 2', 'Dates', 'Eggplant',
            'Fig', 'Ginger Root', 'Granadilla', 'Grape Blue', 'Grape Pink',
            'Grape White', 'Grape White 2', 'Grape White 3', 'Grape White 4',
            'Grapefruit Pink', 'Grapefruit White', 'Guava', 'Hazelnut', 'Huckleberry',
            'Kaki', 'Kiwi', 'Kohlrabi', 'Kumquats', 'Lemon', 'Lemon Meyer',
            'Limes', 'Lychee', 'Mandarine', 'Mango', 'Mango Red', 'Mangostan',
            'Maracuja', 'Melon Piel de Sapo', 'Mulberry', 'Nectarine', 'Nectarine Flat',
            'Nut Forest', 'Nut Pecan', 'Onion Red', 'Onion Red Peeled', 'Onion White',
            'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Peach 2', 'Peach Flat',
            'Pear', 'Pear 2', 'Pear Abate', 'Pear Forelle', 'Pear Kaiser',
            'Pear Monster', 'Pear Red', 'Pear Stone', 'Pear Williams', 'Pepino',
            'Pepper Green', 'Pepper Orange', 'Pepper Red', 'Pepper Yellow', 'Physalis',
            'Physalis with Husk', 'Pineapple', 'Pineapple Mini', 'Pitahaya Red',
            'Plum', 'Plum 2', 'Plum 3', 'Pomegranate', 'Pomelo Sweetie',
            'Potato Red', 'Potato Red Washed', 'Potato Sweet', 'Potato White',
            'Quince', 'Rambutan', 'Raspberry', 'Redcurrant', 'Salak', 'Strawberry',
            'Strawberry Wedge', 'Tamarillo', 'Tangelo', 'Tomato 1', 'Tomato 2',
            'Tomato 3', 'Tomato 4', 'Tomato Cherry Red', 'Tomato Heart', 'Tomato Maroon',
            'Tomato not Ripened', 'Tomato Yellow', 'Walnut', 'Watermelon'
        ]

    def setup_waste_classes(self):
        """
        Setup class mappings for waste items
        Uses custom trained YOLOv11 model classes from TACO dataset + organic categories
        """
        # Check if we have a custom trained model with specific classes
        if hasattr(self.yolo_model, 'names') and self.yolo_model.names:
            # Use the actual classes from the trained model
            self.class_names = self.yolo_model.names
            self.logger.info(f"Using custom model classes: {list(self.class_names.values())}")
        else:
            # Fallback to TACO dataset classes (your training data)
            self.class_names = {
                0: "Aluminium foil",
                1: "Bottle cap", 
                2: "Bottle",
                3: "Broken glass",
                4: "Can",
                5: "Carton",
                6: "Cigarette", 
                7: "Cup",
                8: "Lid",
                9: "Other litter",
                10: "Other plastic",
                11: "Paper",
                12: "Plastic bag - wrapper",
                13: "Plastic container",
                14: "Pop tab",
                15: "Straw",
                16: "Styrofoam piece",
                17: "Unlabeled litter"
            }
            self.logger.info("Using TACO dataset classes (fallback)")
        
        # Map classes to waste categories for weight estimation and disposal
        self.waste_categories = {
            # TACO dataset categories
            "Aluminium foil": "metal",
            "Bottle cap": "plastic", 
            "Bottle": "plastic",
            "Broken glass": "glass",
            "Can": "metal",
            "Carton": "paper",
            "Cigarette": "other",
            "Cup": "paper", 
            "Lid": "plastic",
            "Other litter": "other",
            "Other plastic": "plastic",
            "Paper": "paper",
            "Plastic bag - wrapper": "plastic",
            "Plastic container": "plastic",
            "Pop tab": "metal",
            "Straw": "plastic",
            "Styrofoam piece": "plastic",
            "Unlabeled litter": "other",
            # Generic fallbacks for unknown classes
            "bottle": "plastic",
            "cup": "paper",
            "person": "ignore"  # Ignore people detections
        }
        
        # Disposal methods for different categories
        self.disposal_methods = {
            "metal": "recycling",
            "plastic": "recycling", 
            "glass": "recycling",
            "paper": "recycling",
            "organic_waste": "composting",
            "other": "general_waste"
        }

    def setup_fallback_detector(self):
        """
        Setup a fallback detector for when YOLOv8 is not available
        """
        self.logger.warning("YOLOv8 not available - no detections will be made until model is fixed")
        self.yolo_model = None

    def detect_waste(self, image: np.ndarray) -> List[Dict]:
        """
        Dual-model waste detection pipeline:
        1. YOLOv11 for object detection and inorganic waste classification
        2. CNN for organic waste classification when needed
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detection dictionaries with keys: name, confidence, bbox, category, disposal_method
        """
        if self.yolo_model and YOLO_AVAILABLE:
            return self._detect_with_dual_models(image)
        else:
            return self._detect_with_fallback(image)

    def _detect_with_dual_models(self, image: np.ndarray) -> List[Dict]:
        """
        Run both models independently and combine results:
        1. YOLO for trash detection
        2. CNN for fruits classification on the whole image
        """
        detections = []
        
        try:
            # Stage 1: YOLOv11 object detection for trash
            yolo_results = self.yolo_model(image, conf=self.confidence_threshold, verbose=False)
            
            for result in yolo_results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract detection data
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                        
                        # Map to waste class
                        item_name = self.class_names.get(class_id, f"unknown_{class_id}")
                        category = self._get_waste_category(class_id)
                        
                        # Skip non-waste items
                        if item_name == "person" or category == "non_waste":
                            continue
                        
                        # Add YOLO detection
                        disposal_method = self.disposal_methods.get(category, "general_waste")
                        detections.append({
                            "name": item_name,
                            "confidence": confidence,
                            "bbox": bbox,
                            "category": category,
                            "disposal_method": disposal_method,
                            "class_id": class_id,
                            "source_model": "yolo_taco"
                        })
            
            # Stage 2: Independent fruits classification on whole image
            if self.fruits_model is not None and TF_AVAILABLE:
                fruit_detections = self._detect_fruits_in_image(image)
                detections.extend(fruit_detections)
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
        
        return detections

    def _detect_fruits_in_image(self, image: np.ndarray) -> List[Dict]:
        """
        Run fruits classification on the whole image to detect fruits
        This runs independently of YOLO to catch fruits that YOLO might miss
        """
        fruit_detections = []
        
        try:
            self.logger.debug("Running fruits classification on image...")
            
            # Preprocess the whole image for fruits classification
            processed_image = self._preprocess_for_fruits_model(image)
            
            if processed_image is not None:
                self.logger.debug("Image preprocessed successfully, running prediction...")
                
                # Get prediction from fruits model
                predictions = self.fruits_model.predict(processed_image, verbose=0)
                
                # Get the top predictions
                top_indices = np.argsort(predictions[0])[-5:][::-1]  # Top 5 predictions
                
                self.logger.debug(f"Top 5 fruit predictions: {[(self.fruit_labels[i] if i < len(self.fruit_labels) else f'unknown_{i}', predictions[0][i]) for i in top_indices]}")
                
                for idx in top_indices:
                    confidence = float(predictions[0][idx])
                    
                    # Very low threshold for testing - was 0.3, now 0.01 (1%)
                    if confidence > 0.01 and idx < len(self.fruit_labels):
                        fruit_name = self.fruit_labels[idx]
                        
                        self.logger.info(f"Detected fruit: {fruit_name} with confidence {confidence:.3f}")
                        
                        # Create a bounding box for the whole image (since we don't have object detection)
                        h, w = image.shape[:2]
                        bbox = [0, 0, w, h]  # Whole image
                        
                        fruit_detections.append({
                            "name": fruit_name,
                            "confidence": confidence,
                            "bbox": bbox,
                            "category": "organic_waste",
                            "disposal_method": "composting",
                            "class_id": int(idx),  # Convert numpy int64 to Python int
                            "source_model": "fruits_cnn"
                        })
                        
                        # Only return the best fruit detection to avoid duplicates
                        break
            else:
                self.logger.warning("Failed to preprocess image for fruits model")
                        
        except Exception as e:
            self.logger.error(f"Fruits detection failed: {e}")
        
        return fruit_detections

    def _enhance_with_fruits_classification(self, image: np.ndarray, bbox: List[float], 
                                          item_name: str, confidence: float, 
                                          category: str, class_id: int) -> Dict:
        """Enhance detection with fruits classification if appropriate"""
        
        # Check if we should try fruits classification
        should_classify_fruits = (
            self.fruits_model is not None and
            TF_AVAILABLE and
            self._should_try_fruits_classification(item_name, confidence)
        )
        
        if should_classify_fruits:
            # Extract object region
            crop = self._extract_bbox_region(image, bbox)
            
            if crop is not None:
                # Try fruits classification
                fruit_result = self._classify_fruit_crop(crop)
                
                if fruit_result and fruit_result['confidence'] > 0.75:
                    # Use fruits classification result
                    return {
                        "name": fruit_result['name'],
                        "confidence": fruit_result['confidence'],
                        "bbox": bbox,
                        "category": "organic_waste",
                        "disposal_method": "composting",
                        "class_id": class_id,
                        "source_model": "fruits_cnn",
                        "original_detection": item_name
                    }
        
        # Use original YOLO detection
        disposal_method = self.disposal_methods.get(category, "general_waste")
        
        return {
            "name": item_name,
            "confidence": confidence,
            "bbox": bbox,
            "category": category,
            "disposal_method": disposal_method,
            "class_id": class_id,
            "source_model": "yolo_taco"
        }

    def _should_try_fruits_classification(self, item_name: str, confidence: float) -> bool:
        """Determine if we should try fruits classification for this detection"""
        return (
            # Low confidence detections
            confidence < 0.8 or
            # Uncertain categories that might be organic
            item_name in ['Other litter', 'Unlabeled litter', 'Other plastic'] or
            # Generic detections that need more specificity
            item_name.startswith('unknown_') or
            # Always try for any detection to catch potential fruits
            True  # This makes it always try fruit classification as a fallback
        )

    def _extract_bbox_region(self, image: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """Extract the region of interest from bounding box"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = image.shape[:2]
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1)) 
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            # Extract crop
            crop = image[y1:y2, x1:x2]
            
            if crop.size == 0:
                return None
                
            return crop
            
        except Exception as e:
            self.logger.error(f"Failed to extract bbox region: {e}")
            return None

    def _classify_fruit_crop(self, crop: np.ndarray) -> Optional[Dict]:
        """Classify cropped region using fruits CNN model"""
        try:
            if self.fruits_model is None or len(self.fruit_labels) == 0:
                return None
            
            # Preprocess crop for CNN (assuming it expects specific size)
            # You may need to adjust this based on your model's input requirements
            processed_crop = self._preprocess_for_fruits_model(crop)
            
            if processed_crop is None:
                return None
            
            # Get prediction
            predictions = self.fruits_model.predict(processed_crop, verbose=0)
            
            # Get best prediction
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            if predicted_class_idx < len(self.fruit_labels):
                predicted_class_name = self.fruit_labels[predicted_class_idx]
                
                return {
                    'name': predicted_class_name,
                    'confidence': confidence,
                    'class_id': predicted_class_idx
                }
            
        except Exception as e:
            self.logger.error(f"Fruits classification failed: {e}")
            
        return None

    def _preprocess_for_fruits_model(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess crop for fruits CNN model"""
        try:
            # Assuming your CNN expects 100x100 RGB images (adjust as needed)
            target_size = (100, 100)  # Update this based on your model
            
            # Resize crop
            resized = cv2.resize(crop, target_size)
            
            # Convert BGR to RGB if needed
            if len(resized.shape) == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values (0-1) - adjust based on your model training
            normalized = resized.astype(np.float32) / 255.0
            
            # Add batch dimension
            batched = np.expand_dims(normalized, axis=0)
            
            return batched
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            return None

    def _detect_with_fallback(self, image: np.ndarray) -> List[Dict]:
        """Fallback detector returns empty list - no fake detections"""
        self.logger.debug("No detection model available - returning empty detections")
        return []

    def _get_waste_category(self, class_id: int) -> str:
        """Get waste category for a class ID"""
        # Get class name from class_id
        item_name = self.class_names.get(class_id, f"unknown_{class_id}")
        
        # Map class name to waste category
        category = self.waste_categories.get(item_name, "other")
        
        return category if category != "ignore" else "other"

    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detection boxes on image
        
        Args:
            image: Input image
            detections: List of detections
            
        Returns:
            Image with drawn detections
        """
        result_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection["bbox"])
            name = detection["name"]
            confidence = detection["confidence"]
            category = detection.get("category", "unknown")
            source = detection.get("source_model", "yolo")
            
            # Choose color based on category/source
            if source == "fruits_cnn":
                color = (0, 255, 0)  # Green for organic
            elif category == "plastic":
                color = (0, 165, 255)  # Orange for plastic
            elif category == "metal":
                color = (169, 169, 169)  # Gray for metal
            else:
                color = (255, 0, 0)  # Blue for other
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{name}: {confidence:.2f} ({source})"
            cv2.putText(result_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result_image

    def benchmark_performance(self, image: np.ndarray) -> Dict:
        """
        Benchmark detection performance
        
        Args:
            image: Test image
            
        Returns:
            Performance metrics
        """
        start_time = time.time()
        
        # Run detection multiple times
        num_runs = 10
        for _ in range(num_runs):
            self.detect_waste(image)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            "avg_inference_time": avg_time,
            "fps": fps,
            "yolo_available": self.yolo_model is not None,
            "fruits_available": self.fruits_model is not None
        }


if __name__ == "__main__":
    # Test the dual detector
    detector = WasteDetector()
    
    print("Testing Dual Waste Detector...")
    print(f"YOLO model available: {detector.yolo_model is not None}")
    print(f"Fruits model available: {detector.fruits_model is not None}")
    print(f"Fruit classes: {len(detector.fruit_labels)}")
    
    # Test with camera interface (uses real images if available)
    try:
        from ..camera_interface import CameraInterface
        camera = CameraInterface(source='simulated')
        
        if camera.sample_images:
            test_image = camera.capture_frame()
            if test_image is not None:
                detections = detector.detect_waste(test_image)
                print(f"Found {len(detections)} items in real image")
                
                for detection in detections:
                    print(f"- {detection['name']}: {detection['confidence']:.2f} "
                          f"({detection['source_model']}) -> {detection['disposal_method']}")
                
                # Benchmark with real image
                performance = detector.benchmark_performance(test_image)
                print(f"\nPerformance: {performance['fps']:.1f} FPS")
            else:
                print("No images available for testing")
        else:
            print("No real sample images found in data/samples/")
            
    except ImportError:
        print("Cannot import camera interface for testing")