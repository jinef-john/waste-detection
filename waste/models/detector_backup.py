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
            self.logger.warning("TensorFlow not available - organic waste detection disabled")norganic waste detection (TACO dataset)  
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
    logging.warning("YOLOv8 not available. Install with: pip install ultralytics")

class WasteDetector:
    """
    Dual-model waste detection system:
    - YOLOv11 for object detection and inorganic waste classification (TACO dataset)
    - CNN for organic waste classification when needed (Fruits-360 dataset)
    """
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        """
        Initialize the waste detector
        
        Args:
            model_path: Path to custom trained model, if None uses YOLOv8n
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = {}
        self.setup_logging()
        
        if YOLO_AVAILABLE:
            self.load_model(model_path)
        else:
            self.setup_fallback_detector()
    
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
        """Load the CNN fruits classification model"""
        try:
            fruits_model_path = "data/fruits/cnn_from_scratch_fruits.hdf5"
            if os.path.exists(fruits_model_path):
                self.fruits_model = keras.models.load_model(fruits_model_path)
                self.logger.info(f"Loaded fruits CNN model: {fruits_model_path}")
                
                # Load fruit labels (you'll need to provide these)
                self.fruit_labels = self.load_fruit_labels()
                self.logger.info(f"Loaded {len(self.fruit_labels)} fruit classes")
            else:
                self.logger.warning(f"Fruits model not found at {fruits_model_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to load fruits model: {e}")
            self.fruits_model = None
    
    def load_fruit_labels(self) -> List[str]:
        """Load fruit class labels - you can customize this list based on your training"""
        # Common fruits from Fruits-360 dataset - update this with your actual labels
        return [
            'Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden', 'Apple Granny Smith',
            'Apple Pink Lady', 'Apple Red', 'Apricot', 'Avocado', 'Banana', 'Banana Lady Finger',
            'Blueberry', 'Cherry', 'Clementine', 'Coconut', 'Grape Blue', 'Grape Pink',
            'Grape White', 'Grapefruit Pink', 'Grapefruit White', 'Kiwi', 'Lemon', 'Lime',
            'Mango', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Pear', 'Pineapple',
            'Plum', 'Pomegranate', 'Raspberry', 'Strawberry', 'Tomato', 'Watermelon'
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
        """Two-stage detection using YOLO + CNN"""
        detections = []
        
        try:
            # Stage 1: YOLOv11 object detection
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
                        
                        # Stage 2: Try fruits classification for uncertain items
                        enhanced_detection = self._enhance_with_fruits_classification(
                            image, bbox, item_name, confidence, category, class_id
                        )
                        
                        detections.append(enhanced_detection)
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
        
        return detections
    
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
            confidence < 0.7 or
            # Uncertain categories that might be organic
            item_name in ['Other litter', 'Unlabeled litter', 'Other plastic'] or
            # Generic detections that need more specificity
            item_name.startswith('unknown_')
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
            
            # Normalize pixel values (0-1)
            normalized = resized.astype(np.float32) / 255.0
            
            # Add batch dimension
            batched = np.expand_dims(normalized, axis=0)
            
            return batched
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            return None
        """Detect using YOLOv8 model"""
        detections = []
        
        try:
            # Run inference
            results = self.yolo_model(image, conf=self.confidence_threshold, verbose=False)
            
            for result in results:
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
                        
                        detections.append({
                            "name": item_name,
                            "confidence": confidence,
                            "bbox": bbox,
                            "category": category,
                            "class_id": class_id
                        })
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
        
        return detections
    
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
            category = detection["category"]
            
            # Color coding by category
            colors = {
                "organic": (0, 255, 0),     # Green
                "plastic": (0, 0, 255),     # Red
                "metal": (128, 128, 128),   # Gray
                "glass": (255, 255, 0),     # Cyan
                "paper": (0, 255, 255),     # Yellow
                "other": (255, 0, 255),     # Magenta
                "unknown": (255, 255, 255)  # White
            }
            
            color = colors.get(category, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result_image
    
    def benchmark_performance(self, image: np.ndarray, iterations: int = 10) -> Dict:
        """
        Benchmark detection performance
        
        Args:
            image: Test image
            iterations: Number of iterations to run
            
        Returns:
            Performance metrics
        """
        times = []
        
        for i in range(iterations):
            start_time = time.time()
            detections = self.detect_waste(image)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            "avg_inference_time": avg_time,
            "fps": fps,
            "min_time": np.min(times),
            "max_time": np.max(times),
            "std_time": np.std(times)
        }


if __name__ == "__main__":
    # Test the detector with real camera interface
    detector = WasteDetector()
    
    print("Testing Waste Detector...")
    print("Note: No synthetic test data - detector requires real images or working YOLOv8 model")
    
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
                    print(f"- {detection['name']}: {detection['confidence']:.2f}")
                
                # Benchmark with real image
                performance = detector.benchmark_performance(test_image)
                print(f"\nPerformance: {performance['fps']:.1f} FPS")
            else:
                print("No images available for testing")
        else:
            print("No real sample images found in data/samples/")
            
    except ImportError:
        print("Cannot import camera interface for testing")