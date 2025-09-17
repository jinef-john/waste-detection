"""
Waste Detection Model using YOLOv8-nano
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
    logging.warning("YOLOv8 not available. Install with: pip install ultralytics")

class WasteDetector:
    """
    Waste detection using YOLOv8-nano model
    Supports both TACO (inorganic litter) and Fruits-360 (fruit) datasets
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
    
    def load_model(self, model_path: Optional[str] = None):
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
                    self.model = YOLO(custom_model_path)
                    self.logger.info(f"Loaded custom YOLOv11 trash detection model: {custom_model_path}")
                elif model_path and Path(model_path).exists():
                    self.model = YOLO(model_path)
                    self.logger.info(f"Loaded custom model from {model_path}")
                else:
                    # Fallback to YOLOv8-nano for Pi 4B performance
                    self.model = YOLO('yolov8n.pt')
                    self.logger.info("Loaded YOLOv8-nano model (fallback)")
                
                # Setup class mappings for waste detection
                self.setup_waste_classes()
                
            finally:
                # Always restore original torch.load
                torch.load = old_load
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.setup_fallback_detector()
    
    def setup_waste_classes(self):
        """
        Setup class mappings for waste items
        Uses custom trained YOLOv11 model classes from TACO dataset
        """
        # Check if we have a custom trained model with specific classes
        if hasattr(self.model, 'names') and self.model.names:
            # Use the actual classes from the trained model
            self.class_names = self.model.names
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
        
        # Map classes to waste categories for weight estimation
        self.waste_categories = {
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
    
    def setup_fallback_detector(self):
        """
        Setup a fallback detector for when YOLOv8 is not available
        """
        self.logger.warning("YOLOv8 not available - no detections will be made until model is fixed")
        self.model = None
    
    def detect_waste(self, image: np.ndarray) -> List[Dict]:
        """
        Detect waste items in an image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detection dictionaries with keys: name, confidence, bbox, category
        """
        if self.model and YOLO_AVAILABLE:
            return self._detect_with_yolo(image)
        else:
            return self._detect_with_fallback(image)
    
    def _detect_with_yolo(self, image: np.ndarray) -> List[Dict]:
        """Detect using YOLOv8 model"""
        detections = []
        
        try:
            # Run inference
            results = self.model(image, conf=self.confidence_threshold, verbose=False)
            
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