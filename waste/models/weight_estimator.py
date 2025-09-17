"""
Weight estimation for detected waste items
Uses pre-mapped average weights and volume-to-mass heuristics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import math

class WeightEstimator:
    """
    Estimates weight of detected waste items using multiple approaches:
    1. Pre-mapped average weights for known items
    2. Volume-to-mass estimation using bounding box dimensions
    3. Category-based density approximations
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_weight_database()
        self.setup_density_database()
    
    def setup_weight_database(self):
        """
        Pre-mapped average weights for common waste items (in grams)
        Based on typical consumer product weights
        """
        self.item_weights = {
            # Plastic items
            "plastic_bottle": 25.0,      # 500ml bottle
            "plastic_bag": 6.0,          # Standard shopping bag
            "plastic_container": 35.0,    # Food container
            "plastic_utensil": 3.0,      # Fork/spoon/knife
            "straw": 0.4,                # Single straw
            
            # Metal items
            "metal_can": 15.0,           # Aluminum can (empty)
            
            # Glass items
            "glass_bottle": 200.0,       # Beer bottle
            "wine_glass": 150.0,         # Wine glass
            
            # Paper items
            "paper_cup": 8.0,            # Coffee cup
            "cardboard": 50.0,           # Small cardboard piece
            "tissue": 0.5,               # Single tissue
            
            # Food items (fresh)
            "apple": 150.0,              # Medium apple
            "banana": 120.0,             # Medium banana
            "orange": 180.0,             # Medium orange
            "sandwich": 200.0,           # Typical sandwich
            "pizza": 300.0,              # Pizza slice
            "donut": 60.0,               # Single donut
            "cake": 150.0,               # Cake slice
            
            # Rotten/waste food
            "rotten_apple": 140.0,       # Slightly lighter
            "rotten_banana": 100.0,      # Dried out
            "rotten_orange": 160.0,      # Some water loss
            "apple_core": 20.0,          # Just the core
            "banana_peel": 30.0,         # Just the peel
            "orange_peel": 40.0,         # Just the peel
            
            # Other items
            "cigarette_butt": 1.2,       # Single butt
            "food_wrapper": 2.0,         # Candy wrapper
            "bottle": 25.0,              # Generic bottle
            "cup": 8.0,                  # Generic cup
            "bowl": 200.0,               # Ceramic bowl
            "fork": 15.0,                # Metal fork
            "knife": 20.0,               # Table knife
            "spoon": 12.0,               # Table spoon
        }
        
        # Weight ranges for uncertainty estimation
        self.weight_ranges = {
            item: (weight * 0.7, weight * 1.3)  # ±30% variation
            for item, weight in self.item_weights.items()
        }
    
    def setup_density_database(self):
        """
        Material density estimates (g/cm³) for volume-based estimation
        """
        self.material_densities = {
            "plastic": 0.9,      # Typical plastic density
            "metal": 2.7,        # Aluminum
            "glass": 2.5,        # Glass
            "paper": 0.8,        # Paper/cardboard
            "organic": 0.6,      # Food waste (average)
            "other": 1.0,        # Default assumption
        }
        
        # Packing efficiency - how much of the bounding box is actual material
        self.packing_efficiency = {
            "plastic": 0.3,      # Bottles are mostly empty space
            "metal": 0.2,        # Cans are hollow
            "glass": 0.25,       # Glass containers
            "paper": 0.4,        # Paper items
            "organic": 0.7,      # Food items are more solid
            "other": 0.4,        # Conservative estimate
        }
    
    def estimate_weight(self, detection: Dict, image_shape: Tuple[int, int], 
                       depth_estimate: Optional[float] = None) -> Dict:
        """
        Estimate weight for a detected item
        
        Args:
            detection: Detection dictionary with name, bbox, category
            image_shape: (height, width) of the image
            depth_estimate: Estimated depth in cm (if available)
            
        Returns:
            Dictionary with weight estimates and confidence
        """
        item_name = detection["name"]
        bbox = detection["bbox"]  # [x1, y1, x2, y2]
        category = detection["category"]
        
        # Method 1: Pre-mapped weight (most accurate)
        mapped_weight = self._get_mapped_weight(item_name)
        
        # Method 2: Volume-based estimation
        volume_weight = self._estimate_from_volume(bbox, category, image_shape, depth_estimate)
        
        # Method 3: Size-based scaling
        size_weight = self._estimate_from_size(bbox, item_name, image_shape)
        
        # Combine estimates with confidence weighting
        final_estimate = self._combine_estimates(mapped_weight, volume_weight, size_weight)
        
        return {
            "weight_grams": final_estimate["weight"],
            "confidence": final_estimate["confidence"],
            "method": final_estimate["method"],
            "weight_range": final_estimate["range"],
            "breakdown": {
                "mapped": mapped_weight,
                "volume": volume_weight,
                "size": size_weight
            }
        }
    
    def _get_mapped_weight(self, item_name: str) -> Dict:
        """Get pre-mapped weight for known items"""
        if item_name in self.item_weights:
            weight = self.item_weights[item_name]
            weight_range = self.weight_ranges[item_name]
            return {
                "weight": weight,
                "confidence": 0.8,  # High confidence for known items
                "range": weight_range
            }
        else:
            return {
                "weight": None,
                "confidence": 0.0,
                "range": None
            }
    
    def _estimate_from_volume(self, bbox: List[float], category: str, 
                             image_shape: Tuple[int, int], 
                             depth_estimate: Optional[float] = None) -> Dict:
        """
        Estimate weight from bounding box volume
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2] in pixels
            category: Waste category
            image_shape: Image dimensions
            depth_estimate: Depth in cm
        """
        x1, y1, x2, y2 = bbox
        width_px = x2 - x1
        height_px = y2 - y1
        
        # Estimate real-world dimensions (assuming camera setup)
        # Typical trash can view: ~30cm width visible in frame
        pixels_per_cm = image_shape[1] / 30.0  # Rough calibration
        
        width_cm = width_px / pixels_per_cm
        height_cm = height_px / pixels_per_cm
        
        # Estimate depth if not provided
        if depth_estimate is None:
            # Assume depth proportional to smaller dimension
            depth_cm = min(width_cm, height_cm) * 0.8
        else:
            depth_cm = depth_estimate
        
        # Calculate volume
        volume_cm3 = width_cm * height_cm * depth_cm
        
        # Apply material properties
        density = self.material_densities.get(category, 1.0)
        packing = self.packing_efficiency.get(category, 0.4)
        
        # Calculate weight
        weight = volume_cm3 * density * packing
        
        # Confidence based on category knowledge
        confidence = 0.5 if category in self.material_densities else 0.3
        
        return {
            "weight": weight,
            "confidence": confidence,
            "range": (weight * 0.5, weight * 2.0),  # High uncertainty
            "volume_cm3": volume_cm3,
            "dimensions_cm": (width_cm, height_cm, depth_cm)
        }
    
    def _estimate_from_size(self, bbox: List[float], item_name: str, 
                           image_shape: Tuple[int, int]) -> Dict:
        """
        Estimate weight by scaling known items based on size
        """
        if item_name not in self.item_weights:
            return {"weight": None, "confidence": 0.0, "range": None}
        
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        
        # Typical size expectations for items (as fraction of image)
        typical_sizes = {
            "plastic_bottle": 0.03,  # 3% of image
            "apple": 0.02,           # 2% of image
            "banana": 0.025,         # 2.5% of image
            "paper_cup": 0.02,       # 2% of image
            "metal_can": 0.015,      # 1.5% of image
        }
        
        if item_name not in typical_sizes:
            return {"weight": None, "confidence": 0.0, "range": None}
        
        # Calculate size ratio
        image_area = image_shape[0] * image_shape[1]
        actual_size_fraction = bbox_area / image_area
        expected_size_fraction = typical_sizes[item_name]
        size_ratio = actual_size_fraction / expected_size_fraction
        
        # Scale weight by size ratio (assuming roughly cubic scaling)
        base_weight = self.item_weights[item_name]
        scaled_weight = base_weight * (size_ratio ** 1.5)  # Between linear and cubic
        
        # Confidence decreases with size deviation
        size_deviation = abs(size_ratio - 1.0)
        confidence = max(0.1, 0.6 - size_deviation * 0.3)
        
        return {
            "weight": scaled_weight,
            "confidence": confidence,
            "range": (scaled_weight * 0.6, scaled_weight * 1.4),
            "size_ratio": size_ratio
        }
    
    def _combine_estimates(self, mapped: Dict, volume: Dict, size: Dict) -> Dict:
        """
        Combine different weight estimates using confidence weighting
        """
        estimates = []
        
        # Add valid estimates
        if mapped["weight"] is not None:
            estimates.append(("mapped", mapped["weight"], mapped["confidence"]))
        
        if volume["weight"] is not None and volume["weight"] > 0:
            estimates.append(("volume", volume["weight"], volume["confidence"]))
        
        if size["weight"] is not None and size["weight"] > 0:
            estimates.append(("size", size["weight"], size["confidence"]))
        
        if not estimates:
            return {
                "weight": 10.0,  # Default fallback weight
                "confidence": 0.1,
                "method": "default",
                "range": (1.0, 50.0)
            }
        
        # Weighted average
        total_weight = 0.0
        total_confidence = 0.0
        best_method = estimates[0][0]
        best_confidence = estimates[0][2]
        
        for method, weight, confidence in estimates:
            total_weight += weight * confidence
            total_confidence += confidence
            
            if confidence > best_confidence:
                best_method = method
                best_confidence = confidence
        
        final_weight = total_weight / total_confidence if total_confidence > 0 else 10.0
        final_confidence = min(1.0, total_confidence / len(estimates))
        
        # Calculate combined range
        weights = [est[1] for est in estimates]
        weight_range = (min(weights) * 0.8, max(weights) * 1.2)
        
        return {
            "weight": final_weight,
            "confidence": final_confidence,
            "method": best_method,
            "range": weight_range
        }
    
    def estimate_batch_weight(self, detections: List[Dict], image_shape: Tuple[int, int]) -> Dict:
        """
        Estimate total weight for a batch of detections
        
        Args:
            detections: List of detections with weight estimates
            image_shape: Image dimensions
            
        Returns:
            Batch weight summary
        """
        total_weight = 0.0
        total_confidence = 0.0
        category_weights = {}
        item_count = len(detections)
        
        for detection in detections:
            if "weight_estimate" in detection:
                weight_info = detection["weight_estimate"]
                weight = weight_info["weight_grams"]
                confidence = weight_info["confidence"]
                category = detection["category"]
                
                total_weight += weight
                total_confidence += confidence
                
                if category not in category_weights:
                    category_weights[category] = 0.0
                category_weights[category] += weight
        
        avg_confidence = total_confidence / item_count if item_count > 0 else 0.0
        
        return {
            "total_weight_grams": total_weight,
            "total_weight_kg": total_weight / 1000.0,
            "average_confidence": avg_confidence,
            "item_count": item_count,
            "category_breakdown": category_weights,
            "estimated_range": (total_weight * 0.7, total_weight * 1.3)
        }


def test_weight_estimator():
    """Test the weight estimator"""
    estimator = WeightEstimator()
    
    # Test detection
    test_detection = {
        "name": "plastic_bottle",
        "bbox": [100, 100, 200, 300],  # 100x200 pixel box
        "category": "plastic",
        "confidence": 0.8
    }
    
    image_shape = (480, 640)  # VGA resolution
    
    weight_estimate = estimator.estimate_weight(test_detection, image_shape)
    
    print("Weight Estimation Test:")
    print(f"Item: {test_detection['name']}")
    print(f"Estimated weight: {weight_estimate['weight_grams']:.1f}g")
    print(f"Confidence: {weight_estimate['confidence']:.2f}")
    print(f"Method: {weight_estimate['method']}")
    print(f"Range: {weight_estimate['weight_range'][0]:.1f}-{weight_estimate['weight_range'][1]:.1f}g")


if __name__ == "__main__":
    test_weight_estimator()