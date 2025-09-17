"""
Batch tracking system for 30-minute waste disposal windows
Handles duplicate detection and item counting within time windows
"""

import time
import threading
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import logging
import hashlib
import json

@dataclass
class WasteItem:
    """Individual waste item with tracking information"""
    name: str
    category: str
    weight_grams: float
    confidence: float
    bbox: List[float]
    timestamp: float
    batch_id: str
    detection_hash: str = ""
    
    def __post_init__(self):
        """Generate detection hash for duplicate detection"""
        if not self.detection_hash:
            self.detection_hash = self._generate_hash()
    
    def _generate_hash(self) -> str:
        """Generate hash for duplicate detection based on position and type"""
        # Use position and item type for duplicate detection
        position_str = f"{self.name}_{int(self.bbox[0]/10)}_{int(self.bbox[1]/10)}"
        return hashlib.md5(position_str.encode()).hexdigest()[:8]

@dataclass
class BatchSummary:
    """Summary of a completed 30-minute batch"""
    batch_id: str
    start_time: float
    end_time: float
    items: List[WasteItem] = field(default_factory=list)
    item_counts: Dict[str, int] = field(default_factory=dict)
    total_weight_grams: float = 0.0
    category_weights: Dict[str, float] = field(default_factory=dict)
    unique_items: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "batch_id": self.batch_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "start_datetime": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_datetime": datetime.fromtimestamp(self.end_time).isoformat(),
            "duration_minutes": (self.end_time - self.start_time) / 60.0,
            "items": [
                {
                    "name": item.name,
                    "category": item.category,
                    "weight_grams": item.weight_grams,
                    "confidence": item.confidence,
                    "timestamp": item.timestamp,
                    "detection_hash": item.detection_hash
                }
                for item in self.items
            ],
            "summary": {
                "unique_items": self.unique_items,
                "total_weight_grams": self.total_weight_grams,
                "total_weight_kg": self.total_weight_grams / 1000.0,
                "item_counts": self.item_counts,
                "category_weights": self.category_weights
            }
        }

class BatchTracker:
    """
    Tracks waste disposal in 30-minute batches with duplicate detection
    """
    
    def __init__(self, batch_duration_minutes: float = 30.0, 
                 duplicate_threshold_seconds: float = 5.0,
                 position_threshold_pixels: float = 50.0):
        """
        Initialize batch tracker
        
        Args:
            batch_duration_minutes: Duration of each batch in minutes
            duplicate_threshold_seconds: Time window for duplicate detection
            position_threshold_pixels: Pixel distance threshold for duplicate detection
        """
        self.batch_duration = batch_duration_minutes * 60.0  # Convert to seconds
        self.duplicate_threshold = duplicate_threshold_seconds
        self.position_threshold = position_threshold_pixels
        
        # Current batch tracking
        self.current_batch_id = self._generate_batch_id()
        self.batch_start_time = time.time()
        self.current_items: List[WasteItem] = []
        self.recent_detections: List[WasteItem] = []  # For duplicate detection
        
        # Completed batches
        self.completed_batches: List[BatchSummary] = []
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Statistics tracking
        self.total_items_detected = 0
        self.total_items_filtered = 0
        
        self.setup_logging()
        self.logger.info(f"Batch tracker initialized: {batch_duration_minutes}min batches")
    
    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)
    
    def _generate_batch_id(self) -> str:
        """Generate unique batch ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"batch_{timestamp}"
    
    def add_detection(self, detection: Dict, weight_estimate: Dict) -> bool:
        """
        Add a new detection to the current batch
        
        Args:
            detection: Detection dictionary with name, bbox, category, etc.
            weight_estimate: Weight estimation dictionary
            
        Returns:
            True if item was added (not a duplicate), False if filtered out
        """
        with self.lock:
            current_time = time.time()
            
            # Check if we need to start a new batch
            if self._should_start_new_batch(current_time):
                self._complete_current_batch(current_time)
                self._start_new_batch(current_time)
            
            # Create waste item
            item = WasteItem(
                name=detection["name"],
                category=detection["category"],
                weight_grams=weight_estimate["weight_grams"],
                confidence=detection["confidence"],
                bbox=detection["bbox"],
                timestamp=current_time,
                batch_id=self.current_batch_id
            )
            
            # Check for duplicates
            if self._is_duplicate(item):
                self.total_items_filtered += 1
                self.logger.debug(f"Filtered duplicate: {item.name}")
                return False
            
            # Add to current batch
            self.current_items.append(item)
            self.recent_detections.append(item)
            self.total_items_detected += 1
            
            # Clean old recent detections
            self._clean_recent_detections(current_time)
            
            self.logger.info(f"Added {item.name} to {self.current_batch_id}")
            return True
    
    def _should_start_new_batch(self, current_time: float) -> bool:
        """Check if we should start a new batch"""
        return (current_time - self.batch_start_time) >= self.batch_duration
    
    def _is_duplicate(self, new_item: WasteItem) -> bool:
        """
        Check if item is a duplicate based on recent detections
        
        Args:
            new_item: New item to check
            
        Returns:
            True if item is likely a duplicate
        """
        current_time = new_item.timestamp
        
        for recent_item in self.recent_detections:
            # Skip if too old
            if (current_time - recent_item.timestamp) > self.duplicate_threshold:
                continue
            
            # Check if same item type
            if recent_item.name != new_item.name:
                continue
            
            # Check position similarity
            if self._calculate_bbox_distance(recent_item.bbox, new_item.bbox) < self.position_threshold:
                return True
        
        return False
    
    def _calculate_bbox_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate distance between bounding box centers"""
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _clean_recent_detections(self, current_time: float):
        """Remove old detections from recent list"""
        cutoff_time = current_time - self.duplicate_threshold * 2  # Keep extra buffer
        self.recent_detections = [
            item for item in self.recent_detections 
            if item.timestamp > cutoff_time
        ]
    
    def _complete_current_batch(self, end_time: float):
        """Complete the current batch and generate summary"""
        if not self.current_items:
            self.logger.info(f"Completed empty batch: {self.current_batch_id}")
            return
        
        # Generate batch summary
        summary = self._generate_batch_summary(end_time)
        self.completed_batches.append(summary)
        
        self.logger.info(
            f"Completed batch {self.current_batch_id}: "
            f"{summary.unique_items} items, {summary.total_weight_grams:.1f}g"
        )
    
    def _start_new_batch(self, start_time: float):
        """Start a new batch"""
        self.current_batch_id = self._generate_batch_id()
        self.batch_start_time = start_time
        self.current_items = []
        
        self.logger.info(f"Started new batch: {self.current_batch_id}")
    
    def _generate_batch_summary(self, end_time: float) -> BatchSummary:
        """Generate summary for completed batch"""
        summary = BatchSummary(
            batch_id=self.current_batch_id,
            start_time=self.batch_start_time,
            end_time=end_time,
            items=self.current_items.copy()
        )
        
        # Calculate statistics
        item_counts = {}
        category_weights = {}
        total_weight = 0.0
        
        for item in self.current_items:
            # Item counts
            item_counts[item.name] = item_counts.get(item.name, 0) + 1
            
            # Category weights
            category_weights[item.category] = category_weights.get(item.category, 0.0) + item.weight_grams
            
            # Total weight
            total_weight += item.weight_grams
        
        summary.item_counts = item_counts
        summary.category_weights = category_weights
        summary.total_weight_grams = total_weight
        summary.unique_items = len(self.current_items)
        
        return summary
    
    def force_complete_batch(self) -> Optional[BatchSummary]:
        """Force completion of current batch (e.g., when bin is emptied)"""
        with self.lock:
            if not self.current_items:
                return None
            
            current_time = time.time()
            self._complete_current_batch(current_time)
            
            # Get the last completed batch
            last_batch = self.completed_batches[-1] if self.completed_batches else None
            
            # Start new batch
            self._start_new_batch(current_time)
            
            return last_batch
    
    def get_current_batch_status(self) -> Dict:
        """Get current batch status"""
        with self.lock:
            current_time = time.time()
            elapsed_time = current_time - self.batch_start_time
            remaining_time = max(0, self.batch_duration - elapsed_time)
            
            # Current batch statistics
            item_counts = {}
            total_weight = 0.0
            
            for item in self.current_items:
                item_counts[item.name] = item_counts.get(item.name, 0) + 1
                total_weight += item.weight_grams
            
            return {
                "batch_id": self.current_batch_id,
                "elapsed_minutes": elapsed_time / 60.0,
                "remaining_minutes": remaining_time / 60.0,
                "progress": min(1.0, elapsed_time / self.batch_duration),
                "items_count": len(self.current_items),
                "total_weight_grams": total_weight,
                "item_counts": item_counts,
                "will_complete_soon": remaining_time < 60.0  # Less than 1 minute
            }
    
    def get_recent_batches(self, count: int = 5) -> List[BatchSummary]:
        """Get recent completed batches"""
        with self.lock:
            return self.completed_batches[-count:] if self.completed_batches else []
    
    def get_statistics(self) -> Dict:
        """Get overall tracking statistics"""
        with self.lock:
            total_batches = len(self.completed_batches)
            total_items_in_batches = sum(batch.unique_items for batch in self.completed_batches)
            total_weight_in_batches = sum(batch.total_weight_grams for batch in self.completed_batches)
            
            avg_items_per_batch = total_items_in_batches / total_batches if total_batches > 0 else 0
            avg_weight_per_batch = total_weight_in_batches / total_batches if total_batches > 0 else 0
            
            duplicate_rate = self.total_items_filtered / (self.total_items_detected + self.total_items_filtered)
            
            return {
                "total_batches_completed": total_batches,
                "total_items_detected": self.total_items_detected,
                "total_items_filtered": self.total_items_filtered,
                "duplicate_filter_rate": duplicate_rate,
                "average_items_per_batch": avg_items_per_batch,
                "average_weight_per_batch_grams": avg_weight_per_batch,
                "total_weight_tracked_kg": total_weight_in_batches / 1000.0,
                "current_batch_items": len(self.current_items)
            }


def test_batch_tracker():
    """Test the batch tracker"""
    tracker = BatchTracker(batch_duration_minutes=1.0)  # 1 minute for testing
    
    # Test detections
    test_detections = [
        {"name": "plastic_bottle", "category": "plastic", "bbox": [100, 100, 200, 200], "confidence": 0.8},
        {"name": "plastic_bottle", "category": "plastic", "bbox": [105, 105, 205, 205], "confidence": 0.8},  # Duplicate
        {"name": "apple", "category": "organic", "bbox": [300, 300, 400, 400], "confidence": 0.9},
        {"name": "metal_can", "category": "metal", "bbox": [150, 150, 250, 250], "confidence": 0.7},
    ]
    
    test_weights = [
        {"weight_grams": 25.0, "confidence": 0.8},
        {"weight_grams": 25.0, "confidence": 0.8},
        {"weight_grams": 150.0, "confidence": 0.9},
        {"weight_grams": 15.0, "confidence": 0.7},
    ]
    
    print("Testing Batch Tracker...")
    
    # Add detections
    for detection, weight in zip(test_detections, test_weights):
        added = tracker.add_detection(detection, weight)
        print(f"Added {detection['name']}: {added}")
        time.sleep(0.1)
    
    # Check status
    status = tracker.get_current_batch_status()
    print(f"\nCurrent batch: {status['items_count']} items, {status['total_weight_grams']:.1f}g")
    
    # Get statistics
    stats = tracker.get_statistics()
    print(f"Duplicate filter rate: {stats['duplicate_filter_rate']:.2f}")


if __name__ == "__main__":
    test_batch_tracker()