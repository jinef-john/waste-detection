"""
Data logging system for CSV/JSON output
Handles local storage of detection results and batch summaries
"""

import csv
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import threading
import time

from .batch_tracker import BatchSummary, WasteItem

class DataLogger:
    """
    Handles logging of waste detection data to CSV and JSON formats
    Thread-safe logging with configurable output formats
    """
    
    def __init__(self, log_dir: str = "logs", enable_csv: bool = True, 
                 enable_json: bool = True, max_log_files: int = 100):
        """
        Initialize data logger
        
        Args:
            log_dir: Directory for log files
            enable_csv: Enable CSV logging
            enable_json: Enable JSON logging  
            max_log_files: Maximum number of log files to keep
        """
        self.log_dir = Path(log_dir)
        self.enable_csv = enable_csv
        self.enable_json = enable_json
        self.max_log_files = max_log_files
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV file paths
        self.csv_batch_file = self.log_dir / "waste_batches.csv"
        self.csv_items_file = self.log_dir / "waste_items.csv"
        
        # JSON file paths  
        self.json_batch_file = self.log_dir / "waste_batches.json"
        self.json_daily_file = self.log_dir / f"waste_log_{datetime.now().strftime('%Y%m%d')}.json"
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Initialize files
        self.setup_logging()
        self._initialize_csv_files()
        self._initialize_json_files()
        
        self.logger.info(f"Data logger initialized: {log_dir}")
    
    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)
    
    def _initialize_csv_files(self):
        """Initialize CSV files with headers if they don't exist"""
        if not self.enable_csv:
            return
        
        # Batch summary CSV headers
        batch_headers = [
            "batch_id", "start_time", "end_time", "start_datetime", "end_datetime",
            "duration_minutes", "unique_items", "total_weight_grams", "total_weight_kg",
            "organic_weight_g", "plastic_weight_g", "metal_weight_g", "glass_weight_g",
            "paper_weight_g", "other_weight_g"
        ]
        
        # Individual items CSV headers
        item_headers = [
            "batch_id", "item_name", "category", "weight_grams", "confidence", 
            "timestamp", "datetime", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "detection_hash"
        ]
        
        # Initialize batch CSV
        if not self.csv_batch_file.exists():
            with open(self.csv_batch_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(batch_headers)
        
        # Initialize items CSV
        if not self.csv_items_file.exists():
            with open(self.csv_items_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(item_headers)
    
    def _initialize_json_files(self):
        """Initialize JSON files if they don't exist"""
        if not self.enable_json:
            return
        
        # Initialize batch JSON file
        if not self.json_batch_file.exists():
            with open(self.json_batch_file, 'w') as f:
                json.dump({"batches": [], "metadata": {"created": datetime.now().isoformat()}}, f)
        
        # Initialize daily JSON file  
        if not self.json_daily_file.exists():
            with open(self.json_daily_file, 'w') as f:
                json.dump({
                    "date": datetime.now().strftime('%Y-%m-%d'),
                    "batches": [],
                    "daily_summary": {
                        "total_items": 0,
                        "total_weight_kg": 0.0,
                        "categories": {}
                    }
                }, f, indent=2)
    
    def log_batch_summary(self, batch_summary: BatchSummary):
        """
        Log a completed batch summary
        
        Args:
            batch_summary: Completed batch summary to log
        """
        with self.lock:
            try:
                if self.enable_csv:
                    self._log_batch_csv(batch_summary)
                    self._log_items_csv(batch_summary)
                
                if self.enable_json:
                    self._log_batch_json(batch_summary)
                    self._update_daily_json(batch_summary)
                
                self.logger.info(f"Logged batch {batch_summary.batch_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to log batch {batch_summary.batch_id}: {e}")
    
    def _log_batch_csv(self, batch_summary: BatchSummary):
        """Log batch summary to CSV"""
        # Prepare category weights with defaults
        category_weights = {
            "organic": 0.0, "plastic": 0.0, "metal": 0.0, 
            "glass": 0.0, "paper": 0.0, "other": 0.0
        }
        category_weights.update(batch_summary.category_weights)
        
        # Create row data
        row = [
            batch_summary.batch_id,
            batch_summary.start_time,
            batch_summary.end_time,
            datetime.fromtimestamp(batch_summary.start_time).isoformat(),
            datetime.fromtimestamp(batch_summary.end_time).isoformat(),
            (batch_summary.end_time - batch_summary.start_time) / 60.0,
            batch_summary.unique_items,
            batch_summary.total_weight_grams,
            batch_summary.total_weight_grams / 1000.0,
            category_weights["organic"],
            category_weights["plastic"],
            category_weights["metal"],
            category_weights["glass"],
            category_weights["paper"],
            category_weights["other"]
        ]
        
        # Append to CSV
        with open(self.csv_batch_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def _log_items_csv(self, batch_summary: BatchSummary):
        """Log individual items to CSV"""
        rows = []
        
        for item in batch_summary.items:
            row = [
                item.batch_id,
                item.name,
                item.category,
                item.weight_grams,
                item.confidence,
                item.timestamp,
                datetime.fromtimestamp(item.timestamp).isoformat(),
                item.bbox[0], item.bbox[1], item.bbox[2], item.bbox[3],
                item.detection_hash
            ]
            rows.append(row)
        
        # Append all rows to CSV
        with open(self.csv_items_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    
    def _log_batch_json(self, batch_summary: BatchSummary):
        """Log batch summary to JSON file"""
        # Load existing data
        with open(self.json_batch_file, 'r') as f:
            data = json.load(f)
        
        # Add new batch
        data["batches"].append(batch_summary.to_dict())
        data["metadata"]["last_updated"] = datetime.now().isoformat()
        
        # Write back to file
        with open(self.json_batch_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _update_daily_json(self, batch_summary: BatchSummary):
        """Update daily JSON log"""
        # Check if we need a new daily file
        current_date = datetime.now().strftime('%Y%m%d')
        expected_file = self.log_dir / f"waste_log_{current_date}.json"
        
        if expected_file != self.json_daily_file:
            self.json_daily_file = expected_file
            self._initialize_json_files()
        
        # Load current daily data
        with open(self.json_daily_file, 'r') as f:
            daily_data = json.load(f)
        
        # Add batch to daily log
        daily_data["batches"].append(batch_summary.to_dict())
        
        # Update daily summary
        summary = daily_data["daily_summary"]
        summary["total_items"] += batch_summary.unique_items
        summary["total_weight_kg"] += batch_summary.total_weight_grams / 1000.0
        
        # Update category breakdown
        for category, weight in batch_summary.category_weights.items():
            if category not in summary["categories"]:
                summary["categories"][category] = 0.0
            summary["categories"][category] += weight / 1000.0
        
        summary["last_updated"] = datetime.now().isoformat()
        
        # Write back
        with open(self.json_daily_file, 'w') as f:
            json.dump(daily_data, f, indent=2)
    
    def log_real_time_detection(self, detection: Dict, weight_estimate: Dict, 
                               batch_id: str, timestamp: Optional[float] = None):
        """
        Log real-time detection (before batch completion)
        
        Args:
            detection: Detection dictionary
            weight_estimate: Weight estimation 
            batch_id: Current batch ID
            timestamp: Detection timestamp
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Create real-time log entry
        log_entry = {
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            "batch_id": batch_id,
            "detection": detection,
            "weight_estimate": weight_estimate
        }
        
        # Log to real-time file
        realtime_file = self.log_dir / "realtime_detections.jsonl"
        
        with self.lock:
            with open(realtime_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
    
    def get_daily_summary(self, date: Optional[str] = None) -> Dict:
        """
        Get daily summary for a specific date
        
        Args:
            date: Date string in YYYYMMDD format, defaults to today
            
        Returns:
            Daily summary dictionary
        """
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
        
        daily_file = self.log_dir / f"waste_log_{date}.json"
        
        if not daily_file.exists():
            return {
                "date": date,
                "total_items": 0,
                "total_weight_kg": 0.0,
                "categories": {},
                "batches_count": 0
            }
        
        with open(daily_file, 'r') as f:
            data = json.load(f)
        
        summary = data.get("daily_summary", {})
        summary["batches_count"] = len(data.get("batches", []))
        
        return summary
    
    def get_recent_batches(self, count: int = 10) -> List[Dict]:
        """
        Get recent batch summaries
        
        Args:
            count: Number of recent batches to return
            
        Returns:
            List of batch summary dictionaries
        """
        if not self.json_batch_file.exists():
            return []
        
        with open(self.json_batch_file, 'r') as f:
            data = json.load(f)
        
        batches = data.get("batches", [])
        return batches[-count:] if batches else []
    
    def export_csv_range(self, start_date: str, end_date: str, 
                        output_file: Optional[str] = None) -> str:
        """
        Export CSV data for a date range
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_file: Output file path
            
        Returns:
            Path to exported file
        """
        if output_file is None:
            output_file = self.log_dir / f"export_{start_date}_to_{end_date}.csv"
        
        # Read all batch data
        if not self.csv_batch_file.exists():
            raise FileNotFoundError("No batch CSV data found")
        
        exported_rows = []
        with open(self.csv_batch_file, 'r') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            
            for row in reader:
                batch_date = row['start_datetime'][:10]  # Extract YYYY-MM-DD
                if start_date <= batch_date <= end_date:
                    exported_rows.append(row)
        
        # Write filtered data
        with open(output_file, 'w', newline='') as f:
            if headers and exported_rows:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(exported_rows)
        
        return str(output_file)
    
    def cleanup_old_logs(self):
        """Remove old log files beyond max_log_files limit"""
        # Get all log files sorted by modification time
        log_files = []
        for pattern in ["waste_log_*.json", "waste_batches_*.json", "*.csv"]:
            log_files.extend(self.log_dir.glob(pattern))
        
        log_files.sort(key=lambda x: x.stat().st_mtime)
        
        # Remove oldest files if over limit
        if len(log_files) > self.max_log_files:
            files_to_remove = log_files[:-self.max_log_files]
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                    self.logger.info(f"Removed old log file: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove {file_path}: {e}")
    
    def get_log_statistics(self) -> Dict:
        """Get logging statistics"""
        stats = {
            "log_directory": str(self.log_dir),
            "csv_enabled": self.enable_csv,
            "json_enabled": self.enable_json,
            "files": {}
        }
        
        # Check file sizes and counts
        for log_file in self.log_dir.iterdir():
            if log_file.is_file():
                stats["files"][log_file.name] = {
                    "size_bytes": log_file.stat().st_size,
                    "modified": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                }
        
        # Count total batches logged
        if self.json_batch_file.exists():
            try:
                with open(self.json_batch_file, 'r') as f:
                    data = json.load(f)
                    stats["total_batches_logged"] = len(data.get("batches", []))
            except:
                stats["total_batches_logged"] = 0
        
        return stats


def test_data_logger():
    """Test the data logger"""
    from datetime import datetime
    import tempfile
    import shutil
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    
    try:
        logger = DataLogger(log_dir=temp_dir)
        
        # Create test batch summary
        test_items = [
            WasteItem(
                name="plastic_bottle",
                category="plastic", 
                weight_grams=25.0,
                confidence=0.8,
                bbox=[100, 100, 200, 200],
                timestamp=time.time(),
                batch_id="test_batch_001"
            ),
            WasteItem(
                name="apple",
                category="organic",
                weight_grams=150.0, 
                confidence=0.9,
                bbox=[300, 300, 400, 400],
                timestamp=time.time(),
                batch_id="test_batch_001"
            )
        ]
        
        from .batch_tracker import BatchSummary
        test_batch = BatchSummary(
            batch_id="test_batch_001",
            start_time=time.time() - 300,
            end_time=time.time(),
            items=test_items
        )
        
        # Calculate summary stats
        test_batch.unique_items = len(test_items)
        test_batch.total_weight_grams = sum(item.weight_grams for item in test_items)
        test_batch.item_counts = {"plastic_bottle": 1, "apple": 1}
        test_batch.category_weights = {"plastic": 25.0, "organic": 150.0}
        
        # Log the batch
        logger.log_batch_summary(test_batch)
        
        print("Data Logger Test:")
        print(f"Logged batch: {test_batch.batch_id}")
        print(f"Items: {test_batch.unique_items}")
        print(f"Weight: {test_batch.total_weight_grams}g")
        
        # Get statistics
        stats = logger.get_log_statistics()
        print(f"Log files created: {len(stats['files'])}")
        
        # Get daily summary
        daily = logger.get_daily_summary()
        print(f"Daily items: {daily.get('total_items', 0)}")
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_data_logger()