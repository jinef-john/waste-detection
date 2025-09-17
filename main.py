#!/usr/bin/env python3
"""
Smart Trash Detection System - Main Application
Raspberry Pi 4B standalone waste monitoring system

This script orchestrates all components:
- Camera interface for image capture
- YOLOv8-nano detection model
- Weight estimation 
- 30-minute batch tracking
- CSV/JSON logging

Usage:
    python main.py [--config config.json] [--simulate] [--debug]
"""

import argparse
import logging
import time
import signal
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import threading

# Import waste detection components
from waste.camera_interface import CameraInterface
from waste.models.detector import WasteDetector
from waste.models.weight_estimator import WeightEstimator
from waste.utils.batch_tracker import BatchTracker
from waste.utils.data_logger import DataLogger

class SmartTrashSystem:
    """
    Main smart trash detection system
    Coordinates all components for continuous waste monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the smart trash system
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.running = False
        self.setup_logging()
        
        # Initialize components
        self.camera = None
        self.detector = None
        self.weight_estimator = None
        self.batch_tracker = None
        self.data_logger = None
        
        # Performance tracking
        self.detection_count = 0
        self.start_time = None
        
        # Thread for background processing
        self.processing_thread = None
        
        self.logger.info("Smart Trash System initialized")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.get('log_level', 'INFO'))
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/smart_trash.log')
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def initialize_components(self):
        """Initialize all system components"""
        try:
            self.logger.info("Initializing system components...")
            
            # Initialize camera
            camera_config = self.config.get('camera', {})
            camera_source = camera_config.get('source', 'auto')
            resolution = tuple(camera_config.get('resolution', [640, 480]))
            
            self.camera = CameraInterface(
                source=camera_source,
                resolution=resolution,
                fps=camera_config.get('fps', 30)
            )
            
            # Initialize detector
            detector_config = self.config.get('detector', {})
            model_path = detector_config.get('model_path')
            confidence_threshold = detector_config.get('confidence_threshold', 0.5)
            
            self.detector = WasteDetector(
                model_path=model_path,
                confidence_threshold=confidence_threshold
            )
            
            # Initialize weight estimator
            self.weight_estimator = WeightEstimator()
            
            # Initialize batch tracker
            batch_config = self.config.get('batch_tracking', {})
            batch_duration = batch_config.get('duration_minutes', 30.0)
            duplicate_threshold = batch_config.get('duplicate_threshold_seconds', 5.0)
            
            self.batch_tracker = BatchTracker(
                batch_duration_minutes=batch_duration,
                duplicate_threshold_seconds=duplicate_threshold
            )
            
            # Initialize data logger
            logging_config = self.config.get('logging', {})
            log_dir = logging_config.get('directory', 'logs')
            
            self.data_logger = DataLogger(
                log_dir=log_dir,
                enable_csv=logging_config.get('enable_csv', True),
                enable_json=logging_config.get('enable_json', True)
            )
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def start(self):
        """Start the smart trash system"""
        if self.running:
            self.logger.warning("System is already running")
            return
        
        try:
            self.initialize_components()
            self.running = True
            self.start_time = time.time()
            
            # Start camera recording
            self.camera.start_recording()
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            self.logger.info("Smart Trash System started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop the smart trash system"""
        if not self.running:
            return
        
        self.logger.info("Stopping Smart Trash System...")
        self.running = False
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        # Force complete current batch
        if self.batch_tracker:
            final_batch = self.batch_tracker.force_complete_batch()
            if final_batch and self.data_logger:
                self.data_logger.log_batch_summary(final_batch)
        
        # Release camera
        if self.camera:
            self.camera.release()
        
        # Log final statistics
        self._log_final_statistics()
        
        self.logger.info("Smart Trash System stopped")
    
    def _processing_loop(self):
        """Main processing loop for continuous detection"""
        self.logger.info("Starting detection processing loop")
        
        detection_interval = self.config.get('detection_interval_seconds', 1.0)
        last_detection_time = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # Check if it's time for next detection
                if (current_time - last_detection_time) < detection_interval:
                    time.sleep(0.1)
                    continue
                
                # Capture frame
                frame = self.camera.capture_frame()
                if frame is None:
                    self.logger.warning("Failed to capture frame")
                    time.sleep(1.0)
                    continue
                
                # Run detection
                detections = self.detector.detect_waste(frame)
                
                if detections:
                    self.logger.info(f"Detected {len(detections)} items")
                    
                    # Process each detection
                    for detection in detections:
                        self._process_detection(detection, frame.shape[:2])
                
                last_detection_time = current_time
                self.detection_count += 1
                
                # Check for batch completion
                self._check_batch_completion()
                
                # Periodic cleanup and statistics
                if self.detection_count % 100 == 0:
                    self._periodic_maintenance()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(1.0)
    
    def _process_detection(self, detection: Dict, image_shape: tuple):
        """Process a single detection"""
        try:
            # Estimate weight
            weight_estimate = self.weight_estimator.estimate_weight(
                detection, image_shape
            )
            
            # Add to current batch
            added = self.batch_tracker.add_detection(detection, weight_estimate)
            
            if added:
                # Log real-time detection
                batch_status = self.batch_tracker.get_current_batch_status()
                self.data_logger.log_real_time_detection(
                    detection, weight_estimate, batch_status['batch_id']
                )
                
                self.logger.info(
                    f"Added {detection['name']} ({weight_estimate['weight_grams']:.1f}g) "
                    f"to batch {batch_status['batch_id']}"
                )
            
        except Exception as e:
            self.logger.error(f"Failed to process detection: {e}")
    
    def _check_batch_completion(self):
        """Check if current batch should be completed"""
        try:
            batch_status = self.batch_tracker.get_current_batch_status()
            
            # Log batch completion
            if batch_status['will_complete_soon']:
                self.logger.info(
                    f"Batch {batch_status['batch_id']} will complete soon: "
                    f"{batch_status['remaining_minutes']:.1f} minutes remaining"
                )
            
            # Check if a batch was automatically completed
            recent_batches = self.batch_tracker.get_recent_batches(1)
            if recent_batches:
                last_batch = recent_batches[-1]
                # Check if this is a newly completed batch (within last 10 seconds)
                if (time.time() - last_batch.end_time) < 10.0:
                    self.data_logger.log_batch_summary(last_batch)
                    self.logger.info(f"Completed and logged batch {last_batch.batch_id}")
        
        except Exception as e:
            self.logger.error(f"Error checking batch completion: {e}")
    
    def _periodic_maintenance(self):
        """Perform periodic maintenance tasks"""
        try:
            # Log statistics
            self._log_system_statistics()
            
            # Cleanup old logs
            if self.detection_count % 1000 == 0:
                self.data_logger.cleanup_old_logs()
            
        except Exception as e:
            self.logger.error(f"Error in periodic maintenance: {e}")
    
    def _log_system_statistics(self):
        """Log current system statistics"""
        try:
            # Batch tracker statistics
            batch_stats = self.batch_tracker.get_statistics()
            
            # Camera information
            camera_info = self.camera.get_camera_info()
            
            # Current batch status
            batch_status = self.batch_tracker.get_current_batch_status()
            
            # System uptime
            uptime_minutes = (time.time() - self.start_time) / 60.0 if self.start_time else 0
            
            self.logger.info(
                f"System Stats - Uptime: {uptime_minutes:.1f}min, "
                f"Detections: {self.detection_count}, "
                f"Batches: {batch_stats['total_batches_completed']}, "
                f"Current batch: {batch_status['items_count']} items"
            )
            
        except Exception as e:
            self.logger.error(f"Error logging statistics: {e}")
    
    def _log_final_statistics(self):
        """Log final system statistics on shutdown"""
        try:
            if not self.start_time:
                return
            
            uptime_hours = (time.time() - self.start_time) / 3600.0
            batch_stats = self.batch_tracker.get_statistics()
            
            self.logger.info(
                f"Final Stats - Runtime: {uptime_hours:.2f}h, "
                f"Total detections: {self.detection_count}, "
                f"Completed batches: {batch_stats['total_batches_completed']}, "
                f"Total weight tracked: {batch_stats['total_weight_tracked_kg']:.2f}kg"
            )
            
        except Exception as e:
            self.logger.error(f"Error logging final statistics: {e}")
    
    def get_status(self) -> Dict:
        """Get current system status"""
        if not self.running:
            return {"status": "stopped"}
        
        try:
            batch_status = self.batch_tracker.get_current_batch_status()
            batch_stats = self.batch_tracker.get_statistics()
            camera_info = self.camera.get_camera_info()
            
            uptime = (time.time() - self.start_time) / 3600.0 if self.start_time else 0
            
            return {
                "status": "running",
                "uptime_hours": uptime,
                "detection_count": self.detection_count,
                "camera": camera_info,
                "current_batch": batch_status,
                "batch_statistics": batch_stats
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file or use defaults"""
    default_config = {
        "camera": {
            "source": "auto",
            "resolution": [640, 480],
            "fps": 30
        },
        "detector": {
            "model_path": None,
            "confidence_threshold": 0.5
        },
        "batch_tracking": {
            "duration_minutes": 30.0,
            "duplicate_threshold_seconds": 5.0
        },
        "logging": {
            "directory": "logs",
            "enable_csv": True,
            "enable_json": True
        },
        "detection_interval_seconds": 1.0,
        "log_level": "INFO"
    }
    
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            
            # Merge user config with defaults
            def merge_dicts(default, user):
                result = default.copy()
                for key, value in user.items():
                    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = merge_dicts(result[key], value)
                    else:
                        result[key] = value
                return result
            
            return merge_dicts(default_config, user_config)
        except Exception as e:
            print(f"Warning: Failed to load config {config_path}: {e}")
            print("Using default configuration")
    
    return default_config


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\nReceived shutdown signal. Stopping system...")
    global system
    if system:
        system.stop()
    sys.exit(0)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Smart Trash Detection System")
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--simulate', '-s', action='store_true', 
                       help='Force simulation mode')
    parser.add_argument('--video', '-v', help='Use video file as input source')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config based on arguments
    if args.simulate:
        config['camera']['source'] = 'simulated'
    elif args.video:
        config['camera']['source'] = args.video
    
    if args.debug:
        config['log_level'] = 'DEBUG'
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start system
    global system
    system = SmartTrashSystem(config)
    
    try:
        print("Starting Smart Trash Detection System...")
        print(f"Camera source: {config['camera']['source']}")
        print(f"Batch duration: {config['batch_tracking']['duration_minutes']} minutes")
        print("Press Ctrl+C to stop")
        
        system.start()
        
        # Keep main thread alive
        while system.running:
            time.sleep(1.0)
            
            # Optional: Print periodic status
            if system.detection_count > 0 and system.detection_count % 60 == 0:
                status = system.get_status()
                print(f"Status: {status['detection_count']} detections, "
                      f"uptime: {status['uptime_hours']:.1f}h")
    
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"System error: {e}")
    finally:
        if system:
            system.stop()


if __name__ == "__main__":
    main()