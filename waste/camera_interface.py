"""
Camera Interface for Raspberry Pi and Simulation
Supports Pi Camera, USB cameras, and simulated mode for development
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import logging
import time
import os
from pathlib import Path
import random

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

class CameraInterface:
    """
    Unified camera interface supporting:
    - Raspberry Pi Camera (via picamera2)
    - USB/Webcam (via OpenCV)
    - Simulated mode with sample images
    """
    
    def __init__(self, source='auto', resolution=(640, 480), fps=30):
        """
        Initialize camera interface
        
        Args:
            source: 'auto', 'pi', 'usb', 'simulated', video file path, or device index
            resolution: (width, height) tuple
            fps: Target frames per second
        """
        self.resolution = resolution
        self.fps = fps
        self.camera = None
        self.camera_type = None
        self.is_recording = False
        
        self.setup_logging()
        self.setup_sample_images()
        
        if source == 'auto':
            self.auto_detect_camera()
        else:
            self.setup_camera(source)
    
    def setup_logging(self):
        """Setup logging for camera interface"""
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def setup_sample_images(self):
        """Setup sample images for testing - REAL IMAGES ONLY"""
        self.sample_images = []
        
        # Load only existing real sample images
        sample_dir = Path(__file__).parent.parent / "data" / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing sample images (no synthetic generation)
        for img_path in sample_dir.glob("*.jpg"):
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.resize(img, self.resolution)
                    self.sample_images.append(img)
                    self.logger.info(f"Loaded real image: {img_path.name}")
            except Exception as e:
                self.logger.warning(f"Failed to load sample image {img_path}: {e}")
        
        # If no real images found, warn user
        if not self.sample_images:
            self.logger.warning("No real sample images found in data/samples/. Add .jpg files for simulation mode.")
            self.logger.warning("Simulation mode will return None frames until real images are added.")
    
    def auto_detect_camera(self):
        """Automatically detect available camera"""
        if PICAMERA2_AVAILABLE and self._test_pi_camera():
            self.setup_camera('pi')
        elif self._test_usb_camera():
            self.setup_camera('usb')
        else:
            self.logger.warning("No cameras detected, using simulation mode")
            self.setup_camera('simulated')
    
    def _test_pi_camera(self) -> bool:
        """Test if Pi camera is available"""
        try:
            picam2 = Picamera2()
            picam2.configure(picam2.create_preview_configuration())
            picam2.start()
            time.sleep(0.1)
            picam2.stop()
            picam2.close()
            return True
        except Exception:
            return False
    
    def _test_usb_camera(self) -> bool:
        """Test if USB camera is available"""
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                return ret and frame is not None
            return False
        except Exception:
            return False
    
    def setup_camera(self, source):
        """Setup camera based on source type"""
        try:
            if source == 'pi' and PICAMERA2_AVAILABLE:
                self._setup_pi_camera()
            elif source == 'usb' or isinstance(source, int):
                self._setup_usb_camera(source)
            elif isinstance(source, str) and (source.endswith('.mp4') or source.endswith('.avi') or source.endswith('.mov') or os.path.exists(source)):
                self._setup_video_file(source)
            else:
                self._setup_simulated_camera()
        except Exception as e:
            self.logger.error(f"Failed to setup camera {source}: {e}")
            self._setup_simulated_camera()
    
    def _setup_pi_camera(self):
        """Setup Raspberry Pi camera"""
        try:
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": self.resolution, "format": "RGB888"}
            )
            self.camera.configure(config)
            self.camera.start()
            self.camera_type = 'pi'
            self.logger.info("Pi Camera initialized successfully")
            time.sleep(2)  # Allow camera to warm up
        except Exception as e:
            self.logger.error(f"Pi Camera setup failed: {e}")
            raise
    
    def _setup_usb_camera(self, source='usb'):
        """Setup USB/webcam"""
        device_id = 0 if source == 'usb' else source
        try:
            self.camera = cv2.VideoCapture(device_id)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            
            if not self.camera.isOpened():
                raise RuntimeError("Failed to open camera")
            
            self.camera_type = 'usb'
            self.logger.info(f"USB Camera {device_id} initialized successfully")
        except Exception as e:
            self.logger.error(f"USB Camera setup failed: {e}")
            raise
    
    def _setup_simulated_camera(self):
        """Setup simulated camera mode"""
        self.camera = None
        self.camera_type = 'simulated'
        self.logger.info("Simulated camera mode initialized")
    
    def _setup_video_file(self, video_path):
        """Setup video file as camera source"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.camera = cv2.VideoCapture(video_path)
        if not self.camera.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")
        
        self.camera_type = 'video'
        self.video_path = video_path
        self.video_fps = self.camera.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.camera.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        
        self.logger.info(f"Video file initialized: {video_path}")
        self.logger.info(f"Video specs: {self.frame_count} frames at {self.video_fps:.1f} FPS")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame
        
        Returns:
            Image as numpy array or None if capture fails
        """
        try:
            if self.camera_type == 'pi':
                return self._capture_pi_frame()
            elif self.camera_type == 'usb':
                return self._capture_usb_frame()
            elif self.camera_type == 'video':
                return self._capture_video_frame()
            else:
                return self._capture_simulated_frame()
        except Exception as e:
            self.logger.error(f"Frame capture failed: {e}")
            return None
    
    def _capture_pi_frame(self) -> Optional[np.ndarray]:
        """Capture frame from Pi camera"""
        try:
            frame = self.camera.capture_array()
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.logger.error(f"Pi camera capture failed: {e}")
            return None
    
    def _capture_usb_frame(self) -> Optional[np.ndarray]:
        """Capture frame from USB camera"""
        try:
            ret, frame = self.camera.read()
            return frame if ret else None
        except Exception as e:
            self.logger.error(f"USB camera capture failed: {e}")
            return None
    
    def _capture_video_frame(self) -> Optional[np.ndarray]:
        """Capture frame from video file"""
        try:
            ret, frame = self.camera.read()
            if ret:
                self.current_frame += 1
                return frame
            else:
                # End of video - loop back to start
                self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame = 0
                ret, frame = self.camera.read()
                if ret:
                    self.current_frame += 1
                return frame if ret else None
        except Exception as e:
            self.logger.error(f"Video capture failed: {e}")
            return None
    
    def _capture_simulated_frame(self) -> Optional[np.ndarray]:
        """Get simulated frame from real images only"""
        if not self.sample_images:
            self.logger.warning("No real sample images available for simulation")
            return None
        
        # Return random real sample with slight variations
        base_img = random.choice(self.sample_images).copy()
        
        # Add minimal temporal variation to simulate slight camera movement
        noise = np.random.randint(-5, 5, base_img.shape, dtype=np.int16)
        varied_img = np.clip(base_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return varied_img
    
    def start_recording(self):
        """Start continuous recording mode"""
        self.is_recording = True
        self.logger.info("Started recording mode")
    
    def stop_recording(self):
        """Stop recording mode"""
        self.is_recording = False
        self.logger.info("Stopped recording mode")
    
    def get_camera_info(self) -> dict:
        """Get camera information"""
        info = {
            "type": self.camera_type,
            "resolution": self.resolution,
            "fps": self.fps,
            "is_recording": self.is_recording,
            "available": self.camera is not None or self.camera_type == 'simulated'
        }
        
        # Add video-specific information
        if self.camera_type == 'video':
            info.update({
                "video_path": getattr(self, 'video_path', None),
                "video_fps": getattr(self, 'video_fps', None),
                "frame_count": getattr(self, 'frame_count', None),
                "current_frame": getattr(self, 'current_frame', None)
            })
        
        return info
    
    def release(self):
        """Release camera resources"""
        try:
            if self.camera_type == 'pi' and self.camera:
                self.camera.stop()
                self.camera.close()
            elif (self.camera_type == 'usb' or self.camera_type == 'video') and self.camera:
                self.camera.release()
            
            self.camera = None
            self.logger.info("Camera resources released")
        except Exception as e:
            self.logger.error(f"Failed to release camera: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()


def test_camera_interface():
    """Test the camera interface"""
    print("Testing Camera Interface...")
    
    with CameraInterface(source='simulated') as camera:
        info = camera.get_camera_info()
        print(f"Camera Type: {info['type']}")
        print(f"Resolution: {info['resolution']}")
        print(f"Available: {info['available']}")
        
        # Capture a few test frames
        for i in range(3):
            frame = camera.capture_frame()
            if frame is not None:
                print(f"Frame {i+1}: {frame.shape}, dtype: {frame.dtype}")
            else:
                print(f"Frame {i+1}: Failed to capture")
            time.sleep(0.1)


if __name__ == "__main__":
    test_camera_interface()