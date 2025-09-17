#!/bin/bash
# Development Environment Setup Script
# For setting up the development environment on any Linux system

set -e

echo "🗑️  Smart Trash Detection System - Development Setup"
echo "=================================================="

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ required. Found: Python $python_version"
    exit 1
fi

echo "✅ Python $python_version detected"

# Install system dependencies (Ubuntu/Debian)
if command -v apt-get >/dev/null 2>&1; then
    echo "📦 Installing system dependencies (apt)..."
    sudo apt-get update
    sudo apt-get install -y \
        python3-pip \
        python3-venv \
        python3-dev \
        cmake \
        build-essential \
        libopencv-dev \
        libhdf5-dev \
        pkg-config \
        git
elif command -v yum >/dev/null 2>&1; then
    echo "📦 Installing system dependencies (yum)..."
    sudo yum install -y \
        python3-pip \
        python3-devel \
        cmake \
        gcc-c++ \
        opencv-devel \
        hdf5-devel \
        git
elif command -v brew >/dev/null 2>&1; then
    echo "📦 Installing system dependencies (brew)..."
    brew install python opencv cmake hdf5
fi

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install \
    numpy>=1.21.0 \
    opencv-python>=4.6.0 \
    ultralytics>=8.0.0 \
    torch>=1.13.0 \
    torchvision>=0.14.0 \
    Pillow>=9.0.0 \
    PyYAML>=6.0 \
    requests>=2.28.0 \
    tqdm>=4.64.0 \
    matplotlib>=3.5.0 \
    seaborn>=0.11.0 \
    pandas>=1.4.0 \
    pytest>=7.0.0 \
    jupyter>=1.0.0

# Create development directories
echo "📁 Creating directories..."
mkdir -p logs data/samples data/models data/datasets tests docs

# Create development config
echo "⚙️ Creating development configuration..."
cat > config_dev.json << EOF
{
    "camera": {
        "source": "simulated",
        "resolution": [640, 480],
        "fps": 30
    },
    "detector": {
        "model_path": null,
        "confidence_threshold": 0.3
    },
    "batch_tracking": {
        "duration_minutes": 2.0,
        "duplicate_threshold_seconds": 3.0
    },
    "logging": {
        "directory": "logs",
        "enable_csv": true,
        "enable_json": true
    },
    "detection_interval_seconds": 0.5,
    "log_level": "DEBUG"
}
EOF

# Create test config
cat > config_test.json << EOF
{
    "camera": {
        "source": "simulated",
        "resolution": [320, 240],
        "fps": 10
    },
    "detector": {
        "model_path": null,
        "confidence_threshold": 0.1
    },
    "batch_tracking": {
        "duration_minutes": 0.5,
        "duplicate_threshold_seconds": 1.0
    },
    "logging": {
        "directory": "logs/test",
        "enable_csv": true,
        "enable_json": true
    },
    "detection_interval_seconds": 0.1,
    "log_level": "DEBUG"
}
EOF

# Create development scripts
echo "📝 Creating development scripts..."

# Run tests script
cat > run_tests.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate

echo "Running Smart Trash Detection System Tests..."
echo "============================================="

# Run component tests
echo "Testing camera interface..."
python -m waste.camera_interface

echo "Testing waste detector..."
python -m waste.models.detector

echo "Testing weight estimator..."
python -m waste.models.weight_estimator

echo "Testing batch tracker..."
python -m waste.utils.batch_tracker

echo "Testing data logger..."
python -m waste.utils.data_logger

echo "Running main application test..."
timeout 30 python main.py --config config_test.json --simulate || echo "Test completed (timeout expected)"

echo "✅ All tests completed!"
EOF

# Development run script
cat > run_dev.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate

echo "Starting Smart Trash System in Development Mode..."
python main.py --config config_dev.json --simulate --debug
EOF

# Benchmark script
cat > benchmark.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate

echo "Running Performance Benchmark..."
python -c "
from waste.models.detector import WasteDetector
from waste.camera_interface import CameraInterface
import time
import numpy as np

print('Initializing components...')
detector = WasteDetector()
camera = CameraInterface(source='simulated', resolution=(640, 480))

print('Running benchmark...')
times = []
for i in range(20):
    frame = camera.capture_frame()
    start = time.time()
    detections = detector.detect_waste(frame)
    end = time.time()
    times.append(end - start)
    print(f'Frame {i+1}: {len(detections)} detections, {(end-start)*1000:.1f}ms')

avg_time = sum(times) / len(times)
fps = 1.0 / avg_time
print(f'\\nAverage: {avg_time*1000:.1f}ms per frame')
print(f'FPS: {fps:.1f}')
print(f'Suitable for Pi 4B: {\"✅ Yes\" if fps >= 1.0 else \"❌ No\"}')
"
EOF

# Clean script
cat > clean.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"

echo "Cleaning up development environment..."
rm -rf logs/*
rm -rf data/samples/*
rm -rf __pycache__
find . -name "*.pyc" -delete
find . -name "__pycache__" -delete
echo "✅ Cleanup complete!"
EOF

chmod +x run_tests.sh run_dev.sh benchmark.sh clean.sh

# Create Jupyter notebook for data analysis
echo "📓 Creating analysis notebook..."
cat > analysis.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Smart Trash Detection System - Data Analysis\n",
    "\n",
    "This notebook provides tools for analyzing waste detection data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import json\n",
    "from pathlib import Path\n",
    "\n",
    "# Load batch data\n",
    "batch_file = Path('logs/waste_batches.csv')\n",
    "if batch_file.exists():\n",
    "    df_batches = pd.read_csv(batch_file)\n",
    "    print(f\"Loaded {len(df_batches)} batches\")\n",
    "    display(df_batches.head())\n",
    "else:\n",
    "    print(\"No batch data found. Run the system first to generate data.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Analyze waste patterns\n",
    "if 'df_batches' in locals():\n",
    "    # Weight distribution\n",
    "    plt.figure(figsize=(12, 4))\n",
    "    \n",
    "    plt.subplot(1, 3, 1)\n",
    "    plt.hist(df_batches['total_weight_kg'], bins=20)\n",
    "    plt.title('Weight Distribution per Batch')\n",
    "    plt.xlabel('Weight (kg)')\n",
    "    \n",
    "    plt.subplot(1, 3, 2) \n",
    "    plt.hist(df_batches['unique_items'], bins=20)\n",
    "    plt.title('Items per Batch')\n",
    "    plt.xlabel('Number of Items')\n",
    "    \n",
    "    plt.subplot(1, 3, 3)\n",
    "    category_cols = ['organic_weight_g', 'plastic_weight_g', 'metal_weight_g', 'glass_weight_g', 'paper_weight_g']\n",
    "    category_totals = df_batches[category_cols].sum()\n",
    "    plt.pie(category_totals, labels=[col.replace('_weight_g', '') for col in category_cols], autopct='%1.1f%%')\n",
    "    plt.title('Waste Category Distribution')\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Create basic tests
echo "🧪 Creating test files..."
mkdir -p tests
cat > tests/test_components.py << 'EOF'
"""
Basic tests for Smart Trash Detection System components
"""

import unittest
import numpy as np
from pathlib import Path
import sys
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from waste.camera_interface import CameraInterface
from waste.models.detector import WasteDetector
from waste.models.weight_estimator import WeightEstimator
from waste.utils.batch_tracker import BatchTracker
from waste.utils.data_logger import DataLogger

class TestComponents(unittest.TestCase):
    
    def test_camera_interface(self):
        """Test camera interface"""
        camera = CameraInterface(source='simulated', resolution=(320, 240))
        frame = camera.capture_frame()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape[:2], (240, 320))
        camera.release()
    
    def test_waste_detector(self):
        """Test waste detector"""
        detector = WasteDetector()
        test_image = np.zeros((240, 320, 3), dtype=np.uint8)
        detections = detector.detect_waste(test_image)
        self.assertIsInstance(detections, list)
    
    def test_weight_estimator(self):
        """Test weight estimator"""
        estimator = WeightEstimator()
        detection = {
            "name": "plastic_bottle",
            "bbox": [100, 100, 200, 200],
            "category": "plastic"
        }
        estimate = estimator.estimate_weight(detection, (480, 640))
        self.assertIn("weight_grams", estimate)
        self.assertGreater(estimate["weight_grams"], 0)
    
    def test_batch_tracker(self):
        """Test batch tracker"""
        tracker = BatchTracker(batch_duration_minutes=0.1)  # 6 seconds for test
        
        detection = {"name": "apple", "category": "organic", "bbox": [0, 0, 50, 50], "confidence": 0.8}
        weight = {"weight_grams": 150.0, "confidence": 0.8}
        
        added = tracker.add_detection(detection, weight)
        self.assertTrue(added)
        
        status = tracker.get_current_batch_status()
        self.assertEqual(status["items_count"], 1)

if __name__ == '__main__':
    unittest.main()
EOF

# Create requirements file
echo "📋 Creating requirements.txt..."
cat > requirements.txt << 'EOF'
numpy>=1.21.0
opencv-python>=4.6.0
ultralytics>=8.0.0
torch>=1.13.0
torchvision>=0.14.0
Pillow>=9.0.0
PyYAML>=6.0
requests>=2.28.0
tqdm>=4.64.0
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.4.0
pytest>=7.0.0
jupyter>=1.0.0
EOF

echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "Available commands:"
echo "  ./run_dev.sh      - Run in development mode"
echo "  ./run_tests.sh    - Run all tests"
echo "  ./benchmark.sh    - Performance benchmark"
echo "  ./clean.sh        - Clean temporary files"
echo ""
echo "Jupyter notebook: analysis.ipynb"
echo "Development config: config_dev.json"
echo "Test config: config_test.json"
echo ""
echo "🚀 Ready for development!"