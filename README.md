# Smart Trash Detection System


## 📁 Project Structure

## 🎯 Overview

Real-time waste detection and classification using:
- **YOLOv11 model** for trash detection (bottles, cans, plastic bags, etc.)
- **CNN model** for fruit classification (apples, bananas, oranges, etc.)
- **Dual-model architecture** supporting both inorganic and organic waste
- **30-minute batch tracking** with duplicate detection
- **Weight estimation** and disposal method classification


### Detection Pipeline
1. **YOLO Model**: Detects trash objects with bounding boxes
2. **CNN Model**: Classifies fruits from full-frame images
3. **Weight Estimator**: Estimates weight based on object dimensions/type
4. **Batch Tracker**: Groups detections into 30-minute batches
5. **Data Logger**: Saves results to CSV/JSON files

### Supported Categories
- **Trash**: Bottles, cans, plastic bags, caps, paper, cigarettes (→ recycling/waste) 18 classes.
- **Fruits**: 131+ fruit types including apples, bananas, citrus (→ composting)

## 🚀 Quick Start

### 1. Raspberry Pi Setup
```bash
# Download and run setup script
wget https://your-repo/scripts/install_pi.sh
chmod +x install_pi.sh
sudo ./install_pi.sh

# Copy your project files to /home/pi/smart-trash/
```

### 2. Run System
```bash
# Start normally
./scripts/run.sh

# Test with simulation
./scripts/run.sh --simulate

# Run with video file
./scripts/run.sh --video data/samples/trash.mp4

# Debug mode
./scripts/run.sh --debug
```

### 3. View Results
```bash
# Real-time log
tail -f logs/realtime_detections.jsonl

# Batch summaries
cat logs/waste_batches.csv
```

## 📊 Example Output

```
2025-09-18 07:46:35 - INFO - Detected 2 items
2025-09-18 07:46:35 - INFO - Added Bottle cap (4.0g) to batch batch_20250918_074635
2025-09-18 07:46:35 - INFO - Added Chestnut (2733.8g) to batch batch_20250918_074635
```

## ⚙️ Configuration

Edit detection parameters in the code or via command line:
```bash
# Custom confidence threshold
python main.py --confidence 0.6

# Different batch duration
# Edit batch_tracking config in main.py
```

## 📁 Project Structure

```
smart-trash/
├── main.py                     # Main application
├── scripts/
│   ├── install_pi.sh           # Pi setup script
│   └── run.sh                  # Run script
├── waste/
│   ├── camera_interface.py     # Camera/video handling
│   ├── models/
│   │   ├── detector.py         # Dual YOLO+CNN detection
│   │   └── weight_estimator.py # Weight estimation
│   └── utils/
│       ├── batch_tracker.py    # Batch management
│       └── data_logger.py      # Data logging
├── data/
│   ├── samples/                # Test videos/images
│   ├── fruits/                 # CNN model files
│   └── runs/detect/            # YOLO model files
└── logs/                       # Output logs
```

## 🔧 Model Performance

### Current Status
- **Trash Detection (YOLO)**: Excellent performance (70-85% confidence)
- **Fruit Classification (CNN)**: Lower performance (~1.5% confidence) because of the issue below...

### Known Limitation
The CNN was trained on cropped fruit images but receives full camera frames, causing confidence issues. This is a common dataset-vs-reality mismatch in computer vision.



## 🔍 Troubleshooting

### No Detections
```bash
# Check camera
python -c "import cv2; print(cv2.VideoCapture(0).read())"

# Test detection models
python -m waste.models.detector

# Verify video file
python main.py --video data/samples/trash.mp4 --debug
```


## 📈 Data Analysis

Output files contain detection details:
- `realtime_detections.jsonl`: Individual detections with timestamps
- `waste_batches.csv`: 30-minute batch summaries
- `waste_items.csv`: Individual item records





## 📊 Output Data

### CSV Files
- `logs/waste_batches.csv` - Batch summaries
- `logs/waste_items.csv` - Individual item records

### JSON Files
- `logs/waste_batches.json` - Complete batch data
- `logs/waste_log_YYYYMMDD.json` - Daily summaries
- `logs/realtime_detections.jsonl` - Real-time detection stream

### Sample Output
```csv
batch_id,start_time,end_time,unique_items,total_weight_grams,organic_weight_g,plastic_weight_g
batch_20241217_143022,1703257822,1703259622,5,245.5,180.0,65.5
```

## 🎛️ System Control

### Manual Control
```bash
# Start system
python main.py --config config.json

# Simulation mode
python main.py --config config.json --simulate

# Debug mode
python main.py --config config.json --debug
```

### Service Control (Pi deployment)
```bash
# Check status
sudo systemctl status smart-trash

# View logs
sudo journalctl -u smart-trash -f

# Start/stop
sudo systemctl start smart-trash
sudo systemctl stop smart-trash

# Enable/disable auto-start
sudo systemctl enable smart-trash
sudo systemctl disable smart-trash
```




## 🧪 Testing

### Component Tests
```bash
# Test individual components
python -m waste.camera_interface
python -m waste.models.detector
python -m waste.models.weight_estimator
python -m waste.utils.batch_tracker
python -m waste.utils.data_logger
```





### Log Analysis
```bash
# System logs
sudo journalctl -u smart-trash --since "1 hour ago"

# Application logs
tail -f logs/smart_trash.log

# Real-time detections
tail -f logs/realtime_detections.jsonl
```

## 🛠️ Development

### Extending Weight Estimation
Edit `waste/models/weight_estimator.py`:
```python
# Add new items to weight database
self.item_weights["new_item"] = 25.0  # grams

# Add new material densities
self.material_densities["new_material"] = 1.2  # g/cm³
```

### Custom Batch Logic
Modify `waste/utils/batch_tracker.py` for different timing or grouping logic.

