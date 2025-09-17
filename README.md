# Smart Trash Detection System

A standalone Raspberry Pi 4B system for intelligent waste monitoring using computer vision.

## 🎯 Project Overview

This system provides real-time waste detection and logging using:
- **YOLOv8-nano** model optimized for Pi 4B performance (≥1 FPS)
- **30-minute batch tracking** with duplicate detection
- **Weight estimation** using pre-mapped averages and volume heuristics
- **Local CSV/JSON logging** (no cloud dependencies)
- **Pi Camera + USB camera support** with simulation mode

### Supported Datasets
- **TACO** (inorganic litter detection)
- **Fruits-360** (fruit/organic waste detection)

## 📁 Project Structure

```
smart-trash/
├── main.py                  # Main application entry point
├── requirements.txt         # Python dependencies
├── config.json             # Runtime configuration
├── waste/                  # Core package
│   ├── __init__.py
│   ├── camera_interface.py # Camera handling
│   ├── models/             # ML components
│   │   ├── detector.py     # YOLOv8 waste detection
│   │   └── weight_estimator.py # Weight estimation
│   └── utils/              # Utilities
│       ├── batch_tracker.py   # 30-min batch logic
│       └── data_logger.py     # CSV/JSON logging
├── data/                   # Data storage
│   └── samples/            # Sample images
├── logs/                   # Output logs
├── scripts/                # Installation & setup
│   ├── install_pi.sh       # Pi installation
│   ├── setup_dev.sh        # Development setup
│   └── create_image.sh     # SD card image builder
└── README.md               # This file
```

## 🚀 Quick Start

### Option 1: Raspberry Pi Deployment

1. **Automated Installation:**
   ```bash
   curl -sSL https://raw.githubusercontent.com/your-repo/smart-trash/main/scripts/install_pi.sh | bash
   ```

2. **Manual Installation:**
   ```bash
   git clone <repository-url>
   cd smart-trash
   chmod +x scripts/install_pi.sh
   sudo ./scripts/install_pi.sh
   ```

3. **Test the system:**
   ```bash
   ./test.sh
   ```

4. **Start automatically:**
   ```bash
   sudo systemctl enable smart-trash
   sudo systemctl start smart-trash
   ```

### Option 2: Development Setup

1. **Setup development environment:**
   ```bash
   git clone <repository-url>
   cd smart-trash
   chmod +x scripts/setup_dev.sh
   ./scripts/setup_dev.sh
   ```

2. **Run in development mode:**
   ```bash
   ./run_dev.sh
   ```

3. **Run tests:**
   ```bash
   ./run_tests.sh
   ```

### Option 3: Using Pre-built SD Image

1. Download the pre-built SD card image
2. Flash to SD card using Raspberry Pi Imager or `dd`
3. Insert into Pi 4B and power on
4. System starts automatically

## ⚙️ Configuration

Edit `config.json` to customize behavior:

```json
{
    "camera": {
        "source": "auto",           // "auto", "pi", "usb", "simulated"
        "resolution": [640, 480],   // Camera resolution
        "fps": 10                   // Target FPS (Pi 4B limit)
    },
    "detector": {
        "model_path": null,         // Custom model path (null = YOLOv8n)
        "confidence_threshold": 0.5 // Detection confidence threshold
    },
    "batch_tracking": {
        "duration_minutes": 30.0,   // Batch duration
        "duplicate_threshold_seconds": 5.0 // Duplicate detection window
    },
    "logging": {
        "directory": "logs",        // Log output directory
        "enable_csv": true,         // Enable CSV logging
        "enable_json": true         // Enable JSON logging
    },
    "detection_interval_seconds": 2.0, // Time between detections
    "log_level": "INFO"             // Logging level
}
```

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

## 🔧 Hardware Requirements

### Minimum (Raspberry Pi 4B)
- **RAM:** 4GB recommended (2GB minimum)
- **Storage:** 32GB microSD (Class 10, A1/A2 recommended)
- **Camera:** Pi Camera Module or USB webcam
- **Power:** 5V 3A power supply

### Recommended Setup
- **Pi 4B 8GB** for best performance
- **64GB microSD** for extended logging
- **Pi Camera Module v2** for optimal integration
- **Heatsink/fan** for thermal management
- **UPS/battery backup** for continuous operation

## 📈 Performance

### Expected Performance (Pi 4B)
- **Detection Speed:** 1-3 FPS (YOLOv8-nano)
- **Memory Usage:** ~500MB
- **CPU Usage:** 40-60% (single core)
- **Storage:** ~10MB/day (typical usage)

### Optimization Tips
```bash
# GPU memory split
sudo raspi-config # Advanced Options → Memory Split → 128

# Overclocking (optional)
# Add to /boot/config.txt:
arm_freq=1750
over_voltage=4
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

### Performance Benchmark
```bash
./benchmark.sh
```

### Full System Test
```bash
# 30-second test with simulated data
timeout 30 python main.py --config config_test.json --simulate
```

## 📋 Supported Waste Items

### Organic Waste
- Fresh fruits (apple, banana, orange)
- Rotten fruits and peels
- Food scraps and leftovers

### Plastic Waste
- Bottles and containers
- Bags and packaging
- Utensils and straws

### Metal Waste
- Aluminum cans
- Small metal objects

### Glass Waste
- Bottles and jars
- Drinking glasses

### Paper Waste
- Cups and containers
- Cardboard packaging
- Tissues and napkins

## 🔍 Troubleshooting

### Common Issues

**Camera not detected:**
```bash
# Check camera connection
vcgencmd get_camera

# Test Pi camera
libcamera-hello --display 0

# Test USB camera
v4l2-ctl --list-devices
```

**Low detection performance:**
```bash
# Check system resources
htop

# Reduce resolution in config.json
"resolution": [320, 240]

# Increase detection interval
"detection_interval_seconds": 3.0
```

**Storage full:**
```bash
# Clean old logs
./clean.sh

# Check disk usage
df -h

# Archive old data
tar -czf logs_backup.tar.gz logs/
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

### Adding Custom Models
1. Train YOLOv8 model on TACO + Fruits-360 datasets
2. Export to ONNX or PyTorch format
3. Place model file in `data/models/`
4. Update `config.json` with model path

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

## 📚 Data Analysis

Use the included Jupyter notebook for data analysis:
```bash
source venv/bin/activate
jupyter notebook analysis.ipynb
```

Or analyze CSV data directly:
```python
import pandas as pd
df = pd.read_csv('logs/waste_batches.csv')
print(df.groupby('organic_weight_g').sum())
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Run tests: `./run_tests.sh`
4. Commit changes: `git commit -am 'Add feature'`
5. Push branch: `git push origin feature-name`
6. Submit pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🆘 Support

- **Documentation:** [Full docs](docs/)
- **Issues:** [GitHub Issues](https://github.com/your-repo/smart-trash/issues)
- **Discussions:** [GitHub Discussions](https://github.com/your-repo/smart-trash/discussions)

## 🏆 Acknowledgments

- **TACO Dataset:** [TACO: Trash Annotations in Context](http://tacodataset.org/)
- **Fruits-360 Dataset:** [Kaggle Fruits-360](https://www.kaggle.com/moltean/fruits)
- **Ultralytics YOLOv8:** [YOLOv8 Repository](https://github.com/ultralytics/ultralytics)
- **Raspberry Pi Foundation** for excellent hardware platform

---

## 📊 Example Output

After running for a day, you might see logs like this:

```
2024-12-17 14:30:22 - smart_trash - INFO - Smart Trash System started successfully
2024-12-17 14:31:15 - smart_trash - INFO - Added plastic_bottle (25.0g) to batch_20241217_143022
2024-12-17 14:32:45 - smart_trash - INFO - Added apple_core (20.0g) to batch_20241217_143022
2024-12-17 15:00:22 - smart_trash - INFO - Completed batch batch_20241217_143022: 8 items, 180.5g
```

Ready to deploy your smart trash monitoring system! 🚀