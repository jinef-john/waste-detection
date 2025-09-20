# Smart Trash System - Client Instructions

## For Raspberry Pi Users

### 🔧 Setup (One Time)
```bash
# 1. give permission
chmod +x install_pi.sh

# 2. Run setup
sudo ./install_pi.sh

# 3. Copy project files to /home/pi/smart-trash/
```
```bash
cd /home/pi/smart-trash

# Start system
./scripts/run.sh

# Test mode
./scripts/run.sh --simulate

# Video file
./scripts/run.sh --video data/samples/trash.mp4
```

### 📊 View Results
```bash
# Real-time log
tail -f logs/realtime_detections.jsonl

# System status
sudo systemctl status smart-trash
```

### 🔧 Auto-Start (Optional)
```bash
sudo systemctl enable smart-trash
sudo systemctl start smart-trash
```
