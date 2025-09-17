#!/bin/bash
# Smart Trash Detection System - Raspberry Pi Setup Script
# Run this script on a fresh Raspberry Pi OS installation

set -e

echo "🗑️  Smart Trash Detection System - Raspberry Pi Setup"
echo "======================================================"

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo "⚠️  Warning: This script is designed for Raspberry Pi"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    cmake \
    build-essential \
    libopencv-dev \
    python3-opencv \
    libatlas-base-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libatlas-base-dev \
    libjasper-dev \
    libqtgui4 \
    libqt4-test \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libcanberra-gtk-module \
    libcanberra-gtk3-module

# Install Pi Camera dependencies (if available)
if command -v raspi-config >/dev/null 2>&1; then
    echo "📷 Installing Pi Camera dependencies..."
    sudo apt install -y \
        python3-picamera2 \
        python3-libcamera \
        python3-kms++
    
    # Enable camera interface
    echo "📷 Enabling camera interface..."
    sudo raspi-config nonint do_camera 0
fi

# Create project directory
PROJECT_DIR="/home/pi/smart-trash"
echo "📁 Creating project directory: $PROJECT_DIR"
sudo mkdir -p $PROJECT_DIR
sudo chown pi:pi $PROJECT_DIR

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
cd $PROJECT_DIR
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install \
    numpy==1.24.3 \
    opencv-python==4.8.0.74 \
    ultralytics==8.0.120 \
    torch==2.0.1 \
    torchvision==0.15.2 \
    Pillow==9.5.0 \
    PyYAML==6.0 \
    requests==2.31.0 \
    tqdm==4.65.0 \
    matplotlib==3.7.1 \
    seaborn==0.12.2 \
    pandas==2.0.3

# Install Pi-specific packages if available
if [ -f "/etc/rpi-issue" ]; then
    echo "🔧 Installing Raspberry Pi specific packages..."
    pip install RPi.GPIO gpiozero
fi

# Create directories
echo "📁 Creating project directories..."
mkdir -p logs data/samples scripts

# Create sample config file
echo "⚙️ Creating configuration file..."
cat > config.json << EOF
{
    "camera": {
        "source": "auto",
        "resolution": [640, 480],
        "fps": 10
    },
    "detector": {
        "model_path": null,
        "confidence_threshold": 0.5
    },
    "batch_tracking": {
        "duration_minutes": 30.0,
        "duplicate_threshold_seconds": 5.0
    },
    "logging": {
        "directory": "logs",
        "enable_csv": true,
        "enable_json": true
    },
    "detection_interval_seconds": 2.0,
    "log_level": "INFO"
}
EOF

# Create systemd service file
echo "🔧 Creating systemd service..."
sudo tee /etc/systemd/system/smart-trash.service > /dev/null << EOF
[Unit]
Description=Smart Trash Detection System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/venv/bin
ExecStart=$PROJECT_DIR/venv/bin/python main.py --config config.json
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Set permissions
echo "🔒 Setting permissions..."
sudo chmod 644 /etc/systemd/system/smart-trash.service
sudo systemctl daemon-reload

# Create convenience scripts
echo "📝 Creating convenience scripts..."

# Start script
cat > start.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python main.py --config config.json
EOF

# Test script
cat > test.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python main.py --config config.json --simulate --debug
EOF

# Status script
cat > status.sh << 'EOF'
#!/bin/bash
echo "Smart Trash System Status:"
sudo systemctl status smart-trash
echo ""
echo "Recent logs:"
sudo journalctl -u smart-trash -n 20 --no-pager
EOF

chmod +x start.sh test.sh status.sh

# Create update script
cat > update.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
echo "Updating Smart Trash Detection System..."

# Pull latest code (if using git)
if [ -d ".git" ]; then
    git pull
fi

# Update Python dependencies
source venv/bin/activate
pip install --upgrade ultralytics torch torchvision

echo "Update complete!"
EOF

chmod +x update.sh

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy your source code to: $PROJECT_DIR"
echo "2. Test the system: ./test.sh"
echo "3. Start manually: ./start.sh"
echo "4. Enable auto-start: sudo systemctl enable smart-trash"
echo "5. Start service: sudo systemctl start smart-trash"
echo "6. Check status: ./status.sh"
echo ""
echo "Configuration file: config.json"
echo "Logs directory: logs/"
echo ""

if [ -f "/etc/rpi-issue" ]; then
    echo "📷 Camera setup:"
    echo "- Pi Camera should be automatically detected"
    echo "- USB camera will be used as fallback"
    echo "- Simulation mode available for testing"
    echo ""
    echo "⚡ Performance tips:"
    echo "- Consider overclocking for better performance"
    echo "- GPU memory split: sudo raspi-config -> Advanced -> Memory Split -> 128"
    echo "- Use fast SD card (Class 10, A1/A2)"
fi

echo "🎉 Smart Trash Detection System is ready to use!"