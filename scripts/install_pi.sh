#!/bin/bash
# Smart Trash Detection System - Raspberry Pi Setup

set -e

echo "🗑️  Smart Trash Detection System - Pi Setup"
echo "==========================================="

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
echo "📦 Updating system..."
sudo apt update && sudo apt upgrade -y

# Install dependencies
echo "🔧 Installing dependencies..."
sudo apt install -y \
    python3 python3-pip python3-venv \
    git cmake build-essential \
    python3-opencv libopencv-dev \
    python3-picamera2 python3-libcamera \
    libhdf5-dev libatlas-base-dev

# Enable camera
echo "📷 Enabling camera..."
sudo raspi-config nonint do_camera 0

# Setup project
PROJECT_DIR="/home/pi/smart-trash"
echo "📁 Setting up project in: $PROJECT_DIR"
sudo mkdir -p $PROJECT_DIR
sudo chown pi:pi $PROJECT_DIR
cd $PROJECT_DIR

# Clone or setup code
if [ ! -d ".git" ]; then
    echo "� Getting project code..."
    git clone https://github.com/jinef-john/waste-detection.git .
    echo "⚠️  Please copy your project files to $PROJECT_DIR"
fi

# Create virtual environment
echo "� Creating Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install Python packages
echo "📦 Installing Python packages..."
pip install -r requirements.txt

# Create directories
mkdir -p logs data/samples

# Create systemd service
echo "🔧 Creating system service..."
sudo tee /etc/systemd/system/smart-trash.service > /dev/null << EOF
[Unit]
Description=Smart Trash Detection System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/venv/bin
ExecStart=$PROJECT_DIR/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload

echo ""
echo "✅ Setup Complete!"
echo ""
echo "Usage:"
echo "  Start:    ./scripts/run.sh"
echo "  Test:     ./scripts/run.sh --simulate"
echo "  Service:  sudo systemctl start smart-trash"
echo "  Auto-start: sudo systemctl enable smart-trash"
echo ""
echo "Done"