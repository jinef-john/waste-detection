#!/bin/bash
# Flash SD Card Image Creation Script
# Creates a deployable SD card image for Raspberry Pi

set -e

echo "🗑️  Smart Trash Detection System - SD Card Image Builder"
echo "======================================================="

# Check if running on appropriate system
if ! command -v debootstrap >/dev/null 2>&1; then
    echo "❌ This script requires debootstrap. Install with:"
    echo "   sudo apt install debootstrap qemu-user-static"
    exit 1
fi

# Configuration
IMAGE_NAME="smart-trash-pi.img"
IMAGE_SIZE="4G"
MOUNT_DIR="/tmp/smart-trash-build"
PROJECT_NAME="smart-trash"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ This script must be run as root (use sudo)"
    exit 1
fi

echo "📁 Creating image file..."
dd if=/dev/zero of=$IMAGE_NAME bs=1 count=0 seek=$IMAGE_SIZE

echo "🔧 Setting up loop device..."
LOOP_DEVICE=$(losetup --find --show $IMAGE_NAME)

echo "💾 Creating partitions..."
parted $LOOP_DEVICE mklabel msdos
parted $LOOP_DEVICE mkpart primary fat32 0% 512MB
parted $LOOP_DEVICE mkpart primary ext4 512MB 100%

echo "🔄 Re-reading partition table..."
partprobe $LOOP_DEVICE
sleep 2

# Get partition devices
BOOT_PARTITION="${LOOP_DEVICE}p1"
ROOT_PARTITION="${LOOP_DEVICE}p2"

echo "💾 Formatting partitions..."
mkfs.vfat -F 32 -n BOOT $BOOT_PARTITION
mkfs.ext4 -L rootfs $ROOT_PARTITION

echo "📁 Creating mount directories..."
mkdir -p $MOUNT_DIR
mkdir -p $MOUNT_DIR/boot

echo "🔗 Mounting partitions..."
mount $ROOT_PARTITION $MOUNT_DIR
mount $BOOT_PARTITION $MOUNT_DIR/boot

echo "📦 Installing base Raspberry Pi OS..."
# Download and install minimal Raspberry Pi OS
debootstrap --arch armhf --foreign bullseye $MOUNT_DIR http://archive.raspberrypi.org/debian/

# Set up QEMU for ARM emulation
cp /usr/bin/qemu-arm-static $MOUNT_DIR/usr/bin/

# Complete the installation
chroot $MOUNT_DIR /debootstrap/debootstrap --second-stage

echo "⚙️ Configuring system..."
# Basic system configuration
cat > $MOUNT_DIR/etc/hostname << EOF
smart-trash-pi
EOF

cat > $MOUNT_DIR/etc/hosts << EOF
127.0.0.1       localhost
127.0.1.1       smart-trash-pi
EOF

# Enable SSH
chroot $MOUNT_DIR systemctl enable ssh

# Create pi user
chroot $MOUNT_DIR useradd -m -s /bin/bash pi
chroot $MOUNT_DIR usermod -aG sudo pi
echo "pi:raspberry" | chroot $MOUNT_DIR chpasswd

# Install essential packages
cat > $MOUNT_DIR/install_packages.sh << 'EOF'
#!/bin/bash
apt update
apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    cmake \
    build-essential \
    libopencv-dev \
    python3-opencv \
    python3-picamera2 \
    openssh-server \
    systemd \
    sudo \
    nano \
    htop \
    curl \
    wget

# Clean up
apt autoremove -y
apt autoclean
EOF

chmod +x $MOUNT_DIR/install_packages.sh
chroot $MOUNT_DIR /install_packages.sh
rm $MOUNT_DIR/install_packages.sh

echo "📦 Installing Smart Trash Detection System..."
# Copy project files
mkdir -p $MOUNT_DIR/home/pi/smart-trash
cp -r ../waste $MOUNT_DIR/home/pi/smart-trash/
cp ../main.py $MOUNT_DIR/home/pi/smart-trash/
cp ../requirements.txt $MOUNT_DIR/home/pi/smart-trash/
cp -r ../scripts $MOUNT_DIR/home/pi/smart-trash/

# Create virtual environment and install dependencies
cat > $MOUNT_DIR/setup_project.sh << 'EOF'
#!/bin/bash
cd /home/pi/smart-trash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create configuration
cat > config.json << 'EOFCONFIG'
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
EOFCONFIG

# Create directories
mkdir -p logs data/samples

# Set ownership
chown -R pi:pi /home/pi/smart-trash
EOF

chmod +x $MOUNT_DIR/setup_project.sh
chroot $MOUNT_DIR /setup_project.sh
rm $MOUNT_DIR/setup_project.sh

# Install systemd service
cat > $MOUNT_DIR/etc/systemd/system/smart-trash.service << 'EOF'
[Unit]
Description=Smart Trash Detection System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/smart-trash
Environment=PATH=/home/pi/smart-trash/venv/bin
ExecStart=/home/pi/smart-trash/venv/bin/python main.py --config config.json
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable the service
chroot $MOUNT_DIR systemctl enable smart-trash

echo "🚀 Creating boot configuration..."
# Boot configuration
cat > $MOUNT_DIR/boot/config.txt << 'EOF'
# Smart Trash Pi Configuration

# Enable camera
start_x=1
gpu_mem=128

# Enable I2C and SPI if needed
dtparam=i2c_arm=on
dtparam=spi=on

# Performance settings
arm_freq=1500
over_voltage=2
temp_limit=80

# GPIO settings
gpio=2-27=op
EOF

cat > $MOUNT_DIR/boot/cmdline.txt << 'EOF'
console=serial0,115200 console=tty1 root=PARTUUID=12345678-02 rootfstype=ext4 elevator=deadline fsck.repair=yes rootwait modules-load=dwc2,g_ether
EOF

echo "📄 Creating documentation..."
cat > $MOUNT_DIR/home/pi/README.txt << 'EOF'
Smart Trash Detection System - Raspberry Pi Image
================================================

This SD card contains a pre-configured Raspberry Pi system for waste detection.

Quick Start:
1. Insert SD card and power on Raspberry Pi
2. System will start automatically
3. Check status: sudo systemctl status smart-trash
4. View logs: sudo journalctl -u smart-trash -f

Configuration:
- Main config: /home/pi/smart-trash/config.json
- Logs: /home/pi/smart-trash/logs/
- Service: /etc/systemd/system/smart-trash.service

Manual Control:
- Start: sudo systemctl start smart-trash
- Stop: sudo systemctl stop smart-trash
- Restart: sudo systemctl restart smart-trash
- Disable auto-start: sudo systemctl disable smart-trash

Troubleshooting:
- Test camera: cd /home/pi/smart-trash && ./test.sh
- Check system: ./status.sh
- Update system: ./update.sh

Default Login:
- Username: pi
- Password: raspberry
- SSH: Enabled

For support, see documentation in /home/pi/smart-trash/
EOF

chown pi:pi $MOUNT_DIR/home/pi/README.txt

echo "🧹 Cleaning up..."
# Remove QEMU
rm $MOUNT_DIR/usr/bin/qemu-arm-static

# Clean package cache
chroot $MOUNT_DIR apt clean

echo "💾 Finalizing image..."
# Unmount filesystems
sync
umount $MOUNT_DIR/boot
umount $MOUNT_DIR
rmdir $MOUNT_DIR

# Detach loop device
losetup -d $LOOP_DEVICE

# Compress image
echo "🗜️ Compressing image..."
gzip $IMAGE_NAME

echo ""
echo "✅ SD card image created successfully!"
echo ""
echo "📁 Image file: ${IMAGE_NAME}.gz"
echo "💾 Size: $(du -h ${IMAGE_NAME}.gz | cut -f1)"
echo ""
echo "To flash to SD card:"
echo "  gunzip ${IMAGE_NAME}.gz"
echo "  sudo dd if=$IMAGE_NAME of=/dev/sdX bs=4M status=progress"
echo "  (replace /dev/sdX with your SD card device)"
echo ""
echo "🎉 Ready to deploy!"