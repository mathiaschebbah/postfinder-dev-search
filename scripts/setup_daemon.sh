#!/bin/bash
# Setup script for Mac Studio - works via SSH (headless)
# Requires sudo

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PLIST_NAME="com.postfinder.embed.daemon.plist"
DAEMON_DIR="/Library/LaunchDaemons"

echo "=== Postfinder Embedding Server - Daemon Setup ==="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo: sudo ./setup_daemon.sh"
    exit 1
fi

# 1. Create logs directory
echo "1. Creating logs directory..."
mkdir -p "$PROJECT_DIR/logs"
chown -R mathias:staff "$PROJECT_DIR/logs"

# 2. Sync dependencies
echo "2. Syncing dependencies..."
cd "$PROJECT_DIR"
sudo -u mathias /opt/homebrew/bin/uv sync

# 3. Copy plist to LaunchDaemons
echo "3. Installing LaunchDaemon..."
cp "$SCRIPT_DIR/$PLIST_NAME" "$DAEMON_DIR/com.postfinder.embed.plist"
chown root:wheel "$DAEMON_DIR/com.postfinder.embed.plist"
chmod 644 "$DAEMON_DIR/com.postfinder.embed.plist"

# 4. Load the daemon
echo "4. Loading daemon..."
launchctl load "$DAEMON_DIR/com.postfinder.embed.plist"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Commands:"
echo "  Status:  curl http://localhost:8000/health"
echo "  Logs:    tail -f $PROJECT_DIR/logs/server.log"
echo "  Errors:  tail -f $PROJECT_DIR/logs/server.error.log"
echo "  Stop:    sudo launchctl unload /Library/LaunchDaemons/com.postfinder.embed.plist"
echo "  Start:   sudo launchctl load /Library/LaunchDaemons/com.postfinder.embed.plist"
echo ""
