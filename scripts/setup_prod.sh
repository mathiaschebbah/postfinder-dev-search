#!/bin/bash
# Setup script for Mac Studio production deployment

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PLIST_NAME="com.postfinder.embed.plist"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"

echo "=== Postfinder Embedding Server - Production Setup ==="
echo ""

# 1. Create logs directory
echo "1. Creating logs directory..."
mkdir -p "$PROJECT_DIR/logs"

# 2. Make scripts executable
echo "2. Making scripts executable..."
chmod +x "$SCRIPT_DIR/start_server.sh"

# 3. Update plist with correct paths
echo "3. Updating plist with your username..."
sed -i '' "s|/Users/mathias|$HOME|g" "$SCRIPT_DIR/$PLIST_NAME"

# 4. Install launchd service
echo "4. Installing launchd service..."
mkdir -p "$LAUNCH_AGENTS_DIR"
cp "$SCRIPT_DIR/$PLIST_NAME" "$LAUNCH_AGENTS_DIR/"

# 5. Sync dependencies
echo "5. Syncing dependencies..."
cd "$PROJECT_DIR"
uv sync

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Commands:"
echo "  Start:   launchctl load ~/Library/LaunchAgents/$PLIST_NAME"
echo "  Stop:    launchctl unload ~/Library/LaunchAgents/$PLIST_NAME"
echo "  Logs:    tail -f $PROJECT_DIR/logs/server.log"
echo "  Status:  curl http://localhost:8000/health"
echo ""
echo "Next: Setup Cloudflare Tunnel"
echo "  brew install cloudflared"
echo "  cloudflared tunnel --url http://localhost:8000"
