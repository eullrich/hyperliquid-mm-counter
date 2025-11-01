#!/bin/bash
# Management script for Hyperliquid data collector

PLIST_FILE="/Users/ericullrich/Code/hedge-v4/com.hyperliquid.collector.plist"
LAUNCHD_DEST="$HOME/Library/LaunchAgents/com.hyperliquid.collector.plist"
SERVICE_NAME="com.hyperliquid.collector"

case "$1" in
    install)
        echo "Installing collector service..."
        mkdir -p "$HOME/Library/LaunchAgents"
        cp "$PLIST_FILE" "$LAUNCHD_DEST"
        launchctl bootstrap gui/$(id -u) "$LAUNCHD_DEST" 2>&1
        if [ $? -eq 0 ]; then
            echo "✓ Collector service installed and started"
            echo "  Use 'manage_collector.sh status' to check status"
        else
            echo "✓ Service updated (may already be loaded)"
        fi
        ;;

    uninstall)
        echo "Uninstalling collector service..."
        launchctl bootout gui/$(id -u)/"$SERVICE_NAME" 2>/dev/null
        rm -f "$LAUNCHD_DEST"
        echo "✓ Collector service uninstalled"
        ;;

    start)
        echo "Starting collector..."
        launchctl start "$SERVICE_NAME"
        echo "✓ Collector started"
        ;;

    stop)
        echo "Stopping collector..."
        launchctl stop "$SERVICE_NAME"
        echo "✓ Collector stopped"
        ;;

    restart)
        echo "Restarting collector..."
        launchctl stop "$SERVICE_NAME" 2>/dev/null
        sleep 2
        launchctl start "$SERVICE_NAME"
        echo "✓ Collector restarted"
        ;;

    status)
        echo "Checking collector status..."
        if launchctl list | grep -q "$SERVICE_NAME"; then
            echo "✓ Collector is running"
            echo ""
            echo "Recent logs:"
            tail -20 /Users/ericullrich/Code/hedge-v4/logs/collector.log
        else
            echo "✗ Collector is not running"
        fi
        ;;

    logs)
        echo "Tailing collector logs (Ctrl+C to exit)..."
        tail -f /Users/ericullrich/Code/hedge-v4/logs/collector.log
        ;;

    test)
        echo "Running collector in foreground (Ctrl+C to stop)..."
        cd /Users/ericullrich/Code/hedge-v4
        python3 collector.py
        ;;

    prevent-sleep)
        echo "Configuring Mac to prevent sleep while on power..."
        sudo pmset -c sleep 0
        sudo pmset -c disksleep 0
        sudo pmset -c displaysleep 10
        echo "✓ Sleep prevention configured"
        echo "  Display will sleep after 10 min, but system stays awake"
        echo "  To revert: sudo pmset -c sleep 1 disksleep 10 displaysleep 10"
        ;;

    *)
        echo "Hyperliquid Data Collector Management"
        echo ""
        echo "Usage: $0 {install|uninstall|start|stop|restart|status|logs|test|prevent-sleep}"
        echo ""
        echo "Commands:"
        echo "  install        Install and start the collector service"
        echo "  uninstall      Stop and remove the collector service"
        echo "  start          Start the collector"
        echo "  stop           Stop the collector"
        echo "  restart        Restart the collector"
        echo "  status         Check if collector is running"
        echo "  logs           Tail the collector logs"
        echo "  test           Run collector in foreground for testing"
        echo "  prevent-sleep  Configure Mac to not sleep (requires sudo)"
        exit 1
        ;;
esac
