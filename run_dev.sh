#!/bin/bash
# Run Scope server with a local TURN relay for remote access via SSH tunnel.
#
# SSH tunnel setup (run from your local machine):
#   ssh -L 8000:localhost:8000 -L 3478:localhost:3478 user@remote-host
#
# Then open http://localhost:8000 in your local browser.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Local TURN server (coturn) ---
TURNSERVER="${HOME}/.local/bin/turnserver"
TURN_CONF="${SCRIPT_DIR}/coturn.conf"

if [ -x "$TURNSERVER" ]; then
    # Kill any existing turnserver
    pkill -f "turnserver.*coturn.conf" 2>/dev/null
    sleep 0.5

    echo "Starting local TURN server on port 3478..."
    "$TURNSERVER" -c "$TURN_CONF" &
    TURN_PID=$!
    sleep 1

    if kill -0 "$TURN_PID" 2>/dev/null; then
        echo "TURN server started (PID $TURN_PID)"
    else
        echo "WARNING: TURN server failed to start, WebRTC may not work remotely"
    fi

    # Point the Scope server at the local TURN server (TCP)
    export TURN_URL="turn:127.0.0.1:3478?transport=tcp"
    export TURN_USERNAME="scope"
    export TURN_PASSWORD="scope123"

    # Cleanup on exit
    trap "kill $TURN_PID 2>/dev/null; echo 'TURN server stopped'" EXIT
else
    echo "WARNING: turnserver not found at $TURNSERVER"
    echo "WebRTC will use HF_TOKEN/Twilio TURN or default STUN (needs internet)."
fi

# --- Scope server ---
PYTHONPATH="${SCRIPT_DIR}/src:$PYTHONPATH"
python -m scope.server.app
