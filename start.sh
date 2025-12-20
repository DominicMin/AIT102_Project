#!/bin/bash
# Script to start frontend and backend services

echo "========================================="
echo "Starting AI Style Transfer Project"
echo "========================================="

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Start backend service
echo ""
echo "Starting backend service..."
cd src
python server.py &
BACKEND_PID=$!
echo "Backend service started (PID: $BACKEND_PID)"
cd ..

# Wait for backend to start
sleep 3

# Start frontend service
echo ""
echo "Starting frontend service..."
cd frontend
npm run dev &
FRONTEND_PID=$!
echo "Frontend service started (PID: $FRONTEND_PID)"
cd ..

# Wait for services to be ready
sleep 2

# Open browser
echo ""
echo "Opening browser..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    cmd.exe /c start http://localhost:3000
elif command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:3000
elif command -v open &> /dev/null; then
    open http://localhost:3000
fi

echo ""
echo "========================================="
echo "All services are running!"
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "========================================="
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for user interrupt
wait
