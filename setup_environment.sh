#!/bin/bash

echo "🔧 Assignment Grader Environment Setup"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Manjaro/Arch
if command -v pacman &> /dev/null; then
    print_status "Detected Manjaro/Arch Linux system"
    PACKAGE_MANAGER="pacman"
elif command -v apt &> /dev/null; then
    print_status "Detected Ubuntu/Debian system"
    PACKAGE_MANAGER="apt"
else
    print_warning "Package manager not detected, proceeding with Python-only setup"
    PACKAGE_MANAGER="none"
fi

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_status "Python version: $PYTHON_VERSION"
    
    # Check if Python version is compatible (>=3.7)
    if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 7) else 1)"; then
        print_status "Python version is compatible"
    else
        print_error "Python 3.7 or higher required. Please upgrade Python."
        exit 1
    fi
else
    print_error "Python 3 not found. Please install Python 3.7+"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    print_warning "pip3 not found, attempting to install..."
    if [ "$PACKAGE_MANAGER" = "pacman" ]; then
        sudo pacman -S python-pip --noconfirm
    elif [ "$PACKAGE_MANAGER" = "apt" ]; then
        sudo apt update && sudo apt install python3-pip -y
    else
        print_error "Please install pip3 manually"
        exit 1
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install requirements
print_status "Installing Python dependencies..."

# Create a comprehensive requirements file
cat > requirements.txt << EOF
# Core dependencies
nbformat>=5.7.0
jupyter>=1.0.0
notebook>=6.0.0

# File system monitoring
watchdog>=2.0.0

# Data analysis and utilities
pandas>=1.3.0
numpy>=1.20.0

# Optional but useful
matplotlib>=3.5.0
seaborn>=0.11.0

# Code analysis
flake8>=4.0.0
black>=22.0.0

# Testing
pytest>=7.0.0
EOF

# Install requirements
pip install -r requirements.txt

# Check if installation was successful
print_status "Verifying installations..."

# Test nbformat
if python3 -c "import nbformat; print('nbformat version:', nbformat.__version__)" 2>/dev/null; then
    print_status "✅ nbformat installed successfully"
else
    print_error "❌ nbformat installation failed"
    exit 1
fi

# Test watchdog
if python3 -c "import watchdog; print('watchdog installed successfully')" 2>/dev/null; then
    print_status "✅ watchdog installed successfully"
else
    print_warning "⚠️  watchdog installation failed - monitor_grading.py may not work"
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p feedback
mkdir -p grading_results
mkdir -p scripts

# Make scripts executable
print_status "Making scripts executable..."
chmod +x scripts/quick_start.sh 2>/dev/null || true
chmod +x scripts/test_grader.py 2>/dev/null || true

# Test the grader
print_status "Testing the grader..."
if python3 grade_assignments.py --help > /dev/null 2>&1; then
    print_status "✅ Grade assignments script is working"
else
    print_error "❌ Grade assignments script has issues"
fi

# Run the test script
print_status "Running test script..."
if python3 scripts/test_grader.py; then
    print_status "✅ Test script passed"
else
    print_warning "⚠️  Test script had issues but continuing..."
fi

# Create a launcher script
print_status "Creating launcher script..."
cat > run_grader.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python3 grade_assignments.py --generate-feedback "$@"
EOF

chmod +x run_grader.sh

# Create monitor launcher
cat > run_monitor.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python3 monitor_grading.py "$@"
EOF

chmod +x run_monitor.sh

print_status "🎉 Setup completed successfully!"
echo ""
echo "Usage:"
echo "  ./run_grader.sh              # Run the grader once"
echo "  ./run_monitor.sh --mode monitor  # Start monitoring mode"
echo "  ./run_monitor.sh --mode grade    # Run full grading"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo "  python3 grade_assignments.py --generate-feedback"
echo "  python3 monitor_grading.py"
echo ""
print_status "Check the logs and results in:"
echo "  - grading_results.json"
echo "  - feedback/ directory"
echo "  - grader.log"