#!/bin/bash

echo " Quick Start - Assignment Grader"
echo "=================================="

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo " Python not found. Please install Python 3.7+"
    exit 1
fi

# Check if we have the grader script
if [ ! -f "grade_assignments.py" ]; then
    echo " grade_assignments.py not found. Please ensure it's in the repository root."
    exit 1
fi

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Run a test
echo "Running test..."
python scripts/test_grader.py

# Run the actual grader
echo "Running assignment grader..."
python grade_assignments.py --generate-feedback

echo "Grading complete!"
echo "Check grading_results.json for results"
echo "Check feedback/ directory for individual feedback files"