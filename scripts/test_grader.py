#!/usr/bin/env python3
"""
Test script for the assignment grader
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

def create_test_structure():
    """Create a test directory structure"""
    test_dir = Path(tempfile.mkdtemp(prefix="grader_test_"))
    
    # Create test repository structure
    submissions_dir = test_dir / "Submissions" / "assignments"
    submissions_dir.mkdir(parents=True)
    
    # Create test student directory
    student_dir = submissions_dir / "TestStudent"
    student_dir.mkdir()
    
    # Create test Python file
    test_code = '''
def calculate_sum(a, b):
    """Calculate the sum of two numbers"""
    return a + b

def main():
    # Test the function
    result = calculate_sum(5, 3)
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
'''
    
    with open(student_dir / "test_assignment.py", "w") as f:
        f.write(test_code)
    
    # Create test notebook
    notebook_content = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Test notebook cell\n",
                    "def fibonacci(n):\n",
                    "    \"\"\"Calculate fibonacci number\"\"\"\n",
                    "    if n <= 1:\n",
                    "        return n\n",
                    "    return fibonacci(n-1) + fibonacci(n-2)\n",
                    "\n",
                    "print(fibonacci(10))"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    import json
    with open(student_dir / "test_notebook.ipynb", "w") as f:
        json.dump(notebook_content, f, indent=2)
    
    return test_dir

def run_test():
    """Run the grader test"""
    print("Creating test structure...")
    test_dir = create_test_structure()
    
    try:
        # Import the grader (assumes it's in the same directory)
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from grade_assignments import AssignmentGrader
        
        print("Running grader test...")
        grader = AssignmentGrader(str(test_dir))
        results = grader.grade_all_assignments()
        
        print("Test Results:")
        print(f"Students processed: {len([k for k in results.keys() if k != '_summary'])}")
        print(f"Total submissions: {results.get('_summary', {}).get('total_submissions', 0)}")
        
        # Check if we got expected results
        if 'TestStudent' in results:
            student_results = results['TestStudent']
            print(f"TestStudent submissions: {student_results.get('individual', {}).get('total_submissions', 0)}")
            print("✅ Test passed!")
        else:
            print("❌ Test failed - TestStudent not found in results")
            return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
    finally:
        # Clean up
        shutil.rmtree(test_dir)
        print("Test cleanup completed")
    
    return True

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)