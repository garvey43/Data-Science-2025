#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python3 grade_assignments.py --generate-feedback "$@"
