#!/usr/bin/env python3
"""
Update Dashboard Data Script
Copies latest completion analysis data to dashboard directory for Vercel deployment
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

def update_dashboard_data():
    """Update dashboard with latest completion data"""

    print("🔄 Updating dashboard data...")

    # Paths
    root_dir = Path(__file__).parent
    dashboard_dir = root_dir / "dashboard"
    completion_json = root_dir / "completion_analysis.json"
    completion_csv = root_dir / "completion_analysis.csv"

    # Ensure dashboard directory exists
    dashboard_dir.mkdir(exist_ok=True)

    # Copy data files to dashboard
    files_copied = []

    if completion_json.exists():
        shutil.copy2(completion_json, dashboard_dir / "completion_analysis.json")
        files_copied.append("completion_analysis.json")
        print("✅ Copied completion_analysis.json")

    if completion_csv.exists():
        shutil.copy2(completion_csv, dashboard_dir / "completion_analysis.csv")
        files_copied.append("completion_analysis.csv")
        print("✅ Copied completion_analysis.csv")

    # Create metadata file
    metadata = {
        "last_updated": datetime.now().isoformat(),
        "data_files": files_copied,
        "total_students": 0,
        "completed_all": 0,
        "average_completion": 0.0
    }

    # Try to extract summary from JSON data
    if completion_json.exists():
        try:
            with open(completion_json, 'r') as f:
                data = json.load(f)

            if "_summary" in data:
                summary = data["_summary"]
                metadata.update({
                    "total_students": summary.get("total_students", 0),
                    "completed_all": summary.get("grade_distribution", {}).get("A (90-100)", 0),
                    "average_completion": summary.get("average_grade", 0)
                })
        except Exception as e:
            print(f"⚠️ Could not extract metadata: {e}")

    # Save metadata
    with open(dashboard_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print("✅ Created metadata.json")

    # Create a simple API endpoint data file
    if completion_json.exists():
        try:
            with open(completion_json, 'r') as f:
                data = json.load(f)

            # Create a simplified version for the dashboard
            simplified_data = {}
            for student_name, student_data in data.items():
                if student_name != "_summary":
                    simplified_data[student_name] = {
                        "completed": student_data.get("completed", 0),
                        "remaining": student_data.get("remaining", 0),
                        "completion_rate": student_data.get("completion_rate", 0),
                        "total_files": student_data.get("total_files", 0),
                        "assignment_numbers": student_data.get("assignment_numbers", [])
                    }

            with open(dashboard_dir / "students_data.json", 'w') as f:
                json.dump(simplified_data, f, indent=2)

            print("✅ Created students_data.json")

        except Exception as e:
            print(f"⚠️ Could not create simplified data: {e}")

    print("\n🎯 Dashboard update complete!")
    print(f"📁 Files in dashboard: {list(dashboard_dir.glob('*'))}")
    print("\n🚀 Ready for Vercel deployment!")
    print("   Run: cd dashboard && vercel --prod")
def create_deployment_script():
    """Create a deployment script for easy Vercel updates"""

    script_content = '''#!/bin/bash
# Deploy Dashboard to Vercel

echo "🚀 Deploying Data Science Dashboard to Vercel..."

# Update dashboard data
python3 ../update_dashboard.py

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Install with: npm i -g vercel"
    exit 1
fi

# Deploy to Vercel
cd "$(dirname "$0")"
vercel --prod

echo "✅ Deployment complete!"
echo "🌐 Your dashboard is now live!"
'''

    script_path = Path(__file__).parent / "dashboard" / "deploy.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)

    # Make executable
    os.chmod(script_path, 0o755)

    print("✅ Created deploy.sh script")

def create_readme():
    """Create README for dashboard deployment"""

    readme_content = '''# Data Science 2025 - Assignment Completion Dashboard

A beautiful, interactive dashboard showing student assignment completion statistics.

## 🌟 Features

- 📊 Real-time completion statistics
- 📈 Interactive charts and visualizations
- 👥 Individual student performance tracking
- 📋 Assignment completion patterns
- 🎯 Key insights and recommendations
- 📱 Responsive design for all devices

## 🚀 Deployment

### Quick Deploy
```bash
# Update data and deploy
./deploy.sh
```

### Manual Deploy
```bash
# Update dashboard data
python3 ../update_dashboard.py

# Deploy to Vercel
cd dashboard
vercel --prod
```

## 📊 Data Sources

The dashboard reads data from:
- `completion_analysis.json` - Detailed completion data
- `completion_analysis.csv` - Spreadsheet format
- `metadata.json` - Dashboard metadata

## 🔧 Development

### Local Testing
```bash
# Start local server
python3 -m http.server 8000

# Open in browser
# http://localhost:8000/dashboard/
```

### Updating Data
```bash
# Run completion analysis
python3 analyze_completion.py

# Update dashboard
python3 update_dashboard.py
```

## 📈 Automatic Updates

The dashboard automatically refreshes every 5 minutes when live. For manual updates:

1. Run assignment analysis: `python3 analyze_completion.py`
2. Update dashboard: `python3 update_dashboard.py`
3. Deploy: `./dashboard/deploy.sh`

## 🎨 Customization

### Colors
Edit `styles.css` to customize the color scheme.

### Charts
Modify `script.js` to add new chart types or change existing ones.

### Data
Update `analyze_completion.py` to include additional metrics.

## 📞 Support

For issues or feature requests, please check the main project repository.

---
*Built with ❤️ for Data Science 2025*
'''

    readme_path = Path(__file__).parent / "dashboard" / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print("✅ Created dashboard README.md")

if __name__ == "__main__":
    update_dashboard_data()
    create_deployment_script()
    create_readme()