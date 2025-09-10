import os
import re
import json
from pathlib import Path

def parse_feedback_file(filepath):
    student = Path(filepath).stem.replace('_feedback', '')
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Capture average grade for individual assignments
    avg_match = re.search(r'\*\*Average Grade:\*\* ([\d.]+)/100', content)
    average = float(avg_match.group(1)) if avg_match else 0
    
    submissions = {}
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#### ') and (line.endswith('.ipynb') or line.endswith('.py')):
            current_file = line[5:].strip()
            i += 2  # Skip to grade line
            while i < len(lines):
                grade_line = lines[i].strip()
                grade_match = re.search(r'\*\*Grade:\*\* ([\d.]+)/100', grade_line)
                if grade_match:
                    submissions[current_file] = float(grade_match.group(1))
                    i += 1
                    break
                i += 1
        else:
            i += 1
    
    return {
        'student': student,
        'average_grade': average,
        'submissions': submissions
    }

def analyze_feedback():
    feedback_dir = 'feedback'
    feedback_data = {}
    for file in os.listdir(feedback_dir):
        if file.endswith('_feedback.md'):
            filepath = os.path.join(feedback_dir, file)
            data = parse_feedback_file(filepath)
            feedback_data[data['student']] = data
    
    # Load completion data
    with open('completion_analysis.json', 'r') as f:
        completion_data = json.load(f)
    
    # Merge feedback into completion data
    for student, comp in completion_data.items():
        if student in feedback_data:
            comp['average_grade'] = feedback_data[student]['average_grade']
            comp['feedback_submissions'] = feedback_data[student]['submissions']
    
    # Save merged data
    with open('feedback_analysis.json', 'w') as f:
        json.dump(completion_data, f, indent=2)
    
    print("Feedback analysis completed. Merged data saved to feedback_analysis.json")

if __name__ == '__main__':
    analyze_feedback()