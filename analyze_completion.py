#!/usr/bin/env python3
"""
Assignment Completion Analysis Script
Analyzes student submission completion rates for assignments 1-22
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
import pandas as pd
from datetime import datetime

class CompletionAnalyzer:
    """Analyzes student assignment completion rates"""

    def __init__(self, submissions_path: str = "Submissions"):
        self.submissions_path = Path(submissions_path)
        self.assignments_path = self.submissions_path / "assignments"
        self.expected_assignments = 22  # Assignments 1-22
        self.valid_extensions = {'.py', '.ipynb'}  # Only count code files

    def get_all_students(self) -> List[str]:
        """Get list of all student directories"""
        if not self.assignments_path.exists():
            print(f"❌ Submissions path not found: {self.assignments_path}")
            return []

        students = []
        for item in self.assignments_path.iterdir():
            if item.is_dir() and item.name not in ['solution']:
                students.append(item.name)

        return sorted(students)

    def analyze_student_submissions(self, student_name: str) -> Dict[str, any]:
        """Analyze submissions for a specific student"""
        student_dir = self.assignments_path / student_name

        if not student_dir.exists():
            return {
                'student': student_name,
                'error': 'Directory not found',
                'completed': 0,
                'total_files': 0,
                'completion_rate': 0.0
            }

        # Find all submission files
        submission_files = []
        assignment_numbers = set()

        for file_path in student_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix in self.valid_extensions:
                submission_files.append(file_path)

                # Try to extract assignment number from filename
                filename = file_path.name.lower()
                assignment_num = self._extract_assignment_number(filename)
                if assignment_num:
                    assignment_numbers.add(assignment_num)

        # Count unique assignments completed
        completed_assignments = len(assignment_numbers)

        return {
            'student': student_name,
            'completed': completed_assignments,
            'remaining': max(0, self.expected_assignments - completed_assignments),
            'total_files': len(submission_files),
            'completion_rate': round((completed_assignments / self.expected_assignments) * 100, 1),
            'assignment_numbers': sorted(list(assignment_numbers)),
            'files': [str(f.relative_to(student_dir)) for f in submission_files]
        }

    def _extract_assignment_number(self, filename: str) -> int:
        """Extract assignment number from filename"""
        import re

        # Common patterns for assignment numbers
        patterns = [
            r'assignment\s*(\d+)',  # "assignment 1", "assignment1"
            r'assign\s*(\d+)',      # "assign 1", "assign1"
            r'assig\s*(\d+)',       # "assig 1", "assig1"
            r'(\d+)',               # Just numbers
            r'wk\s*(\d+)',          # "wk 1", "wk1"
            r'week\s*(\d+)',        # "week 1", "week1"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, filename, re.IGNORECASE)
            for match in matches:
                try:
                    num = int(match)
                    if 1 <= num <= 22:  # Only count valid assignment numbers
                        return num
                except ValueError:
                    continue

        return None

    def analyze_all_students(self) -> Dict[str, any]:
        """Analyze completion for all students"""
        students = self.get_all_students()
        results = {}

        print(f"📊 Analyzing {len(students)} students...")

        for student in students:
            print(f"  Analyzing {student}...")
            results[student] = self.analyze_student_submissions(student)

        return results

    def generate_completion_report(self, results: Dict[str, any]) -> str:
        """Generate a comprehensive completion report"""
        if not results:
            return "❌ No results to report"

        # Calculate summary statistics
        completed_all = sum(1 for r in results.values() if r.get('completed', 0) >= self.expected_assignments)
        total_students = len(results)

        # Group by completion ranges
        completion_ranges = {
            '0-5 assignments': 0,
            '6-10 assignments': 0,
            '11-15 assignments': 0,
            '16-20 assignments': 0,
            '21-22 assignments': 0,
            'All 22 assignments': 0
        }

        for result in results.values():
            completed = result.get('completed', 0)
            if completed == 22:
                completion_ranges['All 22 assignments'] += 1
            elif completed >= 21:
                completion_ranges['21-22 assignments'] += 1
            elif completed >= 16:
                completion_ranges['16-20 assignments'] += 1
            elif completed >= 11:
                completion_ranges['11-15 assignments'] += 1
            elif completed >= 6:
                completion_ranges['6-10 assignments'] += 1
            else:
                completion_ranges['0-5 assignments'] += 1

        # Generate report
        report = []
        report.append("=" * 60)
        report.append("📊 ASSIGNMENT COMPLETION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Summary
        report.append("🎯 SUMMARY")
        report.append("-" * 30)
        report.append(f"Total Students: {total_students}")
        report.append(f"Students with All {self.expected_assignments} Assignments: {completed_all}")
        report.append(f"Average Completion Rate: {(sum(r.get('completion_rate', 0) for r in results.values()) / total_students):.1f}%" if total_students > 0 else "Average Completion Rate: 0.0%")
        report.append("")

        # Completion distribution
        report.append("📈 COMPLETION DISTRIBUTION")
        report.append("-" * 30)
        for range_name, count in completion_ranges.items():
            percentage = (count / total_students) * 100 if total_students > 0 else 0
            report.append(f"  {range_name}: {count} students ({percentage:.1f}%)")
        report.append("")

        # Individual student details
        report.append("👥 INDIVIDUAL STUDENT DETAILS")
        report.append("-" * 40)
        report.append(f"{'Student':<20} {'Completed':<10} {'Remaining':<10} {'Rate':<8} {'Files':<8}")
        report.append("-" * 60)

        # Sort by completion rate (highest first)
        sorted_students = sorted(results.items(),
                               key=lambda x: x[1].get('completion_rate', 0),
                               reverse=True)

        for student, data in sorted_students:
            if 'error' in data:
                report.append(f"{student:<20} ERROR: {data['error']}")
                continue

            completed = data.get('completed', 0)
            remaining = data.get('remaining', 0)
            rate = data.get('completion_rate', 0)
            total_files = data.get('total_files', 0)

            status = "✅" if completed >= self.expected_assignments else "⏳"
            report.append(f"{student:<20} {completed:<10} {remaining:<10} {rate:<7.1f}% {total_files:<8}")

            if completed < self.expected_assignments:
                report.append(f"{'':<22} 📋 Remaining: {remaining} assignments")

        return "\n".join(report)

    def export_to_csv(self, results: Dict[str, any], filename: str = "completion_analysis.csv"):
        """Export results to CSV for further analysis"""
        if not results:
            print("❌ No results to export")
            return

        # Prepare data for CSV
        csv_data = []
        for student, data in results.items():
            if 'error' in data:
                csv_data.append({
                    'Student': student,
                    'Completed': 0,
                    'Remaining': self.expected_assignments,
                    'Completion_Rate': 0.0,
                    'Total_Files': 0,
                    'Status': 'Error',
                    'Error': data['error']
                })
            else:
                csv_data.append({
                    'Student': student,
                    'Completed': data.get('completed', 0),
                    'Remaining': data.get('remaining', 0),
                    'Completion_Rate': data.get('completion_rate', 0),
                    'Total_Files': data.get('total_files', 0),
                    'Status': 'Complete' if data.get('completed', 0) >= self.expected_assignments else 'Incomplete',
                    'Assignment_Numbers': ','.join(map(str, data.get('assignment_numbers', [])))
                })

        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(filename, index=False)
        print(f"✅ Results exported to {filename}")

        return df

def main():
    """Main function to run the analysis"""
    print("🔍 Starting Assignment Completion Analysis...")
    print("=" * 50)

    analyzer = CompletionAnalyzer()

    # Analyze all students
    results = analyzer.analyze_all_students()

    if not results:
        print("❌ No student data found to analyze")
        return

    # Generate and display report
    report = analyzer.generate_completion_report(results)
    print(report)

    # Export to CSV
    analyzer.export_to_csv(results)

    # Save detailed JSON report
    with open('completion_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n📄 Detailed results saved to:")
    print("   • completion_analysis.json")
    print("   • completion_analysis.csv")

if __name__ == "__main__":
    main()