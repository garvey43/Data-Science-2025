#!/usr/bin/env python3
"""
Automated Assignment Grader for GitHub Class Repository
Handles Python files, Jupyter notebooks, and detects AI-generated content
"""

import os
import json
import re
import ast
import nbformat
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import hashlib
import difflib
import statistics
from collections import defaultdict, Counter
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('grader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AIDetector:
    """Detects AI-generated code using multiple heuristics"""
    
    def __init__(self):
        # Common AI-generated code patterns
        self.ai_patterns = [
            r'#\s*This\s+code\s+(was\s+)?generated\s+by',
            r'#\s*AI\s+generated',
            r'#\s*Generated\s+using',
            r'#\s*Created\s+with\s+the\s+help\s+of',
            r'#\s*Note:\s+This\s+is\s+a\s+basic\s+implementation',
            r'#\s*Here\'s\s+a\s+(simple|basic)\s+implementation',
            r'def\s+\w+\([^)]*\):\s*\n\s*"""[^"]*implementation[^"]*"""',
            r'#\s*You\s+can\s+customize\s+this\s+further',
            r'#\s*Feel\s+free\s+to\s+modify',
            r'#\s*This\s+should\s+work\s+for\s+most\s+cases'
        ]
        
        # AI-style comment patterns
        self.ai_comment_patterns = [
            r'#\s*Step\s+\d+:',
            r'#\s*\d+\.\s+',
            r'#\s*First,?\s+we',
            r'#\s*Then,?\s+we',
            r'#\s*Finally,?\s+we',
            r'#\s*Now,?\s+we',
            r'#\s*Let\'s\s+',
            r'#\s*We\s+can\s+',
            r'#\s*This\s+will\s+',
            r'#\s*Here\s+we\s+'
        ]
        
        # Overly perfect variable names (AI tends to use very descriptive names)
        self.ai_variable_patterns = [
            r'user_input_\w+',
            r'result_\w+',
            r'final_\w+',
            r'calculated_\w+',
            r'processed_\w+',
            r'converted_\w+',
            r'validated_\w+'
        ]

    def detect_ai_content(self, code: str) -> Dict[str, Any]:
        """Detect if code is likely AI-generated"""
        score = 0
        indicators = []
        
        # Check for AI patterns
        for pattern in self.ai_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                score += 3
                indicators.append(f"AI pattern found: {pattern}")
        
        # Check comment patterns
        ai_comment_count = 0
        for pattern in self.ai_comment_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            ai_comment_count += len(matches)
        
        if ai_comment_count > 3:
            score += 2
            indicators.append(f"AI-style comments detected: {ai_comment_count}")
        
        # Check variable naming patterns
        ai_var_count = 0
        for pattern in self.ai_variable_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            ai_var_count += len(matches)
        
        if ai_var_count > 2:
            score += 1
            indicators.append(f"AI-style variable names: {ai_var_count}")
        
        # Check for overly verbose docstrings
        docstring_pattern = r'"""[^"]{100,}"""'
        verbose_docstrings = len(re.findall(docstring_pattern, code))
        if verbose_docstrings > 2:
            score += 1
            indicators.append("Overly verbose docstrings detected")
        
        # Check for perfect error handling (AI tends to add comprehensive error handling)
        try_except_count = len(re.findall(r'try:\s*\n.*?except.*?:', code, re.DOTALL))
        if try_except_count > 3:
            score += 1
            indicators.append("Excessive error handling detected")
        
        # Check for code structure complexity (AI tends to create very structured code)
        lines = code.split('\n')
        comment_ratio = len([line for line in lines if line.strip().startswith('#')]) / max(len(lines), 1)
        if comment_ratio > 0.3:
            score += 1
            indicators.append(f"High comment ratio: {comment_ratio:.2f}")
        
        return {
            'ai_score': score,
            'likelihood': self._get_likelihood(score),
            'indicators': indicators,
            'confidence': min(score * 10, 100)
        }
    
    def _get_likelihood(self, score: int) -> str:
        """Convert score to likelihood text"""
        if score >= 5:
            return "Very High"
        elif score >= 3:
            return "High"
        elif score >= 2:
            return "Medium"
        elif score >= 1:
            return "Low"
        else:
            return "Very Low"

class CodeAnalyzer:
    """Analyzes Python code for various metrics"""
    
    def __init__(self):
        self.ai_detector = AIDetector()
    
    def analyze_python_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return self.analyze_code(content, file_path)
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return {'error': str(e)}
    
    def analyze_jupyter_notebook(self, file_path: str) -> Dict[str, Any]:
        """Analyze a Jupyter notebook"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            # Extract code from cells
            code_cells = [cell['source'] for cell in nb.cells if cell.cell_type == 'code']
            combined_code = '\n'.join(code_cells)
            
            analysis = self.analyze_code(combined_code, file_path)
            analysis['notebook_info'] = {
                'total_cells': len(nb.cells),
                'code_cells': len(code_cells),
                'markdown_cells': len([cell for cell in nb.cells if cell.cell_type == 'markdown'])
            }
            
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing notebook {file_path}: {e}")
            return {'error': str(e)}
    
    def analyze_code(self, code: str, file_path: str) -> Dict[str, Any]:
        """Analyze code content"""
        analysis = {
            'file_path': file_path,
            'lines_of_code': len([line for line in code.split('\n') if line.strip()]),
            'functions': self._extract_functions(code),
            'classes': self._extract_classes(code),
            'imports': self._extract_imports(code),
            'complexity': self._calculate_complexity(code),
            'documentation': self._check_documentation(code),
            'ai_detection': self.ai_detector.detect_ai_content(code),
            'syntax_errors': self._check_syntax(code),
            'style_issues': self._check_style(code)
        }
        
        return analysis
    
    def _extract_functions(self, code: str) -> List[str]:
        """Extract function names from code"""
        try:
            tree = ast.parse(code)
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
            return functions
        except:
            return []
    
    def _extract_classes(self, code: str) -> List[str]:
        """Extract class names from code"""
        try:
            tree = ast.parse(code)
            classes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
            return classes
        except:
            return []
    
    def _extract_imports(self, code: str) -> List[str]:
        """Extract import statements"""
        imports = []
        for line in code.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)
        return imports
    
    def _calculate_complexity(self, code: str) -> Dict[str, int]:
        """Calculate code complexity metrics"""
        lines = code.split('\n')
        return {
            'total_lines': len(lines),
            'code_lines': len([line for line in lines if line.strip() and not line.strip().startswith('#')]),
            'comment_lines': len([line for line in lines if line.strip().startswith('#')]),
            'blank_lines': len([line for line in lines if not line.strip()]),
            'indentation_levels': self._max_indentation(code)
        }
    
    def _max_indentation(self, code: str) -> int:
        """Find maximum indentation level"""
        max_indent = 0
        for line in code.split('\n'):
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent // 4)  # Assuming 4-space indentation
        return max_indent
    
    def _check_documentation(self, code: str) -> Dict[str, Any]:
        """Check documentation quality"""
        docstring_count = len(re.findall(r'"""[^"]*"""', code)) + len(re.findall(r"'''[^']*'''", code))
        comment_count = len(re.findall(r'#.*', code))
        
        return {
            'docstrings': docstring_count,
            'comments': comment_count,
            'documented_functions': self._count_documented_functions(code)
        }
    
    def _count_documented_functions(self, code: str) -> int:
        """Count functions with docstrings"""
        try:
            tree = ast.parse(code)
            documented = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if (node.body and isinstance(node.body[0], ast.Expr) and 
                        isinstance(node.body[0].value, ast.Constant) and 
                        isinstance(node.body[0].value.value, str)):
                        documented += 1
            return documented
        except:
            return 0
    
    def _check_syntax(self, code: str) -> List[str]:
        """Check for syntax errors"""
        errors = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            errors.append(f"Parse error: {str(e)}")
        return errors
    
    def _check_style(self, code: str) -> List[str]:
        """Basic style checking"""
        issues = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                issues.append(f"Line {i}: Line too long ({len(line)} characters)")
            
            if line.endswith(' '):
                issues.append(f"Line {i}: Trailing whitespace")
            
            if '    ' in line and '\t' in line:
                issues.append(f"Line {i}: Mixed tabs and spaces")
        
        return issues

class AssignmentGrader:
    """Main grading system"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.submissions_path = self.repo_path / "Submissions"
        self.assignments_path = self.repo_path / "Assignments"
        self.analyzer = CodeAnalyzer()
        self.results = {}
        
    def grade_all_assignments(self) -> Dict[str, Any]:
        """Grade all student submissions"""
        logger.info("Starting assignment grading process...")
        
        # Grade individual assignments
        individual_path = self.submissions_path / "assignments"
        if individual_path.exists():
            self._grade_directory(individual_path, "individual")
        
        # Grade group projects
        group_path = self.submissions_path / "groupwork"
        if group_path.exists():
            self._grade_directory(group_path, "group")
        
        # Grade projects
        projects_path = self.submissions_path / "projects"
        if projects_path.exists():
            self._grade_directory(projects_path, "project")
        
        # Generate summary report
        self._generate_summary_report()
        
        return self.results
    
    def _grade_directory(self, directory: Path, assignment_type: str):
        """Grade submissions in a directory"""
        logger.info(f"Grading {assignment_type} submissions in {directory}")
        
        for student_dir in directory.iterdir():
            if student_dir.is_dir() and not student_dir.name.startswith('.'):
                student_name = student_dir.name
                logger.info(f"Grading {student_name}'s submissions...")
                
                student_results = self._grade_student_submissions(student_dir, student_name)
                
                if student_name not in self.results:
                    self.results[student_name] = {}
                
                self.results[student_name][assignment_type] = student_results
    
    def _grade_student_submissions(self, student_dir: Path, student_name: str) -> Dict[str, Any]:
        """Grade all submissions for a student"""
        submissions = []
        
        for file_path in student_dir.rglob("*"):
            if file_path.is_file() and self._is_gradeable_file(file_path):
                logger.info(f"Analyzing {file_path}")
                
                if file_path.suffix == '.py':
                    analysis = self.analyzer.analyze_python_file(str(file_path))
                elif file_path.suffix == '.ipynb':
                    analysis = self.analyzer.analyze_jupyter_notebook(str(file_path))
                else:
                    continue
                
                # Generate grade and feedback
                grade_info = self._calculate_grade(analysis)
                
                submission = {
                    'file_name': file_path.name,
                    'file_path': str(file_path.relative_to(self.repo_path)),
                    'analysis': analysis,
                    'grade': grade_info['grade'],
                    'feedback': grade_info['feedback'],
                    'ai_detection': analysis.get('ai_detection', {}),
                    'timestamp': datetime.now().isoformat()
                }
                
                submissions.append(submission)
        
        return {
            'submissions': submissions,
            'total_submissions': len(submissions),
            'average_grade': self._calculate_average_grade(submissions),
            'ai_flagged': len([s for s in submissions if s['ai_detection'].get('likelihood') in ['High', 'Very High']])
        }
    
    def _is_gradeable_file(self, file_path: Path) -> bool:
        """Check if file should be graded"""
        return (file_path.suffix in ['.py', '.ipynb'] and 
                not file_path.name.startswith('.') and
                file_path.name not in ['instruction.md', 'desktop.ini'])
    
    def _calculate_grade(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate grade based on analysis"""
        if 'error' in analysis:
            return {
                'grade': 0,
                'feedback': f"Error analyzing file: {analysis['error']}"
            }
        
        score = 100
        feedback = []
        
        # Syntax errors
        if analysis.get('syntax_errors'):
            score -= 30
            feedback.append(f"Syntax errors found: {'; '.join(analysis['syntax_errors'])}")
        
        # Code quality
        complexity = analysis.get('complexity', {})
        if complexity.get('code_lines', 0) < 5:
            score -= 20
            feedback.append("Code is too short - needs more implementation")
        
        # Documentation
        doc_info = analysis.get('documentation', {})
        if doc_info.get('comments', 0) == 0:
            score -= 10
            feedback.append("No comments found - add explanatory comments")
        
        # Style issues
        style_issues = analysis.get('style_issues', [])
        if len(style_issues) > 5:
            score -= 15
            feedback.append(f"Multiple style issues found: {len(style_issues)} issues")
        elif style_issues:
            score -= 5
            feedback.append(f"Minor style issues: {len(style_issues)} issues")
        
        # AI detection penalty
        ai_info = analysis.get('ai_detection', {})
        if ai_info.get('likelihood') == 'Very High':
            score -= 50
            feedback.append("⚠️ HIGH PROBABILITY OF AI-GENERATED CODE - Please submit original work")
        elif ai_info.get('likelihood') == 'High':
            score -= 30
            feedback.append("⚠️ Possible AI assistance detected - Ensure this is your original work")
        elif ai_info.get('likelihood') == 'Medium':
            score -= 10
            feedback.append("Some patterns suggest possible AI assistance")
        
        # Positive feedback
        if analysis.get('functions'):
            feedback.append(f"✓ Good use of functions: {len(analysis['functions'])} functions defined")
        
        if doc_info.get('documented_functions', 0) > 0:
            feedback.append(f"✓ Well documented: {doc_info['documented_functions']} functions have docstrings")
        
        if not style_issues:
            feedback.append("✓ Clean code style")
        
        if not analysis.get('syntax_errors'):
            feedback.append("✓ No syntax errors")
        
        score = max(0, min(100, score))  # Clamp between 0 and 100
        
        return {
            'grade': score,
            'feedback': feedback
        }
    
    def _calculate_average_grade(self, submissions: List[Dict]) -> float:
        """Calculate average grade for submissions"""
        if not submissions:
            return 0
        
        grades = [s['grade'] for s in submissions if isinstance(s['grade'], (int, float))]
        return round(statistics.mean(grades), 2) if grades else 0
    
    def _generate_summary_report(self):
        """Generate summary report"""
        total_students = len(self.results)
        total_submissions = sum(
            len(student_data.get('individual', {}).get('submissions', [])) +
            len(student_data.get('group', {}).get('submissions', [])) +
            len(student_data.get('project', {}).get('submissions', []))
            for student_data in self.results.values()
        )
        
        all_grades = []
        ai_flagged_count = 0
        
        for student_data in self.results.values():
            for assignment_type in ['individual', 'group', 'project']:
                if assignment_type in student_data:
                    submissions = student_data[assignment_type].get('submissions', [])
                    for submission in submissions:
                        if isinstance(submission['grade'], (int, float)):
                            all_grades.append(submission['grade'])
                        
                        ai_likelihood = submission['ai_detection'].get('likelihood', 'Very Low')
                        if ai_likelihood in ['High', 'Very High']:
                            ai_flagged_count += 1
        
        summary = {
            'total_students': total_students,
            'total_submissions': total_submissions,
            'average_grade': round(statistics.mean(all_grades), 2) if all_grades else 0,
            'grade_distribution': {
                'A (90-100)': len([g for g in all_grades if g >= 90]),
                'B (80-89)': len([g for g in all_grades if 80 <= g < 90]),
                'C (70-79)': len([g for g in all_grades if 70 <= g < 80]),
                'D (60-69)': len([g for g in all_grades if 60 <= g < 70]),
                'F (0-59)': len([g for g in all_grades if g < 60])
            },
            'ai_flagged_submissions': ai_flagged_count,
            'ai_flagged_percentage': round((ai_flagged_count / max(total_submissions, 1)) * 100, 2)
        }
        
        self.results['_summary'] = summary
        logger.info(f"Grading complete: {total_students} students, {total_submissions} submissions")
    
    def generate_feedback_files(self):
        """Generate individual feedback files for each student"""
        feedback_dir = self.repo_path / "feedback"
        feedback_dir.mkdir(exist_ok=True)
        
        for student_name, student_data in self.results.items():
            if student_name == '_summary':
                continue
            
            feedback_file = feedback_dir / f"{student_name}_feedback.md"
            self._write_student_feedback(feedback_file, student_name, student_data)
    
    def _write_student_feedback(self, file_path: Path, student_name: str, student_data: Dict):
        """Write feedback file for a student"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# Feedback for {student_name}\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for assignment_type in ['individual', 'group', 'project']:
                if assignment_type in student_data:
                    type_data = student_data[assignment_type]
                    f.write(f"## {assignment_type.title()} Assignments\n\n")
                    f.write(f"**Average Grade:** {type_data.get('average_grade', 'N/A')}/100\n")
                    f.write(f"**Total Submissions:** {type_data.get('total_submissions', 0)}\n")
                    
                    if type_data.get('ai_flagged', 0) > 0:
                        f.write(f"**⚠️ AI Detection Flags:** {type_data['ai_flagged']} submissions\n")
                    
                    f.write("\n### Individual Submissions:\n\n")
                    
                    for submission in type_data.get('submissions', []):
                        f.write(f"#### {submission['file_name']}\n")
                        f.write(f"**Grade:** {submission['grade']}/100\n")
                        f.write(f"**AI Detection:** {submission['ai_detection'].get('likelihood', 'N/A')}\n")
                        
                        if submission['ai_detection'].get('confidence', 0) > 50:
                            f.write(f"**AI Confidence:** {submission['ai_detection'].get('confidence', 0)}%\n")
                        
                        f.write("\n**Feedback:**\n")
                        for feedback_item in submission['feedback']:
                            f.write(f"- {feedback_item}\n")
                        
                        f.write("\n")
            
            # Overall recommendations
            f.write("## Overall Recommendations\n\n")
            all_submissions = []
            for assignment_type in ['individual', 'group', 'project']:
                if assignment_type in student_data:
                    all_submissions.extend(student_data[assignment_type].get('submissions', []))
            
            if all_submissions:
                avg_grade = statistics.mean([s['grade'] for s in all_submissions if isinstance(s['grade'], (int, float))])
                ai_flagged = sum(1 for s in all_submissions if s['ai_detection'].get('likelihood') in ['High', 'Very High'])
                
                if avg_grade >= 90:
                    f.write("- Excellent work! Keep up the great coding practices.\n")
                elif avg_grade >= 80:
                    f.write("- Good work overall. Focus on the feedback points to improve further.\n")
                elif avg_grade >= 70:
                    f.write("- Satisfactory work. Please address the issues mentioned in feedback.\n")
                else:
                    f.write("- Needs improvement. Please review the feedback and seek help if needed.\n")
                
                if ai_flagged > 0:
                    f.write("- ⚠️ **IMPORTANT:** Some submissions show signs of AI assistance. Please ensure all work is original.\n")
                    f.write("- If you used AI tools for learning, please disclose this and submit your own implementation.\n")

def main():
    """Main function to run the grader"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated Assignment Grader')
    parser.add_argument('--repo-path', default='.', help='Path to the repository')
    parser.add_argument('--output', default='grading_results.json', help='Output file for results')
    parser.add_argument('--generate-feedback', action='store_true', help='Generate individual feedback files')
    
    args = parser.parse_args()
    
    # Initialize grader
    grader = AssignmentGrader(args.repo_path)
    
    # Grade all assignments
    results = grader.grade_all_assignments()
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Generate feedback files if requested
    if args.generate_feedback:
        grader.generate_feedback_files()
        logger.info("Feedback files generated in 'feedback' directory")
    
    # Print summary
    summary = results.get('_summary', {})
    print(f"\n{'='*50}")
    print("GRADING SUMMARY")
    print(f"{'='*50}")
    print(f"Total Students: {summary.get('total_students', 0)}")
    print(f"Total Submissions: {summary.get('total_submissions', 0)}")
    print(f"Average Grade: {summary.get('average_grade', 0)}/100")
    print(f"AI Flagged: {summary.get('ai_flagged_submissions', 0)} ({summary.get('ai_flagged_percentage', 0)}%)")
    print(f"\nGrade Distribution:")
    for grade_range, count in summary.get('grade_distribution', {}).items():
        print(f"  {grade_range}: {count} students")
    
    print(f"\nResults saved to: {args.output}")
    print("Run with --generate-feedback to create individual feedback files")

if __name__ == "__main__":
    main()