#!/usr/bin/env python3
"""
Real-time Assignment Grading Monitor
Monitors the submissions directory and automatically grades new submissions
"""

import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from grade_assignments import AssignmentGrader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('grading_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GradingEventHandler(FileSystemEventHandler):
    """Handle file system events for grading"""
    
    def __init__(self, grader, debounce_time=5):
        self.grader = grader
        self.debounce_time = debounce_time
        self.last_event_time = {}
        
    def on_created(self, event):
        """Handle file creation events"""
        if not event.is_directory:
            self._handle_file_event(event.src_path, "created")
    
    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory:
            self._handle_file_event(event.src_path, "modified")
    
    def _handle_file_event(self, file_path, event_type):
        """Handle file events with debouncing"""
        file_path = Path(file_path)
        
        # Check if it's a gradeable file
        if not self._is_gradeable_file(file_path):
            return
        
        # Debounce events
        current_time = time.time()
        if file_path in self.last_event_time:
            if current_time - self.last_event_time[file_path] < self.debounce_time:
                return
        
        self.last_event_time[file_path] = current_time
        
        logger.info(f"File {event_type}: {file_path}")
        
        # Wait a bit to ensure file is fully written
        time.sleep(1)
        
        # Grade the specific file
        try:
            self._grade_single_file(file_path)
        except Exception as e:
            logger.error(f"Error grading {file_path}: {e}")
    
    def _is_gradeable_file(self, file_path):
        """Check if file should be graded"""
        return (file_path.suffix in ['.py', '.ipynb'] and 
                not file_path.name.startswith('.') and
                file_path.name not in ['instruction.md', 'desktop.ini'] and
                'submissions' in str(file_path).lower())
    
    def _grade_single_file(self, file_path):
        """Grade a single file"""
        logger.info(f"Grading file: {file_path}")
        
        # Determine student name from path
        parts = file_path.parts
        submissions_idx = None
        for i, part in enumerate(parts):
            if part.lower() == 'submissions':
                submissions_idx = i
                break
        
        if submissions_idx is None or submissions_idx + 2 >= len(parts):
            logger.warning(f"Could not determine student from path: {file_path}")
            return
        
        student_name = parts[submissions_idx + 2]  # submissions/type/student_name
        
        # Analyze the file
        if file_path.suffix == '.py':
            analysis = self.grader.analyzer.analyze_python_file(str(file_path))
        elif file_path.suffix == '.ipynb':
            analysis = self.grader.analyzer.analyze_jupyter_notebook(str(file_path))
        else:
            return
        
        # Calculate grade
        grade_info = self.grader._calculate_grade(analysis)
        
        # Create result
        result = {
            'student': student_name,
            'file_name': file_path.name,
            'file_path': str(file_path),
            'grade': grade_info['grade'],
            'feedback': grade_info['feedback'],
            'ai_detection': analysis.get('ai_detection', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save individual result
        self._save_individual_result(result)
        
        # Generate feedback file
        self._generate_individual_feedback(student_name, result)
        
        logger.info(f"Graded {file_path}: {grade_info['grade']}/100")
    
    def _save_individual_result(self, result):
        """Save individual grading result"""
        results_dir = Path('grading_results')
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{result['student']}_{result['file_name']}_{timestamp}.json"
        
        with open(results_dir / filename, 'w') as f:
            json.dump(result, f, indent=2)
    
    def _generate_individual_feedback(self, student_name, result):
        """Generate feedback for individual submission"""
        feedback_dir = Path('feedback')
        feedback_dir.mkdir(exist_ok=True)
        
        feedback_file = feedback_dir / f"{student_name}_latest.md"
        
        with open(feedback_file, 'w') as f:
            f.write(f"# Latest Feedback for {student_name}\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## File: {result['file_name']}\n\n")
            f.write(f"**Grade:** {result['grade']}/100\n")
            f.write(f"**AI Detection:** {result['ai_detection'].get('likelihood', 'N/A')}\n")
            
            if result['ai_detection'].get('confidence', 0) > 50:
                f.write(f"**AI Confidence:** {result['ai_detection'].get('confidence', 0)}%\n")
            
            f.write("\n**Feedback:**\n")
            for feedback_item in result['feedback']:
                f.write(f"- {feedback_item}\n")

class GradingMonitor:
    """Main monitoring class"""
    
    def __init__(self, repo_path='.'):
        self.repo_path = Path(repo_path)
        self.submissions_path = self.repo_path / "Submissions"
        self.grader = AssignmentGrader(str(self.repo_path))
        self.observer = Observer()
        
    def start_monitoring(self):
        """Start monitoring the submissions directory"""
        if not self.submissions_path.exists():
            logger.error(f"Submissions directory not found: {self.submissions_path}")
            return
        
        event_handler = GradingEventHandler(self.grader)
        
        # Watch all subdirectories
        self.observer.schedule(event_handler, str(self.submissions_path), recursive=True)
        
        self.observer.start()
        logger.info(f"Started monitoring: {self.submissions_path}")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
            logger.info("Monitoring stopped by user")
        
        self.observer.join()
    
    def run_full_grading(self):
        """Run full grading process"""
        logger.info("Running full grading process...")
        results = self.grader.grade_all_assignments()
        
        # Save results
        with open('grading_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate feedback files
        self.grader.generate_feedback_files()
        
        logger.info("Full grading process completed")
        return results

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Assignment Grading Monitor')
    parser.add_argument('--repo-path', default='.', help='Path to the repository')
    parser.add_argument('--mode', choices=['monitor', 'grade'], default='monitor',
                       help='Mode: monitor (watch for changes) or grade (run once)')
    parser.add_argument('--full-grade', action='store_true',
                       help='Run full grading before starting monitor')
    
    args = parser.parse_args()
    
    monitor = GradingMonitor(args.repo_path)
    
    if args.mode == 'grade' or args.full_grade:
        monitor.run_full_grading()
    
    if args.mode == 'monitor':
        if args.full_grade:
            logger.info("Starting monitoring after full grading...")
        monitor.start_monitoring()

if __name__ == "__main__":
    main()