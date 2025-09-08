#!/usr/bin/env python3
"""
Comprehensive Logging System for Debugging
Implements the debugging principle of "information gathering" with structured logging
"""

import os
import sys
import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
from config import config

class DebugLogger:
    """
    Advanced logging system that implements debugging best practices:
    - Structured logging with context
    - Debug vs production modes
    - Performance monitoring
    - Error tracking with stack traces
    """

    def __init__(self):
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        self.debug_mode = config.get('debug.verbose_logging', False)
        self.performance_monitoring = config.get('debug.extra_validation', False)

        # Setup different loggers for different purposes
        self._setup_loggers()

        # Performance tracking
        self.performance_data = {}
        self.start_time = datetime.now()

    def _setup_loggers(self):
        """Setup multiple loggers for different logging purposes"""

        # Main application logger
        self.main_logger = logging.getLogger('grader')
        self.main_logger.setLevel(logging.DEBUG if self.debug_mode else logging.INFO)

        # Error logger (always captures errors)
        self.error_logger = logging.getLogger('grader.errors')
        self.error_logger.setLevel(logging.ERROR)

        # Performance logger
        self.perf_logger = logging.getLogger('grader.performance')
        self.perf_logger.setLevel(logging.DEBUG)

        # Remove any existing handlers
        for logger in [self.main_logger, self.error_logger, self.perf_logger]:
            logger.handlers.clear()

        # Create formatters
        if self.debug_mode:
            main_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s - %(message)s'
            error_format = '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s'
        else:
            main_format = '%(asctime)s - %(levelname)s - %(message)s'
            error_format = '%(asctime)s - %(levelname)s - %(message)s'

        main_formatter = logging.Formatter(main_format)
        error_formatter = logging.Formatter(error_format)
        perf_formatter = logging.Formatter('%(asctime)s - PERF - %(message)s')

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(main_formatter)
        self.main_logger.addHandler(console_handler)

        # File handlers
        main_file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'grader.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        main_file_handler.setFormatter(main_formatter)
        self.main_logger.addHandler(main_file_handler)

        # Error file handler (separate file for errors)
        error_file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'errors.log',
            maxBytes=10*1024*1024,
            backupCount=5
        )
        error_file_handler.setFormatter(error_formatter)
        error_file_handler.setLevel(logging.ERROR)
        self.main_logger.addHandler(error_file_handler)
        self.error_logger.addHandler(error_file_handler)

        # Performance file handler
        perf_file_handler = logging.FileHandler(self.log_dir / 'performance.log')
        perf_file_handler.setFormatter(perf_formatter)
        self.perf_logger.addHandler(perf_file_handler)

    def log_operation_start(self, operation: str, context: Optional[Dict[str, Any]] = None):
        """Log the start of an operation with timing"""
        if self.performance_monitoring:
            start_time = datetime.now()
            self.performance_data[operation] = {
                'start_time': start_time,
                'context': context or {}
            }

        self.main_logger.info(f"START: {operation}", extra={
            'operation': operation,
            'context': context
        })

    def log_operation_end(self, operation: str, result: Any = None, error: Optional[Exception] = None):
        """Log the end of an operation with timing and results"""
        if operation in self.performance_data:
            start_time = self.performance_data[operation]['start_time']
            duration = (datetime.now() - start_time).total_seconds()

            perf_data = {
                'operation': operation,
                'duration': duration,
                'result_type': type(result).__name__ if result is not None else 'None',
                'context': self.performance_data[operation]['context']
            }

            if error:
                perf_data['error'] = str(error)
                perf_data['error_type'] = type(error).__name__

            self.perf_logger.info(json.dumps(perf_data))

        status = "ERROR" if error else "SUCCESS"
        self.main_logger.info(f"END: {operation} - {status}", extra={
            'operation': operation,
            'duration': (datetime.now() - self.performance_data.get(operation, {}).get('start_time', datetime.now())).total_seconds(),
            'result': str(result)[:100] if result is not None else None,
            'error': str(error) if error else None
        })

    def log_file_analysis(self, file_path: Union[str, Path], analysis_result: Dict[str, Any]):
        """Log detailed file analysis results"""
        file_path = Path(file_path)

        log_data = {
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size if file_path.exists() else 0,
            'file_type': file_path.suffix,
            'analysis_summary': {
                'total_lines': analysis_result.get('lines_of_code', 0),
                'syntax_errors': len(analysis_result.get('syntax_errors', [])),
                'style_issues': len(analysis_result.get('style_issues', [])),
                'ai_likelihood': analysis_result.get('ai_detection', {}).get('likelihood', 'Unknown')
            }
        }

        if self.debug_mode:
            # Include full analysis in debug mode
            log_data['full_analysis'] = analysis_result

        self.main_logger.debug(f"File analysis: {file_path.name}", extra=log_data)

    def log_grading_result(self, student: str, assignment_type: str, grade: float, feedback: str):
        """Log grading results with structured data"""
        log_data = {
            'student': student,
            'assignment_type': assignment_type,
            'grade': grade,
            'grade_category': self._categorize_grade(grade),
            'feedback_length': len(feedback),
            'timestamp': datetime.now().isoformat()
        }

        self.main_logger.info(f"Graded {student} ({assignment_type}): {grade}/100", extra=log_data)

    def _categorize_grade(self, grade: float) -> str:
        """Categorize grade into letter grades"""
        if grade >= 90:
            return 'A'
        elif grade >= 80:
            return 'B'
        elif grade >= 70:
            return 'C'
        elif grade >= 60:
            return 'D'
        else:
            return 'F'

    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Log error with full context and stack trace"""
        import traceback

        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {},
            'stack_trace': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }

        # Log to error logger (always)
        self.error_logger.error(f"Exception: {type(error).__name__}: {str(error)}", extra=error_data)

        # Also log to main logger with appropriate level
        if self.debug_mode:
            self.main_logger.error(f"Exception occurred: {str(error)}", extra=error_data)
        else:
            self.main_logger.warning(f"Error: {type(error).__name__}")

    def log_system_status(self):
        """Log current system status for debugging"""
        status_data = {
            'timestamp': datetime.now().isoformat(),
            'debug_mode': self.debug_mode,
            'python_version': sys.version,
            'working_directory': os.getcwd(),
            'uptime': (datetime.now() - self.start_time).total_seconds(),
            'config_summary': {
                'fail_fast': config.get('debug.fail_fast', False),
                'verbose_logging': config.get('debug.verbose_logging', False),
                'extra_validation': config.get('debug.extra_validation', False)
            }
        }

        self.main_logger.info("System status check", extra=status_data)

    def create_debug_report(self, output_file: Optional[str] = None) -> str:
        """Create a comprehensive debug report"""
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.log_dir / f'debug_report_{timestamp}.json'

        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'working_directory': os.getcwd(),
                'debug_mode': self.debug_mode
            },
            'configuration': config.config,
            'performance_summary': self._get_performance_summary(),
            'log_files': self._get_log_file_info(),
            'recent_errors': self._get_recent_errors()
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.main_logger.info(f"Debug report created: {output_file}")
        return str(output_file)

    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        if not self.performance_data:
            return {}

        operations = list(self.performance_data.keys())
        durations = [data.get('duration', 0) for data in self.performance_data.values() if 'duration' in data]

        return {
            'total_operations': len(operations),
            'average_duration': sum(durations) / len(durations) if durations else 0,
            'slowest_operation': max(self.performance_data.items(), key=lambda x: x[1].get('duration', 0)) if self.performance_data else None
        }

    def _get_log_file_info(self) -> Dict[str, Any]:
        """Get information about log files"""
        log_files = {}
        for log_file in self.log_dir.glob('*.log'):
            stat = log_file.stat()
            log_files[log_file.name] = {
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'lines': sum(1 for _ in open(log_file, 'r', encoding='utf-8', errors='ignore'))
            }
        return log_files

    def _get_recent_errors(self) -> List[Dict[str, Any]]:
        """Get recent errors from error log"""
        error_log = self.log_dir / 'errors.log'
        if not error_log.exists():
            return []

        errors = []
        try:
            with open(error_log, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()[-10:]  # Last 10 lines

            for line in lines:
                if 'ERROR' in line or 'Exception' in line:
                    errors.append({
                        'timestamp': datetime.now().isoformat(),
                        'message': line.strip()
                    })
        except Exception:
            pass

        return errors

# Global logger instance
debug_logger = DebugLogger()

def log_function_call(func_name: str, args: tuple = None, kwargs: dict = None):
    """Decorator to log function calls with timing"""
    def decorator(func):
        def wrapper(*func_args, **func_kwargs):
            operation = f"{func.__module__}.{func.__name__}"
            context = {
                'function': func_name or func.__name__,
                'args_count': len(func_args),
                'kwargs_count': len(func_kwargs)
            }

            debug_logger.log_operation_start(operation, context)
            try:
                result = func(*func_args, **func_kwargs)
                debug_logger.log_operation_end(operation, result)
                return result
            except Exception as e:
                debug_logger.log_operation_end(operation, error=e)
                raise
        return wrapper
    return decorator

if __name__ == "__main__":
    # Test the logging system
    print("=== Debug Logger Test ===")

    # Test basic logging
    debug_logger.main_logger.info("Testing debug logger")

    # Test operation timing
    debug_logger.log_operation_start("test_operation", {"test": True})
    import time
    time.sleep(0.1)  # Simulate work
    debug_logger.log_operation_end("test_operation", "success")

    # Test error logging
    try:
        raise ValueError("Test error")
    except Exception as e:
        debug_logger.log_error(e, {"test_context": True})

    # Create debug report
    report_file = debug_logger.create_debug_report()
    print(f"Debug report created: {report_file}")

    print("✅ Debug logging system test completed")