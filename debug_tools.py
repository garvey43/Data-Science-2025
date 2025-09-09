#!/usr/bin/env python3
"""
Custom Debugging Tools and Utilities
Implements debugging principles: custom tools, validation functions, memory debugging
"""

import os
import sys
import time
import psutil
import tracemalloc
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from datetime import datetime
from config import config
from debug_logger import debug_logger

class MemoryDebugger:
    """
    Memory debugging tool that implements the "magic numbers" and guard bands principle
    """

    def __init__(self):
        self.memory_snapshots = []
        self.allocation_tracking = {}
        self.guard_band_pattern = b'\xCD\xCD\xCD\xCD'  # 0xCD pattern for debugging

    def start_memory_tracking(self):
        """Start memory tracking with detailed allocation information"""
        if config.get('debug.memory_debugging', False):
            tracemalloc.start()
            self.take_memory_snapshot("Initial")

    def take_memory_snapshot(self, label: str):
        """Take a memory snapshot for comparison"""
        if not config.get('debug.memory_debugging', False):
            return

        snapshot = tracemalloc.take_snapshot()
        self.memory_snapshots.append({
            'label': label,
            'timestamp': datetime.now(),
            'snapshot': snapshot,
            'stats': snapshot.statistics('lineno')
        })

        debug_logger.perf_logger.info(f"Memory snapshot taken: {label}")

    def compare_memory_snapshots(self, snapshot1: str, snapshot2: str) -> Dict[str, Any]:
        """Compare two memory snapshots to find memory leaks"""
        if not config.get('debug.memory_debugging', False):
            return {}

        snapshots = {s['label']: s for s in self.memory_snapshots}
        if snapshot1 not in snapshots or snapshot2 not in snapshots:
            return {}

        stats1 = snapshots[snapshot1]['stats']
        stats2 = snapshots[snapshot2]['stats']

        # Find differences
        differences = []
        for stat1, stat2 in zip(stats1, stats2):
            if stat1.size != stat2.size:
                differences.append({
                    'file': stat1.traceback[0].filename,
                    'line': stat1.traceback[0].lineno,
                    'size_diff': stat2.size - stat1.size,
                    'count_diff': stat2.count - stat1.count
                })

        return {
            'total_memory_diff': sum(d['size_diff'] for d in differences),
            'differences': differences
        }

    def detect_memory_corruption(self, data: bytes, guard_size: int = 4) -> Dict[str, Any]:
        """Detect memory corruption using guard bands"""
        result = {
            'corruption_detected': False,
            'corrupted_positions': [],
            'expected_pattern': self.guard_band_pattern[:guard_size].hex()
        }

        # Check guard bands at start and end
        if len(data) < guard_size * 2:
            result['error'] = "Data too small for guard band check"
            return result

        start_guard = data[:guard_size]
        end_guard = data[-guard_size:]

        if start_guard != self.guard_band_pattern[:guard_size]:
            result['corruption_detected'] = True
            result['corrupted_positions'].append('start')

        if end_guard != self.guard_band_pattern[:guard_size]:
            result['corruption_detected'] = True
            result['corrupted_positions'].append('end')

        return result

class DataStructureValidator:
    """
    Validation functions for data structures - implements "semantic validation"
    """

    def __init__(self):
        self.validation_functions = {}

    def register_validator(self, data_type: str, validator_func: Callable):
        """Register a validation function for a data type"""
        self.validation_functions[data_type] = validator_func

    def validate(self, data: Any, data_type: str, context: str = "") -> Dict[str, Any]:
        """Validate data structure using registered validator"""
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'data_type': data_type,
            'context': context
        }

        if data_type in self.validation_functions:
            try:
                validation_result = self.validation_functions[data_type](data)
                result.update(validation_result)
            except Exception as e:
                result['errors'].append(f"Validation exception: {str(e)}")
                result['is_valid'] = False
        else:
            result['warnings'].append(f"No validator registered for type: {data_type}")

        # Log validation result
        debug_logger.log_validation_result(result, context)

        return result

    def validate_grading_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate grading result data structure"""
        errors = []
        warnings = []

        # Required fields
        required_fields = ['student', 'file_name', 'grade', 'feedback', 'ai_detection']
        for field in required_fields:
            if field not in result:
                errors.append(f"Missing required field: {field}")

        # Grade validation
        if 'grade' in result:
            grade = result['grade']
            if not isinstance(grade, (int, float)):
                errors.append(f"Grade must be numeric, got {type(grade)}")
            elif not (0 <= grade <= 100):
                errors.append(f"Grade must be between 0-100, got {grade}")

        # AI detection validation
        if 'ai_detection' in result:
            ai_detect = result['ai_detection']
            if not isinstance(ai_detect, dict):
                errors.append("AI detection must be a dictionary")
            elif 'likelihood' not in ai_detect:
                warnings.append("AI detection missing likelihood field")

        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    def validate_student_directory(self, dir_path: Path) -> Dict[str, Any]:
        """Validate student submission directory structure"""
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'file_count': 0,
            'valid_files': [],
            'invalid_files': []
        }

        if not dir_path.exists():
            result['errors'].append(f"Directory does not exist: {dir_path}")
            result['is_valid'] = False
            return result

        allowed_extensions = config.get('file_extensions', ['.py', '.ipynb'])
        excluded_files = config.get('excluded_files', [])

        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                result['file_count'] += 1

                if file_path.name in excluded_files:
                    result['invalid_files'].append(str(file_path))
                    result['warnings'].append(f"Excluded file found: {file_path.name}")
                elif file_path.suffix in allowed_extensions:
                    result['valid_files'].append(str(file_path))
                else:
                    result['invalid_files'].append(str(file_path))
                    result['errors'].append(f"Invalid file extension: {file_path}")

        return result

class PerformanceProfiler:
    """
    Performance profiling tool for identifying bottlenecks
    """

    def __init__(self):
        self.timers = {}
        self.call_counts = {}
        self.performance_data = []

    def start_timer(self, name: str):
        """Start a performance timer"""
        self.timers[name] = time.perf_counter()

    def end_timer(self, name: str) -> float:
        """End a performance timer and return duration"""
        if name not in self.timers:
            debug_logger.main_logger.warning(f"Timer '{name}' was not started")
            return 0.0

        duration = time.perf_counter() - self.timers[name]
        self.call_counts[name] = self.call_counts.get(name, 0) + 1

        # Log performance data
        perf_data = {
            'operation': name,
            'duration': duration,
            'call_count': self.call_counts[name],
            'timestamp': datetime.now().isoformat()
        }
        self.performance_data.append(perf_data)

        debug_logger.perf_logger.info(json.dumps(perf_data))
        return duration

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        if not self.performance_data:
            return {}

        # Group by operation
        operations = {}
        for data in self.performance_data:
            op = data['operation']
            if op not in operations:
                operations[op] = []
            operations[op].append(data['duration'])

        # Calculate statistics
        report = {}
        for op, durations in operations.items():
            report[op] = {
                'call_count': len(durations),
                'total_time': sum(durations),
                'average_time': sum(durations) / len(durations),
                'min_time': min(durations),
                'max_time': max(durations)
            }

        return report

class DebugAssertion:
    """
    Custom assertion system that implements "fail fast" debugging
    """

    def __init__(self):
        self.assertion_history = []
        self.fail_fast = config.get('debug.fail_fast', False)

    def assert_true(self, condition: bool, message: str, context: Optional[Dict[str, Any]] = None):
        """Assert that condition is true"""
        if not condition:
            self._handle_assertion_failure(f"Assertion failed: {message}", context)

    def assert_equal(self, actual: Any, expected: Any, message: str = "", context: Optional[Dict[str, Any]] = None):
        """Assert that actual equals expected"""
        if actual != expected:
            error_msg = f"Values not equal: expected {expected}, got {actual}"
            if message:
                error_msg = f"{message} - {error_msg}"
            self._handle_assertion_failure(error_msg, context)

    def assert_not_none(self, value: Any, message: str = "", context: Optional[Dict[str, Any]] = None):
        """Assert that value is not None"""
        if value is None:
            error_msg = f"Value is None"
            if message:
                error_msg = f"{message} - {error_msg}"
            self._handle_assertion_failure(error_msg, context)

    def assert_file_exists(self, file_path: Union[str, Path], message: str = "", context: Optional[Dict[str, Any]] = None):
        """Assert that file exists"""
        path = Path(file_path)
        if not path.exists():
            error_msg = f"File does not exist: {path}"
            if message:
                error_msg = f"{message} - {error_msg}"
            self._handle_assertion_failure(error_msg, context)

    def _handle_assertion_failure(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Handle assertion failure based on debug mode"""
        assertion_data = {
            'message': message,
            'context': context or {},
            'timestamp': datetime.now().isoformat(),
            'call_stack': self._get_call_stack()
        }

        self.assertion_history.append(assertion_data)

        # Log the assertion failure
        debug_logger.error_logger.error(f"ASSERTION FAILED: {message}", extra=assertion_data)

        if self.fail_fast:
            # In debug mode, raise exception to stop execution
            raise AssertionError(message)
        else:
            # In production mode, just log warning
            debug_logger.main_logger.warning(f"Assertion failed (continuing): {message}")

    def _get_call_stack(self) -> List[Dict[str, Any]]:
        """Get call stack information for debugging"""
        import inspect

        stack = []
        for frame_info in inspect.stack()[2:]:  # Skip this method and the caller
            stack.append({
                'file': frame_info.filename,
                'line': frame_info.lineno,
                'function': frame_info.function,
                'code': frame_info.code_context[0].strip() if frame_info.code_context else ""
            })

        return stack

    def get_assertion_report(self) -> Dict[str, Any]:
        """Get report of all assertion failures"""
        return {
            'total_assertions': len(self.assertion_history),
            'failures': self.assertion_history,
            'fail_fast_enabled': self.fail_fast
        }

# Global instances
memory_debugger = MemoryDebugger()
data_validator = DataStructureValidator()
performance_profiler = PerformanceProfiler()
debug_assert = DebugAssertion()

# Register built-in validators
data_validator.register_validator('grading_result', data_validator.validate_grading_result)
data_validator.register_validator('student_directory', data_validator.validate_student_directory)

def debug_breakpoint(condition: bool = True, message: str = "Debug breakpoint reached"):
    """Conditional breakpoint for debugging"""
    if condition and config.get('debug.verbose_logging', False):
        debug_logger.main_logger.info(f"🔴 {message}")
        if config.get('debug.fail_fast', False):
            breakpoint()  # This will trigger debugger if available

def log_memory_usage(label: str = ""):
    """Log current memory usage"""
    if config.get('debug.memory_debugging', False):
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        memory_data = {
            'label': label,
            'rss': memory_info.rss,  # Resident Set Size
            'vms': memory_info.vms,  # Virtual Memory Size
            'timestamp': datetime.now().isoformat()
        }

        debug_logger.perf_logger.info(f"MEMORY: {json.dumps(memory_data)}")

def create_debug_snapshot(operation: str):
    """Create a comprehensive debug snapshot"""
    snapshot = {
        'timestamp': datetime.now().isoformat(),
        'operation': operation,
        'memory_usage': psutil.Process(os.getpid()).memory_info()._asdict(),
        'performance_report': performance_profiler.get_performance_report(),
        'assertion_report': debug_assert.get_assertion_report(),
        'config_summary': {
            'debug_mode': config.is_debug_mode(),
            'fail_fast': config.get('debug.fail_fast', False),
            'verbose_logging': config.get('debug.verbose_logging', False)
        }
    }

    snapshot_file = Path('logs') / f'debug_snapshot_{int(time.time())}.json'
    snapshot_file.parent.mkdir(exist_ok=True)

    with open(snapshot_file, 'w') as f:
        json.dump(snapshot, f, indent=2, default=str)

    debug_logger.main_logger.info(f"Debug snapshot created: {snapshot_file}")
    return snapshot_file

if __name__ == "__main__":
    print("=== Debug Tools Test ===")

    # Test memory debugger
    memory_debugger.start_memory_tracking()
    log_memory_usage("test_start")

    # Test data validator
    test_result = {
        'student': 'test_student',
        'file_name': 'test.py',
        'grade': 85.5,
        'feedback': 'Good work!',
        'ai_detection': {'likelihood': 'Low'}
    }

    validation = data_validator.validate(test_result, 'grading_result', 'test_validation')
    print(f"Validation result: {validation['is_valid']}")

    # Test performance profiler
    performance_profiler.start_timer('test_operation')
    time.sleep(0.01)  # Simulate work
    duration = performance_profiler.end_timer('test_operation')
    print(f"Operation took: {duration:.4f} seconds")

    # Test debug assertions
    try:
        debug_assert.assert_true(1 + 1 == 2, "Math works")
        print("✅ Assertion passed")
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")

    try:
        debug_assert.assert_equal(2 + 2, 5, "This should fail")
        print("❌ This should not print")
    except AssertionError as e:
        print(f"✅ Assertion correctly failed: {e}")

    # Create debug snapshot
    snapshot_file = create_debug_snapshot("test_completion")
    print(f"Debug snapshot: {snapshot_file}")

    print("✅ Debug tools test completed")