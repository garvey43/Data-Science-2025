# 🔧 Debugging Guide for Data Science Grader

This guide implements the debugging principles from the lecture: "Debugging is the most important skill" and provides tools to "move bugs left" through comprehensive debugging infrastructure.

## 📋 Table of Contents

1. [Debug Configuration](#debug-configuration)
2. [Validation System](#validation-system)
3. [Logging Infrastructure](#logging-infrastructure)
4. [Debug Tools](#debug-tools)
5. [Performance Profiling](#performance-profiling)
6. [Testing Strategy](#testing-strategy)
7. [Common Debugging Scenarios](#common-debugging-scenarios)
8. [Best Practices](#best-practices)

## 🛠️ Debug Configuration

### Environment Setup

```bash
# Enable debug mode
export DEBUG_MODE=true

# Or run with debug flag
python grade_assignments.py --debug
```

### Configuration Options

```python
# In config.py or grader_config.json
{
  "debug": {
    "fail_fast": true,           # Stop on first error
    "verbose_logging": true,     # Detailed logging
    "extra_validation": true,    # Additional checks
    "memory_debugging": true     # Memory tracking
  }
}
```

## ✅ Validation System

### File Validation

```python
from validation import validator

# Validate student submission
result = validator.validate_student_submission("student_file.py")
if not result['is_valid']:
    print("Validation errors:", result['errors'])
```

### Data Structure Validation

```python
from debug_tools import data_validator

# Validate grading result
grading_data = {
    'student': 'john_doe',
    'grade': 85,
    'feedback': 'Good work!'
}

result = data_validator.validate(grading_data, 'grading_result')
```

### Debug Assertions

```python
from validation import debug_assert

# Fail-fast assertions
debug_assert.assert_true(len(students) > 0, "No students found")
debug_assert.assert_equal(actual_grade, expected_grade, "Grade mismatch")
debug_assert.assert_file_exists("assignment.py", "Assignment file missing")
```

## 📊 Logging Infrastructure

### Structured Logging

```python
from debug_logger import debug_logger

# Log operation with timing
debug_logger.log_operation_start("grade_student", {"student": "john_doe"})
# ... perform grading ...
debug_logger.log_operation_end("grade_student", result)

# Log errors with context
try:
    risky_operation()
except Exception as e:
    debug_logger.log_error(e, {"operation": "risky_operation", "context": "additional_info"})
```

### Log Files Structure

```
logs/
├── grader.log          # Main application logs
├── errors.log          # Error-only logs
├── performance.log     # Performance metrics
└── debug_snapshot_*.json  # Debug snapshots
```

### Log Analysis

```bash
# View recent errors
tail -20 logs/errors.log

# Search for specific student
grep "john_doe" logs/grader.log

# Performance analysis
grep "PERF" logs/performance.log | jq '.duration' | sort -n
```

## 🔍 Debug Tools

### Memory Debugging

```python
from debug_tools import memory_debugger

# Start memory tracking
memory_debugger.start_memory_tracking()

# Take snapshots
memory_debugger.take_memory_snapshot("before_grading")
# ... grading operations ...
memory_debugger.take_memory_snapshot("after_grading")

# Compare memory usage
diff = memory_debugger.compare_memory_snapshots("before_grading", "after_grading")
print(f"Memory increase: {diff['total_memory_diff']} bytes")
```

### Performance Profiling

```python
from debug_tools import performance_profiler

# Profile operations
performance_profiler.start_timer("file_analysis")
analyze_file("large_notebook.ipynb")
duration = performance_profiler.end_timer("file_analysis")

# Get performance report
report = performance_profiler.get_performance_report()
print(f"Average file analysis time: {report['file_analysis']['average_time']:.4f}s")
```

### Debug Snapshots

```python
from debug_tools import create_debug_snapshot

# Create comprehensive debug snapshot
snapshot_file = create_debug_snapshot("grading_session")
print(f"Debug data saved to: {snapshot_file}")
```

## 📈 Performance Profiling

### Built-in Profiling

The system automatically profiles:
- File analysis operations
- Grading calculations
- Memory usage patterns
- Error frequencies

### Custom Profiling

```python
@debug_logger.log_function_call("custom_operation")
def my_function():
    # Function automatically profiled
    pass
```

### Performance Reports

```python
# Generate performance report
report = performance_profiler.get_performance_report()

for operation, stats in report.items():
    print(f"{operation}:")
    print(f"  Calls: {stats['call_count']}")
    print(f"  Avg time: {stats['average_time']:.4f}s")
    print(f"  Total time: {stats['total_time']:.4f}s")
```

## 🧪 Testing Strategy

### Focused Testing (Not Redundant)

```bash
# Run critical tests only
python test_debug_system.py

# Test specific components
python -m pytest test_debug_system.py::TestValidationSystem::test_file_validation -v

# Integration tests
python -m pytest test_debug_system.py::TestIntegration -v
```

### Test Categories

- **Unit Tests**: Individual functions and classes
- **Integration Tests**: Component interactions
- **Debug Tests**: Error handling and edge cases
- **Performance Tests**: Bottleneck identification

## 🚨 Common Debugging Scenarios

### 1. Grading Errors

```python
# Enable verbose logging
export DEBUG_MODE=true
python grade_assignments.py --student problematic_student

# Check logs
tail -f logs/errors.log
```

### 2. Memory Issues

```python
# Enable memory debugging
# Set in config: "debug.memory_debugging": true

# Monitor memory usage
from debug_tools import log_memory_usage
log_memory_usage("before_operation")
# ... operation ...
log_memory_usage("after_operation")
```

### 3. Performance Problems

```python
# Profile slow operations
performance_profiler.start_timer("slow_operation")
# ... slow code ...
performance_profiler.end_timer("slow_operation")

# Check performance logs
grep "slow_operation" logs/performance.log
```

### 4. File Processing Issues

```python
# Validate file before processing
result = validator.validate_student_submission("problematic_file.ipynb")
if not result['is_valid']:
    print("File validation failed:", result['errors'])
    # Don't process invalid files
```

## 🎯 Best Practices Implementation

### 1. Fail Fast Principle

```python
# In debug mode: stop immediately on errors
if config.get('debug.fail_fast', False):
    raise ValidationError("Critical error detected")

# In production: graceful degradation
else:
    logger.warning("Error detected, continuing...")
```

### 2. Information Gathering

```python
# Always collect context with errors
try:
    process_grading()
except Exception as e:
    debug_logger.log_error(e, {
        'student': current_student,
        'file': current_file,
        'operation': 'grading',
        'system_state': get_system_state()
    })
```

### 3. Validation Functions

```python
# Register validation for critical data
data_validator.register_validator('grading_result', validate_grading_result)

# Use throughout codebase
result = data_validator.validate(data, 'grading_result')
```

### 4. Debug-Friendly Code

```python
# Clear variable names
student_count = len(students)  # Not: sc = len(s)

# Explicit error messages
raise ValueError(f"Invalid grade {grade} for student {student}")

# Consistent formatting (use black/flake8)
def process_grading(grades, students):
    # Function clearly shows what it does
    pass
```

## 🔧 Code Quality Tools

### Automated Quality Checks

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .

# Run all quality checks
pre-commit run --all-files
```

### Pre-commit Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.10.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.0
    hooks:
      - id: mypy
```

## 📋 Quick Debug Checklist

### When Something Goes Wrong:

1. **Enable Debug Mode**
   ```bash
   export DEBUG_MODE=true
   ```

2. **Check Logs**
   ```bash
   tail -50 logs/errors.log
   tail -50 logs/grader.log
   ```

3. **Validate Input**
   ```python
   result = validator.validate_student_submission(file_path)
   ```

4. **Create Debug Snapshot**
   ```python
   snapshot = create_debug_snapshot("error_investigation")
   ```

5. **Profile Performance**
   ```python
   report = performance_profiler.get_performance_report()
   ```

6. **Run Focused Tests**
   ```bash
   python test_debug_system.py
   ```

## 🎖️ Advanced Debugging Techniques

### 1. Dynamic Breakpoints

```python
# Break on specific conditions
if student_name == "problematic_student":
    breakpoint()  # Or use your debugger
```

### 2. Memory Corruption Detection

```python
# Check for memory corruption
corruption = memory_debugger.detect_memory_corruption(data)
if corruption['corruption_detected']:
    print(f"Memory corruption at: {corruption['corrupted_positions']}")
```

### 3. Deterministic Testing

```python
# Ensure reproducible results
import random
random.seed(42)  # Fixed seed for debugging
```

## 📚 Further Reading

- [Python Debugging Documentation](https://docs.python.org/3/library/pdb.html)
- [Logging Best Practices](https://docs.python.org/3/howto/logging.html)
- [Memory Debugging with tracemalloc](https://docs.python.org/3/library/tracemalloc.html)

---

**Remember**: "Debugging is the most important skill a programmer can possess." Use these tools to move bugs left and catch issues early! 🔧✨