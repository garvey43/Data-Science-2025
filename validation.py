#!/usr/bin/env python3
"""
Validation and Error Handling System
Implements "fail fast" debugging principles with comprehensive validation
"""

import os
import sys
import traceback
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
from config import config

class ValidationError(Exception):
    """Custom exception for validation failures"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}
        self.caller_info = self._get_caller_info()

    def _get_caller_info(self) -> Dict[str, Any]:
        """Get information about where the error occurred"""
        frame = inspect.currentframe()
        try:
            # Go up the stack to find the actual caller
            caller_frame = frame.f_back.f_back
            if caller_frame:
                return {
                    'file': caller_frame.f_code.co_filename,
                    'line': caller_frame.f_lineno,
                    'function': caller_frame.f_code.co_name
                }
        finally:
            del frame
        return {}

class DebugValidator:
    """
    Comprehensive validation system that implements debugging best practices:
    - Fail fast in debug mode
    - Provide maximum information for debugging
    - Validate data structures thoroughly
    """

    def __init__(self):
        self.fail_fast = config.get('debug.fail_fast', False)
        self.extra_validation = config.get('debug.extra_validation', False)
        self.validation_history = []

    def validate_file_exists(self, file_path: Union[str, Path], context: str = "") -> Path:
        """Validate that a file exists, with detailed error reporting"""
        path = Path(file_path)

        if not path.exists():
            error_msg = f"File not found: {path}"
            if context:
                error_msg += f" (Context: {context})"

            if self.fail_fast:
                raise ValidationError(error_msg, {
                    'file_path': str(path),
                    'absolute_path': str(path.absolute()),
                    'context': context,
                    'working_directory': os.getcwd(),
                    'files_in_directory': list(path.parent.glob('*')) if path.parent.exists() else []
                })

        return path

    def validate_directory(self, dir_path: Union[str, Path], context: str = "") -> Path:
        """Validate that a directory exists and is readable"""
        path = Path(dir_path)

        if not path.exists():
            error_msg = f"Directory not found: {path}"
            if context:
                error_msg += f" (Context: {context})"

            if self.fail_fast:
                raise ValidationError(error_msg, {
                    'directory_path': str(path),
                    'context': context,
                    'parent_exists': path.parent.exists(),
                    'contents_of_parent': list(path.parent.glob('*')) if path.parent.exists() else []
                })

        if not path.is_dir():
            error_msg = f"Path exists but is not a directory: {path}"
            if self.fail_fast:
                raise ValidationError(error_msg, {
                    'path': str(path),
                    'is_file': path.is_file(),
                    'permissions': oct(path.stat().st_mode) if path.exists() else 'N/A'
                })

        return path

    def validate_student_submission(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Comprehensive validation of student submission files"""
        path = Path(file_path)
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'file_info': {}
        }

        try:
            # Basic file validation
            if not path.exists():
                validation_result['errors'].append(f"File does not exist: {path}")
                validation_result['is_valid'] = False
                return validation_result

            # File extension validation
            allowed_extensions = config.get('file_extensions', ['.py', '.ipynb'])
            if path.suffix not in allowed_extensions:
                validation_result['errors'].append(f"Invalid file extension: {path.suffix}. Allowed: {allowed_extensions}")
                validation_result['is_valid'] = False

            # Excluded files check
            excluded_files = config.get('excluded_files', [])
            if path.name in excluded_files:
                validation_result['warnings'].append(f"File is in excluded list: {path.name}")

            # File size validation (prevent extremely large files)
            file_size = path.stat().st_size
            max_size = 50 * 1024 * 1024  # 50MB limit
            if file_size > max_size:
                validation_result['errors'].append(f"File too large: {file_size} bytes (max: {max_size})")
                validation_result['is_valid'] = False

            # Extra validation in debug mode
            if self.extra_validation and path.suffix == '.py':
                validation_result.update(self._validate_python_file(path))

            # File info for debugging
            validation_result['file_info'] = {
                'path': str(path),
                'size': file_size,
                'modified': path.stat().st_mtime,
                'permissions': oct(path.stat().st_mode)
            }

        except Exception as e:
            validation_result['errors'].append(f"Validation error: {str(e)}")
            validation_result['is_valid'] = False

        return validation_result

    def _validate_python_file(self, file_path: Path) -> Dict[str, Any]:
        """Extra validation for Python files in debug mode"""
        result = {'syntax_check': {}, 'import_check': {}}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Syntax validation
            try:
                compile(content, str(file_path), 'exec')
                result['syntax_check'] = {'valid': True}
            except SyntaxError as e:
                result['syntax_check'] = {
                    'valid': False,
                    'error': str(e),
                    'line': e.lineno,
                    'offset': e.offset
                }

            # Import validation (check for common import issues)
            import_issues = []
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if line.startswith('import ') or line.startswith('from '):
                    # Check for potential import issues
                    if 'import *' in line:
                        import_issues.append(f"Line {i}: Wildcard import detected")
                    elif 'import os' in line and 'import sys' in content:
                        import_issues.append(f"Line {i}: Consider organizing imports")

            result['import_check'] = {'issues': import_issues}

        except Exception as e:
            result['file_read_error'] = str(e)

        return result

    def validate_data_structure(self, data: Any, schema: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        """Validate data structure against schema (debug-friendly validation)"""
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }

        def validate_recursive(obj: Any, schema_part: Dict[str, Any], path: str = ""):
            if not isinstance(schema_part, dict):
                return

            # Type validation
            expected_type = schema_part.get('type')
            if expected_type and not isinstance(obj, expected_type):
                result['errors'].append(f"{path}: Expected {expected_type.__name__}, got {type(obj).__name__}")

            # Required fields
            required = schema_part.get('required', [])
            if isinstance(obj, dict):
                for field in required:
                    if field not in obj:
                        result['errors'].append(f"{path}: Missing required field '{field}'")

            # Field validation
            fields = schema_part.get('fields', {})
            if isinstance(obj, dict):
                for field_name, field_schema in fields.items():
                    if field_name in obj:
                        validate_recursive(obj[field_name], field_schema, f"{path}.{field_name}")

        try:
            validate_recursive(data, schema)
        except Exception as e:
            result['errors'].append(f"Validation exception: {str(e)}")

        result['is_valid'] = len(result['errors']) == 0
        return result

    def log_validation_result(self, result: Dict[str, Any], context: str = ""):
        """Log validation results with appropriate detail level"""
        import logging
        logger = logging.getLogger(__name__)

        if not result.get('is_valid', True):
            logger.error(f"Validation failed for {context}: {result.get('errors', [])}")

        if result.get('warnings'):
            logger.warning(f"Validation warnings for {context}: {result['warnings']}")

        if config.get('debug.verbose_logging', False):
            logger.debug(f"Validation details for {context}: {result}")

    def assert_condition(self, condition: bool, message: str, context: Optional[Dict[str, Any]] = None):
        """Assert condition with detailed error reporting in debug mode"""
        if not condition:
            if self.fail_fast:
                raise ValidationError(message, context)
            else:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Assertion failed: {message}")
                if context:
                    logger.error(f"Context: {context}")

# Global validator instance
validator = DebugValidator()

def debug_assert(condition: bool, message: str, context: Optional[Dict[str, Any]] = None):
    """Global debug assertion function - fails fast in debug mode"""
    validator.assert_condition(condition, message, context)

def validate_file_operation(func: Callable) -> Callable:
    """Decorator to add file operation validation"""
    def wrapper(*args, **kwargs):
        # Add validation logic here
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if config.get('debug.fail_fast', False):
                raise ValidationError(f"File operation failed: {str(e)}", {
                    'function': func.__name__,
                    'args': args,
                    'kwargs': kwargs
                })
            else:
                raise
    return wrapper

if __name__ == "__main__":
    # Test the validation system
    print("=== Validation System Test ===")

    # Test file validation
    test_file = Path("README.md")
    if test_file.exists():
        result = validator.validate_student_submission(test_file)
        print(f"File validation result: {result['is_valid']}")
        if result['errors']:
            print(f"Errors: {result['errors']}")
        if result['warnings']:
            print(f"Warnings: {result['warnings']}")

    # Test debug assertion
    try:
        debug_assert(1 + 1 == 2, "Math should work")
        print("✅ Debug assertion passed")
    except ValidationError as e:
        print(f"❌ Debug assertion failed: {e}")

    try:
        debug_assert(1 + 1 == 3, "This should fail", {"expected": 2, "got": 3})
        print("❌ This should not print")
    except ValidationError as e:
        print(f"✅ Debug assertion correctly failed: {e}")
        print(f"Context: {e.context}")