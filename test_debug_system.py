#!/usr/bin/env python3
"""
Testing Infrastructure for Debug System
Implements testing principles: focus on what's not controlled, avoid redundant testing
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config, ConfigManager
from validation import validator, ValidationError, debug_assert
from debug_logger import debug_logger
from debug_tools import data_validator, performance_profiler, debug_assert as tools_assert

class TestConfigSystem(unittest.TestCase):
    """Test configuration management system"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.json"

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)

    def test_debug_mode_detection(self):
        """Test automatic debug mode detection"""
        # Test environment variable detection
        with patch.dict(os.environ, {'DEBUG_MODE': 'true'}):
            test_config = ConfigManager()
            self.assertTrue(test_config.is_debug_mode())

        # Test source directory detection
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = True
            with patch('os.getcwd', return_value='/some/path'):
                test_config = ConfigManager()
                # Should detect based on file existence

    def test_config_overrides(self):
        """Test debug vs production configuration overrides"""
        config_data = {
            "debug": {
                "fail_fast": True,
                "verbose_logging": True
            }
        }

        with patch.object(ConfigManager, '_load_config', return_value=config_data):
            test_config = ConfigManager()
            test_config.debug_mode = True

            # In debug mode, should have debug settings
            self.assertTrue(test_config.get('debug.fail_fast', False))

class TestValidationSystem(unittest.TestCase):
    """Test validation system"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_file_validation(self):
        """Test file validation functionality"""
        # Create a test Python file
        test_file = Path(self.temp_dir) / "test.py"
        test_file.write_text("print('Hello, World!')\n")

        result = validator.validate_student_submission(str(test_file))
        self.assertTrue(result['is_valid'])
        self.assertIn('test.py', result['valid_files'])

    def test_invalid_file_extension(self):
        """Test validation of invalid file extensions"""
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("This is not code")

        result = validator.validate_student_submission(str(test_file))
        self.assertFalse(result['is_valid'])
        self.assertIn('Invalid file extension', ' '.join(result['errors']))

    def test_debug_assertions(self):
        """Test debug assertion system"""
        # Test successful assertion
        debug_assert.assert_true(True, "This should pass")

        # Test failing assertion (should not raise in test mode)
        with patch.object(config, 'get', return_value=False):  # Disable fail_fast
            debug_assert.assert_true(False, "This should fail but not raise")

class TestLoggingSystem(unittest.TestCase):
    """Test logging system"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "logs"
        self.log_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch('debug_logger.debug_logger.main_logger')
    def test_operation_logging(self, mock_logger):
        """Test operation start/end logging"""
        debug_logger.log_operation_start("test_operation", {"param": "value"})
        debug_logger.log_operation_end("test_operation", "success")

        # Verify logging calls were made
        self.assertTrue(mock_logger.info.called)

    def test_performance_logging(self):
        """Test performance data logging"""
        performance_profiler.start_timer("test_perf")
        import time
        time.sleep(0.01)
        duration = performance_profiler.end_timer("test_perf")

        self.assertGreater(duration, 0)
        report = performance_profiler.get_performance_report()
        self.assertIn("test_perf", report)

class TestDebugTools(unittest.TestCase):
    """Test debugging tools"""

    def test_data_validation(self):
        """Test data structure validation"""
        test_data = {
            'student': 'test_student',
            'file_name': 'test.py',
            'grade': 85.5,
            'feedback': 'Good work!',
            'ai_detection': {'likelihood': 'Low'}
        }

        result = data_validator.validate(test_data, 'grading_result')
        self.assertTrue(result['is_valid'])

    def test_invalid_grading_data(self):
        """Test validation of invalid grading data"""
        invalid_data = {
            'student': 'test_student',
            # Missing required fields
            'grade': 'not_a_number'
        }

        result = data_validator.validate(invalid_data, 'grading_result')
        self.assertFalse(result['is_valid'])
        self.assertGreater(len(result['errors']), 0)

    def test_performance_profiling(self):
        """Test performance profiling functionality"""
        performance_profiler.start_timer("test_profile")
        import time
        time.sleep(0.01)
        duration = performance_profiler.end_timer("test_profile")

        self.assertGreater(duration, 0)

        report = performance_profiler.get_performance_report()
        self.assertIn("test_profile", report)
        self.assertGreater(report["test_profile"]["average_time"], 0)

class TestIntegration(unittest.TestCase):
    """Integration tests for the debug system"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.submissions_dir = Path(self.temp_dir) / "Submissions"
        self.submissions_dir.mkdir()

        # Create test student directory
        self.student_dir = self.submissions_dir / "test_student"
        self.student_dir.mkdir()

        # Create test file
        self.test_file = self.student_dir / "test_assignment.py"
        self.test_file.write_text("""
def calculate_average(numbers):
    '''Calculate average of a list of numbers'''
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

# Test the function
if __name__ == "__main__":
    test_data = [1, 2, 3, 4, 5]
    result = calculate_average(test_data)
    print(f"Average: {result}")
""")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_full_validation_workflow(self):
        """Test complete validation workflow"""
        # Validate file
        file_result = validator.validate_student_submission(str(self.test_file))
        self.assertTrue(file_result['is_valid'])

        # Validate directory
        dir_result = data_validator.validate(self.student_dir, 'student_directory')
        self.assertTrue(dir_result['is_valid'])

        # Test logging integration
        debug_logger.log_file_analysis(str(self.test_file), {
            'lines_of_code': 15,
            'syntax_errors': [],
            'style_issues': [],
            'ai_detection': {'likelihood': 'Low'}
        })

    def test_error_handling(self):
        """Test error handling throughout the system"""
        # Test with non-existent file
        result = validator.validate_student_submission("/non/existent/file.py")
        self.assertFalse(result['is_valid'])

        # Test with invalid directory
        invalid_dir = Path("/non/existent/directory")
        dir_result = data_validator.validate(invalid_dir, 'student_directory')
        self.assertFalse(dir_result['is_valid'])

class TestDebuggingPrinciples(unittest.TestCase):
    """Test that debugging principles from the lecture are implemented"""

    def test_fail_fast_principle(self):
        """Test fail-fast debugging principle"""
        # In debug mode with fail_fast enabled, assertions should raise exceptions
        with patch.object(config, 'get', return_value=True):  # Enable fail_fast
            with self.assertRaises(AssertionError):
                debug_assert.assert_true(False, "Test fail-fast")

    def test_information_gathering(self):
        """Test information gathering principle"""
        # Create a debug snapshot
        from debug_tools import create_debug_snapshot
        snapshot_file = create_debug_snapshot("test_principle")
        self.assertTrue(snapshot_file.exists())

        # Verify snapshot contains expected information
        import json
        with open(snapshot_file, 'r') as f:
            snapshot = json.load(f)

        self.assertIn('timestamp', snapshot)
        self.assertIn('operation', snapshot)
        self.assertIn('config_summary', snapshot)

    def test_validation_functions(self):
        """Test validation functions principle"""
        # Test that we have validation functions for critical data structures
        self.assertIn('grading_result', data_validator.validation_functions)

        # Test validation execution
        test_grading = {
            'student': 'test',
            'file_name': 'test.py',
            'grade': 85,
            'feedback': 'Good',
            'ai_detection': {'likelihood': 'Low'}
        }

        result = data_validator.validate(test_grading, 'grading_result')
        self.assertTrue(result['is_valid'])

def run_focused_tests():
    """Run tests focusing on critical functionality (not redundant testing)"""
    # Focus on external dependencies and complex logic
    suite = unittest.TestSuite()

    # Core functionality tests
    suite.addTest(TestConfigSystem('test_debug_mode_detection'))
    suite.addTest(TestValidationSystem('test_file_validation'))
    suite.addTest(TestDebugTools('test_data_validation'))

    # Integration tests
    suite.addTest(TestIntegration('test_full_validation_workflow'))

    # Debugging principles tests
    suite.addTest(TestDebuggingPrinciples('test_fail_fast_principle'))
    suite.addTest(TestDebuggingPrinciples('test_information_gathering'))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()

if __name__ == '__main__':
    print("=== Debug System Test Suite ===")
    print("Testing debugging principles implementation...")

    success = run_focused_tests()

    if success:
        print("✅ All critical tests passed!")
        print("Debug system is ready for production use.")
    else:
        print("❌ Some tests failed. Check the output above.")
        sys.exit(1)