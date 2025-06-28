# Assignment Grader Documentation

## Overview
This automated grading system evaluates student submissions for Python assignments and Jupyter notebooks. It provides comprehensive feedback and can detect potential AI-generated content.

## Features
- ✅ Supports Python (.py) and Jupyter Notebook (.ipynb) files
- ✅ AI content detection with configurable sensitivity
- ✅ Comprehensive code analysis (syntax, style, complexity)
- ✅ Automated feedback generation
- ✅ GitHub Actions integration
- ✅ Detailed reporting and analytics

## Usage

### Manual Grading
```bash
# Grade all assignments
python grade_assignments.py

# Grade with feedback generation
python grade_assignments.py --generate-feedback

# Grade specific directory
python grade_assignments.py --repo-path /path/to/repo --output results.json
```

### GitHub Actions
The grader runs automatically when:
- New submissions are pushed to the `Submissions/` directory
- Pull requests are created with submission changes
- Manually triggered via GitHub Actions

## Configuration
Edit `grader_config.json` to customize:
- Grading weights and penalties
- AI detection sensitivity
- File types to process
- Feedback settings

## Output Files
- `grading_results.json`: Complete grading results
- `feedback/`: Individual student feedback files
- `grader.log`: Detailed operation logs
- `GRADING_REPORT.md`: Summary report

## Interpreting Results

### Grades
- **90-100**: Excellent work
- **80-89**: Good work with minor issues
- **70-79**: Satisfactory with some problems
- **60-69**: Below average, needs improvement
- **0-59**: Significant issues, requires attention

### AI Detection
- **Very High**: Strong indicators of AI assistance
- **High**: Likely AI assistance
- **Medium**: Some suspicious patterns
- **Low/Very Low**: Appears to be original work

## Troubleshooting
- Check `grader.log` for detailed error messages
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify file permissions and repository structure
- Test with `python scripts/test_grader.py`

## Support
For issues or questions, please check the grader logs and ensure your repository structure matches the expected format.
