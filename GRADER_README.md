# Eldohub Data Science Automated Assignment Grader

## Overview

This system automates the grading of student submissions for Python (`.py`) and Jupyter Notebook (`.ipynb`) assignments. It provides detailed feedback, performs code quality analysis, detects potential AI-generated content, and integrates with GitHub Actions for event-driven grading (e.g., on Pull Request merge). It also feeds data into a Streamlit dashboard for progress tracking.

## Key Features

- ✅ **Multi-Format Support**: Grades Python (`.py`) and Jupyter Notebook (`.ipynb`) files.
- ✅ **AI Content Detection**: Identifies potential AI assistance using pattern matching.
- ✅ **Comprehensive Code Analysis**:
    - Syntax error checking
    - Basic style guide adherence (line length, indentation)
    - Documentation/docstring analysis
    - Complexity assessment (basic)
- ✅ **Automated Feedback Generation**: Creates individual Markdown feedback files for students.
- ✅ **Event-Driven Grading (GitHub Actions)**: Automatically grades submissions when a Pull Request containing changes to a student's submission folder is merged into the `main` branch.
- ✅ **Targeted Grading**: Can grade submissions for a specific student via command line.
- ✅ **Batch Grading**: Can grade *all* student submissions in the repository.
- ✅ **Dashboard Integration**: Logs key metrics for individual grading events to feed a Streamlit dashboard.
- ✅ **Email Notifications (GitHub Actions)**: Sends feedback directly to students via email upon grading (requires configuration).
- ✅ **Detailed Reporting**: Generates JSON results files and human-readable Markdown feedback.




## Usage

### Manual Grading (Local)

1.  **Grade a Specific Student:**
    Grades only the submissions for `<StudentName>` found within `Submissions/assignments/<StudentName>/`, `Submissions/groupwork/<StudentName>/`, etc. Generates their feedback file and appends data to the dashboard log.
    ```bash
    python grade_assignments.py --student <StudentName> --generate-feedback --output "grading_results_<StudentName>.json"
    ```

2.  **Grade All Students:**
    Grades submissions for all students found in the `Submissions` directory structure. Generates feedback files for all and saves a comprehensive results file. *Does not* update the dashboard log.
    ```bash
    python grade_assignments.py --generate-feedback --output "grading_results_all_students.json"
    ```

3.  **Basic Options:**
    ```bash
    # Specify repository path (if running script from elsewhere)
    python grade_assignments.py --repo-path /path/to/your/repo
    # Generate feedback files
    python grade_assignments.py --generate-feedback
    # Specify output JSON file name
    python grade_assignments.py --output custom_results.json
    ```

### GitHub Actions Workflow (`grade_on_merge.yml`)

The grading system is designed to run automatically within GitHub Actions:

1.  **Trigger:** The workflow is triggered when a Pull Request (PR) targeting the `main` branch is **closed** (merged).
2.  **Student Identification:** It identifies the student based on the directory path of the files changed in the PR (e.g., `Submissions/assignments/StudentName/...`).
3.  **Targeted Grading:** It runs the grader script targeting *only* the identified student (`--student <StudentName>`).
4.  **Feedback & Results:** It generates the student's feedback file (`feedback/<StudentName>_feedback.md`) and their specific results file (`grading_results_<StudentName>.json`).
5.  **Commit & Push:** It commits and pushes these new/updated files back to the `main` branch.
6.  **Dashboard Logging:** It appends summary data (timestamp, student name, average grade, AI flags) for this grading event to `dashboard_grading_history.csv`.
7.  **Email Notification:** It retrieves the student's email from `student_emails.json` and sends them the content of their feedback file.

### Streamlit Dashboard

A Streamlit application provides visual insights:

1.  **Run Locally:**
    ```bash
    streamlit run dashboard/app.py
    ```
2.  **Features:** Displays different views (Teacher, Dean, Stakeholder) with filters for date ranges and visualizations based on data from `dashboard_grading_history.csv`.

## Configuration

- **`student_emails.json`:** A JSON file mapping student directory names to their email addresses for notifications.
    ```json
    {
      "StudentName1": "student1@example.com",
      "StudentName2": "student2@example.com"
    }
    ```
- **Grading Logic:** Weights and penalties are defined within `grade_assignments.py` (in the `_calculate_grade` method). Modify this code to adjust scoring.
- **Dashboard:** The Streamlit dashboard (`dashboard/app.py`) reads `dashboard_grading_history.csv`. Modify `app.py` to change visualizations or views.

## Output Files

- **`grading_results_<StudentName>.json` / `grading_results_all_students.json`:** Detailed JSON reports containing analysis, grades, and feedback for submissions.
- **`feedback/<StudentName>_feedback.md`:** Human-readable Markdown feedback files generated for each student.
- **`dashboard_grading_history.csv`:** A time-series log of individual student grading events, used by the Streamlit dashboard. Appended to during the GitHub Actions PR merge workflow.
- **Terminal Output:** Summary statistics are printed to the console after each run.

## Interpreting Results

### Grades
Grades are calculated out of 100 based on:
- Syntax Errors (Deduction)
- Style Issues (Deduction)
- Documentation/Docstrings (Deduction if low)
- Potential AI Use (Deduction based on likelihood)

Ranges:
- **90-100**: Excellent work
- **80-89**: Good work with minor issues
- **70-79**: Satisfactory with some problems
- **60-69**: Below average, needs improvement
- **0-59**: Significant issues, requires attention

### AI Detection Likelihood
Indicates the confidence level that AI assistance was used:
- **Very High**: Strong indicators detected.
- **High**: Likely AI assistance.
- **Medium**: Some suspicious patterns.
- **Low/Very Low**: Appears to be original work.

## Troubleshooting

- **Ensure Dependencies:** Install required packages: `pip install -r requirements.txt` (must include `nbformat`, `streamlit`, etc.).
- **Check Logs:** Examine terminal output or logs for detailed error messages (e.g., file read errors, syntax errors in student code).
- **Verify Structure:** Ensure your repository follows the expected `Submissions/<type>/<StudentName>/...` structure.
- **GitHub Actions:** Check the Actions tab in your GitHub repository for workflow run logs if automatic grading fails.

## S.O.L.I.D Principles Applied

- **Single Responsibility Principle (SRP):** Scripts (`grade_assignments.py`, `dashboard/app.py`) and workflow jobs have focused responsibilities.
- **Open/Closed Principle (OCP):** Grading logic can be extended (e.g., adding new checks) without modifying core flow.
- **Dependency Inversion Principle (DIP):** Workflows depend on the script interface (`python grade_assignments.py ...`) rather than internal code details. Configuration (`student_emails.json`) injects dependencies.
