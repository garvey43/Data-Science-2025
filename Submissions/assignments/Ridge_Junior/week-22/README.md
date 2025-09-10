# Week 22 – Algorithm Complexity Analysis & Experiments

This assignment explores **algorithm efficiency** through timing, operation counting, complexity analysis, and practical comparisons.

---

## 📂 Contents

1. **Timing Functions (Q1)**
   - Compare loop-based sum (`O(n)`) with formula-based sum (`O(1)`).
   - Shows how runtime grows with input size.

2. **Counting Operations (Q2)**
   - Track number of operations in loop-based summation.
   - Demonstrates why summation grows linearly → `O(n)`.

3. **Linear vs Quadratic Growth (Q3)**
   - Implemented `linear_sum` (O(n)) and `quadratic_pairs` (O(n²)).
   - Operation counts plotted for different input sizes using Matplotlib.

4. **Searching Algorithms (Q4)**
   - **Linear Search**: Θ(n) worst case.
   - **Binary Search**: Θ(log n) with sorted input.
   - Compared against Python’s built-in `in` operator.
   - Plots show scaling differences in operation counts and execution times.

5. **Matrix Multiplication (Q5)**
   - Implemented standard 3-nested loop matrix multiplication.
   - Verified O(n³) complexity by counting operations.

6. **Best, Worst, and Average Case Analysis (Q6)**
   - Analyzed linear search performance for:
     - Best case: element at first position → O(1).
     - Worst case: element missing or last → O(n).
     - Average case: element in middle → O(n).
   - Provides clarity on case-based complexity.

---

## 🛠️ How to Run

1. Install dependencies (only needed for plots):
   ```bash
   pip install matplotlib numpy

