# student managment system
#john nyangara jnyangara303@gmail.com
#teddy Kipkoech amdanyteddy@gmail.com
#felix Onderi feonjunior8@gmail.com
#kigen yego kigenyego@gmail.com

import json
import os
import uuid
from typing import List, Dict, Optional
import sys
import unittest


class Student:
    """Class representing a student with their details."""

    def __init__(self, name: str, age: int, class_name: str, subjects: List[str], grades: List[float], id: str = None):
        self.id = id if id else str(uuid.uuid4())  # Generate ID if not provided
        self.name = name
        self.age = age
        self.class_name = class_name
        self.subjects = subjects
        self.grades = grades

    def calculate_average_grade(self) -> float:
        """Calculate the average grade for the student."""
        return sum(self.grades) / len(self.grades) if self.grades else 0.0

    def to_dict(self) -> Dict:
        """Convert student object to dictionary for JSON storage."""
        return {
            'name': self.name,
            'age': self.age,  # Store as int in JSON
            'class_name': self.class_name,
            'subjects': self.subjects,
            'grades': self.grades
        }

#class containing methods to add delete search edit and display existing records
class StudentManagementSystem:
    """Main class to manage student records and operations."""
    def __init__(self, filename: str = 'student.json'):
        self.filename = filename
        self.students: List[Student] = []
        self.load_students()
    #function to load records for students from a json file
    def load_students(self) -> None:
        """Load students from JSON file."""
        try:
            if not os.path.exists(self.filename):
                print(f"No file found at {self.filename}. Starting with empty student list.")
                return
            with open(self.filename, 'r') as file:
                reader = json.load(file)  # Load JSON data
                self.students = []
                for data in reader:
                    student = Student(
                        name=data['name'],
                        age=int(data['age']),
                        class_name=data['class_name'],
                        subjects=data['subjects'],
                        grades=[float(grade) for grade in data['grades']]
                    )
                    self.students.append(student)
        except FileNotFoundError:
            print(f"Error: {self.filename} not found.")
        except json.JSONDecodeError:
            print(f"Error: {self.filename} contains invalid JSON.")
        except KeyError as e:
            print(f"Error: Missing key {e} in student data.")
        except Exception as e:
            print(f"Error loading students: {e}")

    #function to save a new student and his/her related details in the json file
    def save_students(self) -> None:
        """Save students to JSON file."""
        try:
            with open(self.filename, 'w') as file:
                student_data = [student.to_dict() for student in self.students]
                json.dump(student_data, file, indent=4)
        except Exception as e:
            print(f"Error saving students: {e}")

    #function to add a new student to the json file student.json
    def add_student(self, name: str, age: int, class_name: str, subjects: List[str], grades: List[float]) -> None:
        """Add a new student to the system."""
        try:
            if not (0 <= age <= 100):
                raise ValueError("Invalid age")
            if len(subjects) != len(grades):
                raise ValueError("Number of subjects must match number of grades")
            student = Student(name, age, class_name, subjects, grades)
            self.students.append(student)
            self.save_students()
            print(f"Student {name} added successfully!")
        except ValueError as e:
            print(f"Error: {e}")

    #function that reads data from the students json file and renders all details for each student
    def view_all_students(self) -> None:
        """Display all students' information."""
        if not self.students:
            print("No students found.")
            return
        for student in self.students:
            print(f"ID: {student.id}")
            print(f"Name: {student.name}")
            print(f"Age: {student.age}")
            print(f"Class: {student.class_name}")
            print(f"Subjects: {', '.join(student.subjects)}")
            print(f"Grades: {', '.join(map(str, student.grades))}")
            print(f"Average Grade: {student.calculate_average_grade():.2f}")
            print("-" * 30)
    #function to search a student record based on name
    def search_student(self, search_term: str) -> Optional[Student]:
        """Search for a student by name or ID."""
        for student in self.students:
            if search_term.lower() in student.name.lower() or search_term == student.id:
                return student
        return None

    #function that implements changes to an existing record
    def edit_student(self, student_id: str, name: Optional[str] = None, age: Optional[int] = None,
                     class_name: Optional[str] = None, subjects: Optional[List[str]] = None,
                     grades: Optional[List[float]] = None) -> None:
        """Edit an existing student's details."""
        try:
            student = self.search_student(student_id)
            if not student:
                print("Student not found!")
                return
            if name:
                student.name = name
            if age is not None:
                if not (0 <= age <= 100):
                    raise ValueError("Invalid age")
                student.age = age
            if class_name:
                student.class_name = class_name
            if subjects:
                student.subjects = subjects
            if grades:
                if len(student.subjects) != len(grades):
                    raise ValueError("Number of subjects must match number of grades")
                student.grades = grades
            self.save_students()
            print("Student updated successfully!")
        except ValueError as e:
            print(f"Error: {e}")
    #funtion that deletes a students record  by providing student name
    def delete_student(self, student_id: str) -> None:
        """Delete a student from the system."""
        student = self.search_student(student_id)
        if student:
            self.students.remove(student)
            self.save_students()
            print("Student deleted successfully!")
        else:
            print("Student not found!")
    #fuction that sorts the student records based on name or grade
    def sort_students(self, sort_by: str = 'name') -> List[Student]:
        """Sort students by name or average grade."""
        if sort_by == 'name':
            return sorted(self.students, key=lambda x: x.name.lower())
        elif sort_by == 'grade':
            return sorted(self.students, key=lambda x: x.calculate_average_grade(), reverse=True)
        return self.students

#function that renders command line interface for the user to choose an operation ,it acts as an interface between the user and the system(api)
def run_cli():
    """Run the command-line interface for the student management system."""
    sms = StudentManagementSystem('student.json')  # Use student.json as requested
    while True:
        print("\nSmartStudent Management System")
        print("1. Add Student")
        print("2. View All Students")
        print("3. Search Student")
        print("4. Edit Student")
        print("5. Delete Student")
        print("6. Sort Students")
        print("7. Exit")
        choice = input("Enter choice (1-7): ")
        if choice == '1':
            name = input("Enter name: ")
            age = int(input("Enter age: "))
            class_name = input("Enter class: ")
            subjects = input("Enter subjects (comma-separated): ").split(',')
            grades = [float(x) for x in input("Enter grades (comma-separated): ").split(',')]
            sms.add_student(name, age, class_name, subjects, grades)
        elif choice == '2':
            sms.view_all_students()
        elif choice == '3':
            search_term = input("Enter name to search: ")
            student = sms.search_student(search_term)
            if student:
                print(f"Name: {student.name}")
                print(f"Age: {student.age}")
                print(f"Class: {student.class_name}")
                print(f"Subjects: {', '.join(student.subjects)}")
                print(f"Grades: {', '.join(map(str, student.grades))}")
                print(f"Average Grade: {student.calculate_average_grade():.2f}")
            else:
                print("Student not found!")
        elif choice == '4':
            student_id = input("Enter student name to edit: ")
            name = input("Enter new name (press enter to skip): ") or None
            age_input = input("Enter new age (press enter to skip): ")
            age = int(age_input) if age_input else None
            class_name = input("Enter new class (press enter to skip): ") or None
            subjects_input = input("Enter new subjects (comma-separated, press enter to skip): ")
            subjects = subjects_input.split(',') if subjects_input else None
            grades_input = input("Enter new grades (comma-separated, press enter to skip): ")
            grades = [float(x) for x in grades_input.split(',')] if grades_input else None
            sms.edit_student(student_id, name, age, class_name, subjects, grades)
        elif choice == '5':
            student_id = input("Enter student name to delete: ")
            sms.delete_student(student_id)
        elif choice == '6':
            sort_by = input("Sort by (name/grade): ")
            sorted_students = sms.sort_students(sort_by)
            count = 1
            for student in sorted_students:
                print(f"position:{count} Name: {student.name}, Average Grade: {student.calculate_average_grade():.2f}")
                count+=1
        elif choice == '7':
            print("Exiting...")
            break
        else:
            print("Invalid choice!")


# Unit Tests to check if all functions in the class student managment system are working and bug free
class TestStudentManagementSystem(unittest.TestCase):
    def setUp(self):
        self.sms = StudentManagementSystem('test_student.json')

    def test_add_student(self):
        self.sms.add_student("John", 15, "10A", ["Math", "Science"], [85.0, 90.0])
        self.assertEqual(len(self.sms.students), 1)
        self.assertEqual(self.sms.students[0].name, "John")

    def test_calculate_average_grade(self):
        student = Student("John", 15, "10A", ["Math", "Science"], [85.0, 90.0])
        self.assertEqual(student.calculate_average_grade(), 87.5)

    def test_search_student(self):
        self.sms.add_student("John", 15, "10A", ["Math", "Science"], [85.0, 90.0])
        student = self.sms.search_student("John")
        self.assertIsNotNone(student)
        self.assertEqual(student.name, "John")

    def tearDown(self):
        if os.path.exists('test_student.json'):
            os.remove('test_student.json')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        unittest.main(argv=['first-arg-is-ignored'])
    else:
        run_cli()

run_cli()