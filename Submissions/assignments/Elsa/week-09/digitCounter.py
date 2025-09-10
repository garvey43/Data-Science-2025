# Challenge 1: Digit Frequency Counter

#  Prompt the user
num_str = input("Enter a positive integer: ")

# Validate input
if not num_str.isdigit():
    print("Invalid input. Please enter only digits (0-9).")
else:
    #  Initialize dictionary
    counts = {}

    # Iterate over each character in the string
    for ch in num_str:
        digit = int(ch) 
        counts[ch] = counts.get(ch, 0) + 1

    #print each digit and its frequency in ascending order:
    print("\nDigit Frequency:")
    for digit in sorted(counts):
        print(f"Digit {digit}: {counts[digit]} time(s)")
