#Task : miniadventure stats
#define two variables: steps and health.
#loop steps times:
#  prompt user: did you take a step? (yes/no)
#  if yes, increament a count; if no decrement health by 1.
#  stop early in health <= 0 (use break).
#Report :
# total steps taken
# final health


# Define variables
steps = 5
health = 3

# Initialize step counter
steps_taken = 0

# Loop for steps
for i in range(steps):
    response = input(f"Step {i + 1} - Did you take a step? (yes/no): ").lower()
    if response == 'yes':
        steps_taken += 1
    elif response == 'no':
        health -= 1
        if health <= 0:
            print("You've run out of health!")
            break
    else:
        print("Invalid input. Please answer 'yes' or 'no'.")
        continue

# Report
print("\n--- Adventure Summary ---")
print(f"Total steps taken: {steps_taken}")
print(f"Final health: {health}")

