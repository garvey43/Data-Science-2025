# Import the built in random module.
# Generate a random integer between 1 and 50:
import random
secret=random.randint(1,5)

#Set a counter attempts = 0 and a maximum max_attempts = 5.
attempts=0
max_attempts=5

#implement a while loop to allow up to 5 guesses
#prompt the user for an attempt
while attempts < max_attempts:
    guess=int(input(f"Attempt {attempts+1}/{max_attempts}:Your guess?"))

    # Increment attempts += 1.
    attempts+=1

    #use branching to check if guess is less than,greater than or equall to and print some statements
    if guess == secret:
        print(f"Correct! You guessed the number in {attempts} tries")
        break
    elif guess < secret:
        print("Too low! Try again")
    else:
        print("Too high! Try again.")

#if the max_number of attempts is reached print the secret number
else:
    print(f"out of attempts. the number was: {secret}")


