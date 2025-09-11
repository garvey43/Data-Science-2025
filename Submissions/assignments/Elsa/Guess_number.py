#Challenge 2: Guess the Number Game with Limited Attempts

#import Random module
import random

secret = random.randint(1, 50)

#set counter attempts 
attempts = 0
max_attemps = 5

#use while loops to allow up to 5 guesses
while attempts < max_attemps:
    guess = int(input(f"Attempts{attempts+1}/{max_attemps}: Your guess "))
    attempts +=1 #icreament attemps
    if guess==secret:   #branch
        print("🎉 Correct! You guessed the number in", attempts, "tries.")
        break
    elif guess < secret:
        print("Too low! Try again.")
    else:
        print("Too high! Try again.")
 #if the user didn’t guess correctly:       
print(" Out of attempts. The number was", secret)
        

                
