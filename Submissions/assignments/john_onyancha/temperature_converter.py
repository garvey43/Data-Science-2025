#ask for temperature value fron the user in Celsius
# The code below converts a string input to a float
Celsius=float(input("Enter temperature in celsius:"))

# The code below converts Celsius to Fahrenheit
fahrenheit=Celsius*1.8+32

# The code below checks the temperature and prints a message based on the temperature range
if Celsius<=0:
    print("It's freezing cold!")
elif 0 < Celsius <= 20:
    print("it's a bit chilly")
elif 20 > Celsius <=30:
    print("Nice and warm")
else:
    print("its quite hot")

# The code below prints both the temperature in Celsius and Fahrenheit using a single print statement
print(f"The temperature in Celsius is {Celsius}°C and in Fahrenheit is {fahrenheit}°F")



