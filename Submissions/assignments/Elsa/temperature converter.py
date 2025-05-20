celsius=float(input("Enter temperature in Celsius: "))

fahrenheit= (celsius *9/5)+32

print("Temperature in fahrenheight: ",fahrenheit)

if celsius <0:
    print("Its freezing cold!")
elif  0 <= celsius <20 :
    print("Its abit chilly.")
elif 20 <= celsius <30 :
    print("Nice and warm!")
else:
    print("Its quite hot!")
    
print("temperature in Celcius: ",celsius)

