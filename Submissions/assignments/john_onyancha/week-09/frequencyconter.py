#ask for user input
num_str=input("Enter a positive integer:")
# validate the users input
if num_str.isdigit():

    #initialize a dictionary
    counts={}

    #iterate over each character in the string
    for ch in num_str:
        #populate the dictionary with frequency of values inside the string
        counts[ch]=counts.get(ch,0)+1

    #iterate through the dictionary to display the frequency for each value
    for dg in sorted(counts):
        print(f"Digits{dg}:{counts[dg]}")
