def solve(items,ruleKey,ruleValue):
    #mapping ruleKey to in dex
    key_index={"type":0,"color":1,"name":2}
    index=key_index[ruleKey]

    #initialize value for count
    count=0

    #iterate through the items array
    for item in items:

        #check if items index == rulValue
        if item[index] == ruleValue:

            #increment count
            count+=1
    return count

#give initial values for the items1 array
items1=[["phone","silver","pixel"],["computer","silver","lenovo"],["phone","silver","iphone"]]

#initialize values for both ruleKey and value
rulekey="color"
rulevalue="silver"

#passby reference the items1 array together with rule key and Value
print(solve(items1,rulekey,rulevalue))


