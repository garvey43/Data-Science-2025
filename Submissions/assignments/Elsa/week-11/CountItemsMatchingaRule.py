def countMatches(items, ruleKey, ruleValue):
    # Map ruleKey to index
    key_index = {"type": 0, "color": 1, "name": 2}
    index = key_index[ruleKey]

    count = 0
    for item in items:
        if item[index] == ruleValue:
            count += 1

    return count

items = [["phone","blue","pixel"],
         ["computer","silver","lenovo"],
         ["phone","gold","iphone"]]

ruleKey = "type"
ruleValue = "phone"

print(countMatches(items, ruleKey, ruleValue))
