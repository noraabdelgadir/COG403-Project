import requests

def get_isa(obj, num):
    words = []
    for i in range(num):
        if obj[i]['rel']['@id'] == "/r/IsA":
            w = (obj[i]['start']['term']).split('/')[-1]
            words.append(w)

    return words

# cross reference words with MUSE
muse = []
f = open("en-en.txt", "r")
for line in f.readlines():
    muse.append(line.split('\t')[0])

def muse_check(words):
    checked = []
    for word in words:
        if word in muse:
            checked.append(word)
    return checked

def get_members(category):
    api = "http://api.conceptnet.io/c/en/" + category + "?rel=/r/IsA&limit=1000" 
    obj = requests.get(api).json()

    json = obj["edges"]
    num = len(json)
   
    words = get_isa(json, num)

    return muse_check(list(set(words)))