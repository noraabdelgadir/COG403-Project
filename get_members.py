import requests

def get_isa(word):
    api = "http://api.conceptnet.io/c/en/" + word + "?rel=/r/IsA&limit=100" 
    obj = requests.get(api).json()

    json = obj["edges"]
    num = len(json)

    words = []
    for i in range(num):
        if json[i]['rel']['@id'] == "/r/IsA":
            w = (json[i]['start']['term']).split('/')[-1]
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

if __name__ == "__main__":

    word = "vehicle"
   
    words = get_isa(word)

    print(muse_check(list(set(words))))