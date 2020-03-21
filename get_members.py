import requests

def get_isa(obj, num):
    words = []
    for i in range(num):
        if obj[i]['rel']['@id'] == "/r/IsA":
            w = (obj[i]['start']['term']).split('/')[-1]
            words.append(w)

    return words

if __name__ == "__main__":

    word = "vehicle"

    api = "http://api.conceptnet.io/c/en/" + word + "?rel=/r/IsA&limit=100" 
    obj = requests.get(api).json()

    json = obj["edges"]
    num = len(json)
   
    words = get_isa(json, num)

    print(list(set(words)))