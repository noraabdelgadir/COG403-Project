def translate(lang):
    f = open("translations/en-" + lang + ".txt", "r")
    splitter = ' ' if lang == 'fr' else '\t'
    translations = {}
    for line in f.readlines():
        en = line.split(splitter)[0]
        tr = line.split(splitter)[1][:-1]
        
        if(en not in translations.keys()):
            translations[en] = [tr]
        else:
            translations[en].append(tr)
    
    return translations

def get_translations(lang, words):
    translations = translate(lang)
    translated = {}
    for word in words:
        if word in translations:
            translated[word] = translations[word]
    return translated
