# create a translation dictionary for the lang
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

en_fr = translate('fr') # french dictionary
en_ar = translate('ar') # arabic dictionary

# get translations of a list words in a language
def get_translations(lang, words):
    translations = en_fr if lang == 'fr' else en_ar
    translated = {}
    for word in words:
        if word in translations:
            translated[word] = translations[word]
    return translated
