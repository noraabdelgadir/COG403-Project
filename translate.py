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

en_ar = translate('ar')
en_fr = translate('fr')

# muse = {}
# f = open("translations/en-fr.txt", "r")
# for line in f.readlines():
#     eng = line.split(' ')[0]
#     fr = line.split(' ')[1][:-1]
#     if(eng not in muse.keys()):
#         muse[eng] = [fr]
#     else:
#         muse[eng].append(fr)

def get_translations(lang, words):
    translations = en_ar if lang == 'ar' else en_fr
    translated = {}
    for word in words:
        if word in translations:
            translated[word] = translations[word]
    return translated

# t = []
# for word in ['cucumber', 'legume', 'truffle', 'radish', 'squash', 'greens', 'cauliflower', 'spinach', 'lettuce', 'bean', 'beet', 'fennel', 'gumbo', 'asparagus', 'corn', 'mushroom', 'vegetable', 'plantain', 'eggplant', 'celery', 'pumpkin', 'artichoke']:
#     if get_translation(word):
#         t.extend(get_translation(word))

# print(t)


