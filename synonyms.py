liste_synonyms = [{'Milch': 'magermic'}]

def fill_synonyms():
    key = input('Name main component:')
    value = input('Name synonyms:')
    liste_synonyms.append({key: value})
    return liste_synonyms

fill_synonyms()
print(liste_synonyms)