import re

#removal of unecessary spaces, quotation marks and commas in an indivdual string (ingredient)
def remove_symbols(item):
    new_str = item.strip()
    new_str = re.sub('"', '', new_str)
    new_str = re.sub(',', '', new_str)
    return new_str

# opening the file in read mode
ingredients_txt = open("items2_copy.txt", "r", encoding="utf-8")

# reading the file
ingredients_data = ingredients_txt.read()

ingredients_txt.close() #Close the file when we’re done!'''


# replacing end splitting the text
# when newline ('\n') is seen.
ingredients_lst = [remove_symbols(item) for item in ingredients_data.split('\n')]
ingredients_lst[0] = ','
ingredients_lst[ingredients_lst.index('di-triphosphate')] = 'di-,triphosphate'

print(ingredients_lst)




