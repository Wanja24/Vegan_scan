# - 1 - import: used packages are being installed
import pandas as pd
#! pip install -U spacy
import spacy
import spacy.cli # the client is necessary in order to download the german package
#spacy.cli.download("de_core_news_sm") # spacy's german model for word tokenization; only once
from sklearn.preprocessing import MultiLabelBinarizer


# - 2 - loading dataset: food_data_clean_test.csv is loaded
clean_df = pd.read_csv('../data/food_data_clean_test.csv').copy()


# - 3 - word tokenization:  using spacy's german model: https://spacy.io/models/de
nlp = spacy.load('de_core_news_sm')
clean_df['Ingredients'] = clean_df['Ingredients'].apply(lambda row: [token.text for token in nlp(row)])


# - 4 - binarization: (1,0) with multiple labels (Schweinefett, Nitritpökelsalz, etc.) per product possible
count_vec = MultiLabelBinarizer()
mlb = count_vec.fit(clean_df['Ingredients'])
zutaten = mlb.classes_.tolist()
matrix = pd.DataFrame(mlb.transform(clean_df['Ingredients']), columns=zutaten)


# - 5 - column cleaning 1: drops ingredients that appear only in 1 or 2 products
matrix = matrix.loc[:, matrix.sum()>2]


# - 6 - column cleaning 2: affects ingredients that are none, e.g., "kuehischrank" (stopwords)
from txt_to_lst import ingredients_lst
matrix = matrix.drop(columns=ingredients_lst)


# - 7 - dependent variable: binarized vegan status is being added here
matrix = matrix.join(clean_df['Is_vegan'])


# - 8 - row cleaning: now, there are rows without any label (-> deleted in the step above), these rows are dropped here:
matrix = matrix.drop(matrix[(matrix[matrix.columns.difference(['Is_vegan'])] == 0).all(axis=1)].index)


# - 9 - saving result: save new and completely adjusted matrix to file and upload it
matrix.to_csv("food_matrix.csv", encoding='utf-8', index=False)




# A P P E N D I X

'''
The code here was needed to create the ingredients files items.txt, items2_copy.txt, and items_copy.txt
'''

#save list of all column names (ingredients) to txt
#list_of_tuples = multimat.columns.tolist()
#column_headers = [' '.join(ingredient) for ingredient in list_of_tuples] #mit dieser list comprehensionn einfach die spaltennmen des df umbennen

'''filename = "items2.txt"
#w tells python we are opening the file to write into it
file = open(filename, 'w', encoding="utf-8")

for ingredient in column_headers:
  file.write('"' + ingredient + '", ' + '\n')

file.close() #Close the file when we’re done!'''
