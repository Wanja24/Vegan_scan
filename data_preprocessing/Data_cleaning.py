import pandas as pd
import re
#! pip install langdetect
from langdetect import detect

# load food_data
food_data = pd.read_csv("D:/PyCharm/Techlabs_project/food_data.csv", encoding="utf_8")
# test code with part of the data
# food_data = food_data[0:5000]

# inspect food_data
print(food_data.head())
print(food_data.info())
print(food_data.shape)

# drop the first column
food_data = food_data.drop('Unnamed: 0', axis=1)

# check for duplicate rows
duplicateRows = food_data[food_data.duplicated()]
print(len(duplicateRows))  # there are 7 duplicates

# drop duplicates
food_data = food_data.drop_duplicates()
print(food_data.shape)

# check for missing values in Ingredients
print(food_data.isnull().sum())  # 3142 na in Product Name, 2878 in Ingredients
null_ing = food_data[food_data['Ingredients'].isnull()]

# drop rows with missing values in Ingredients
food_data = food_data.dropna(subset="Ingredients")
print(food_data.isnull().sum())
print(food_data.shape)

# clean Ingredients column
def clean_ingredients(x):
    # turn into lowercase
    x = x.lower()
    # replace underscores
    x = x.replace("_", "")
    # remove percentages and grams (incl. numbers & brackets)
    #x = re.sub(r"\(?\s*\d*[\.\,]?\d+\s*(kg|g|%)\s*\)?", "", x)
    # remove all numbers incl. percentages and (kilo)gram except for e300 and e 300
    x = re.sub(r"(?<!\We\s|.\We|..\d)\d*[\.\,]?\d+\s*(kg|g|%)?", "", x)
    # remove special characters except for comma, hyphen, whitespace, numbers and words
    x = re.sub(r"[^\w\-,\s]", "", x)
    # remove excess whitespace
    x = re.sub(r"\s+", ' ', x).strip()
    return x

food_data["Ingredients"] = food_data["Ingredients"].apply(clean_ingredients)

# check for new missing values in Ingredients
print(food_data["Ingredients"].isnull().sum())
null_ing = food_data[food_data['Ingredients'].isnull()]  # no nan
empty_ing = food_data.loc[food_data["Ingredients"]==""]  # 8 empty rows

# drop rows with new missing values in Ingredients
food_data = food_data.dropna(subset="Ingredients")
food_data = food_data.loc[food_data["Ingredients"]!=""]
print(food_data.isnull().sum())
print(food_data.shape)

"""
# replace non-German ingredients text with None
def detect_lang(x):
    #try:
        lang = detect(x)
        if lang == "de":
            return x
    #except:
        #print("This row throws error:", x)
        #return x

food_data["Ingredients"] = food_data["Ingredients"].apply(detect_lang)
 """

# inspect Analysis Tags column
print(food_data["Analysis Tags"].unique())
# different types of vegan status: en:vegan, en:non-vegan, en:vegan-status-unknown, en:maybe-vegan

# create new column Vegan
def create_vegan(txt):
    lst = txt.split(",")
    for i in lst:
        if i == "en:vegan":
            return "vegan"
        elif i == "en:non-vegan":
            return "non-vegan"
        elif i == "en:vegan-status-unknown":
            return "vegan status unknown"
        elif i == "en:maybe-vegan":
            return "maybe vegan"

food_data["Vegan"] = food_data["Analysis Tags"].apply(create_vegan)

# check for missing values in Vegan
print(food_data["Vegan"].isnull().sum())  # 43 missing values
null_vegan = food_data[food_data['Vegan'].isnull()]
# reason for missing values: no information about vegan in analysis tags (only vegetarian)

# drop rows with missing values in Vegan
food_data = food_data.dropna(subset="Vegan")
print(food_data.isnull().sum())
print(food_data.shape)

# check how many products are vegan, non-vegan or unknown
print(food_data.groupby(by="Vegan")["Vegan"].count())  # maybe vegan: 1607, non-vegan: 29742, vegan: 18567, vegan status unknown: 17782
vegan_unknown = food_data[(food_data["Vegan"]=="vegan status unknown")|(food_data["Vegan"]=="maybe vegan")]
print(len(vegan_unknown))  # 19389 times the vegan status is not known

# create Is_vegan column
def create_Is_vegan(txt):
    lst = txt.split(",")
    for i in lst:
        if i == "en:vegan":
            return 1
        elif i == "en:non-vegan":
            return 0

food_data["Is_vegan"] = food_data["Analysis Tags"].apply(create_Is_vegan)

# check for missing values in Is_vegan
print(food_data["Is_vegan"].isnull().sum())  # 19389 missing values
null_vegan = food_data[food_data['Is_vegan'].isnull()]
# reason for missing values: unknown vegan status

# create a copy of the dataset and drop rows with missing values in Is_vegan
food_data_clean = food_data.copy()
food_data_clean = food_data_clean.dropna(subset="Is_vegan")
print(food_data_clean.isnull().sum())
print(food_data_clean.shape)

# save the two dataframes as csv files
food_data.to_csv("food_data_test.csv", encoding='utf-8', index=False)
food_data_clean.to_csv("food_data_clean_test.csv", encoding='utf-8', index=False)

"""
interesting findings: 

127 Veganer Burger: 48,5% texturiertes Erbsenprotein (Erbsenprotein, Erbsenmehl), Maisgranulat,
Reisgranulat,Verdickungsmittel: Methylcellulose, Sonnenblumenkerne teilent - ölt, Rapsöl, Gewürze (Knoblauch, Pfeffer 
und Zwiebeln), Aromen, Karamellpulver (Karamellzuckersirup, Maltodextrin), 2,9 % enzymatisch hydrolysiertes 
Erbsenprotein, Speisesalz, Rote Beete, Zucker, Raucharoma.

"Sucre de fleur de coco, Kokosblütenzucker"

20270278 italienisch: "it: 75% albicocche, it: zucchero,it: succo di limone concentrato, gelificante:pectina."

"ZU e 500 e390 TEN Hart veizengrieß, 10% Frischei. Das Produkt kann Spuren von Soja, Senf und Lupinen enthalten. DURCHSCHNITTLIGHE NAHRWERT pro % RM 000 pro100g Brennwert kJ/kcal Fett 1547/365 18% 2,69 4% davon: - gesättigte Fettsäuren Kohlenhydrate davon: - Zucker Eiweiß Salz 3% 0,6g 70,0g 27% 4% 3,3g 14,0g 0,03g ferenzmenge für einen durchschnittichen 28 % &lt;1% wachsenen (8400kJ/2000kcal)"
"""
