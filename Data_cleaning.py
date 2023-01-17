import pandas as pd

# load food_data
food_data = pd.read_csv("D:/PyCharm/Techlabs_project/food_data.csv", encoding="utf_8")

# inspect food_data
print(food_data.head())
print(food_data.info())
print(food_data.shape)

# rename the first column
food_data = food_data.rename(columns={'Unnamed: 0': 'Old Index'})

# check for duplicate rows
duplicateRows = food_data[food_data.duplicated()]
print(len(duplicateRows))  # there are no duplicates

# check for missing values
print(food_data.isnull().sum())  # 3142 na in Product Name, 2878 in Ingredients
null = food_data[food_data['Ingredients'].isnull()]
print(null)

# drop rows with missing values in Ingredients
food_data = food_data.dropna(subset="Ingredients")
print(food_data.isnull().sum())
print(food_data.shape)

# clean Ingredients column

# replace underscores
food_data["Ingredients"] = food_data["Ingredients"].apply(lambda x: x.replace("_", ""))

# change uppercase?
#food_data["Ingredients"] = food_data["Ingredients"].apply(lambda x: x.lower())  # oder .title()

# delete percentages?

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

# inspect Vegan column

# check for missing values
print(food_data["Vegan"].isnull().sum())  # 43 missing values
null_vegan = food_data[food_data['Vegan'].isnull()]
# reason for missing values: no information about vegan in analysis tags (only vegetarian)

# check for values with unknown vegan status/maybe vegan
vegan_unknown = food_data[(food_data["Vegan"]=="vegan status unknown")|(food_data["Vegan"]=="maybe vegan")]
print(len(vegan_unknown))  # 19399 times the vegan status is not known

"""
interesting finding: 127 Veganer Burger: 48,5% texturiertes Erbsenprotein (Erbsenprotein, Erbsenmehl), Maisgranulat,
Reisgranulat,Verdickungsmittel: Methylcellulose, Sonnenblumenkerne teilent - ölt, Rapsöl, Gewürze (Knoblauch, Pfeffer 
und Zwiebeln), Aromen, Karamellpulver (Karamellzuckersirup, Maltodextrin), 2,9 % enzymatisch hydrolysiertes 
Erbsenprotein, Speisesalz, Rote Beete, Zucker, Raucharoma.
"""

