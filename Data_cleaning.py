import pandas as pd

# read relevant rows of food data in chunks of 100000 rows at a time
# probably irrelevant: 'categories', 'categories_tags', 'labels', 'labels_tags'
food_data_chunk = pd.read_csv("D:\PyCharm\Techlabs_project\data\en.openfoodfacts.org.products.csv",
                              sep='\t', encoding="utf_8", chunksize=100000,
                              usecols=['code', 'url', 'product_name', 'ingredients_text', 'traces', 'traces_tags'])
food_data = pd.concat(food_data_chunk)


# inspect food_data
print(food_data.head())
print(food_data.info())
print(food_data.shape)

# try to find certain products in food_data
barcode = "0000000001670"
print(food_data.loc[food_data["code"] == barcode])
product_name = "Tofu Natur"
print(food_data.loc[food_data['product_name'] == product_name])

# drop duplicates
duplicateRows = food_data[food_data.duplicated()]
print(len(duplicateRows))
food_data = food_data.drop_duplicates()
print(food_data.shape)

# drop rows with missing values in ingredients_text
print(food_data.isnull().sum())
food_data = food_data.dropna(subset="ingredients_text")
print(food_data.isnull().sum())
print(food_data.shape)
