#import
! pip install openfoodfacts #only do once

import openfoodfacts # https://openfoodfacts.github.io/openfoodfacts-python/Usage/
# https://world.openfoodfacts.org/data / open food facts as our source of choice
import pandas as pd
import time # for a scraping delay


'''
A df is created based on all German products available via the Open Food Facts package.
Only those products are included in the df that also have all the necessary columns (otherwise key error).
The df would normally be saved as food_data.csv at the bottom.
This was, however, not done here because this code snippet was originally written in and later copied from Colab.
At that point in time, the file food_data.csv had already been added manually to the GitHub repo. 
'''



df = pd.DataFrame(columns=["Barcode", "Product Name", "Ingredients", "Analysis Tags"])
i=1
for product in openfoodfacts.products.get_all_by_language('german'):
    try:
        #indiviual values of new row / product
        id = product['_id'] #get Barcode
        ingredient_de = product['ingredients_text_de'] # get Zutaten
        product_name = product['product_name_de']
        analysis_tags = ",".join(product['ingredients_analysis_tags'])

        data = {"Barcode": id, "Product Name": product_name, "Ingredients": ingredient_de, "Analysis Tags": analysis_tags}
        df1 = pd.DataFrame(data, index = [i])
        df = pd.concat([df, df1])
        time.sleep(0.02)
        i +=1
    except KeyError:
        i +=1
        continue

# this is how we saved the df in colab
#from google.colab import drive
#drive.mount('/content/drive')
#path = '/content/drive/My Drive/food_data.csv' #change the filename (output.csv) next time myb
#with open(path, 'w', encoding = 'utf-8-sig') as f:
#  df.to_csv(f)
