
# Vegan Scan - vegan shopping made simple  

## Project Summary  
Over the last years there has been a pronounced increase in the number of vegans in Germany. Especially in the early stages of becoming vegan, it is necessary to learn to discriminate between vegan and non-vegan products. Labels, seemingly endless lists of ingredients or barcode scanners, that’s how it shall work—but, then again, some ingredients are not listed or the databases lack information. 
That’s the setting for our project: Based on the ingredient compositions obtained from Open Food Facts, a binary classification of the vegan statuses of products is carried out. To do so, we compare the effectiveness of a PyTorch deep learning approach with a simple random forest. Eventually, both reach an accuracy of circa 97%.

## Files / Process  

### data

1. df_generation.py - retrieve data from Open Food Facts --> result: food_data.csv  
  
### data_preprocessing   
2. data_cleaning.py - clean food_data.csv --> result: food_data_test.csv (incl. unknown vegan status), food_data_clean_test.csv (excl. unknown vegan status)  
3. tokenization.py -   word tokenization --> binarization --> cleaning steps (with use of txt_to_lst) --> food_matrix.csv (saved to [Google Drive](https://drive.google.com/file/d/1Hs2WyynXT-CUwZ9_F2dliwntaBNEjJ0D/view?usp=share_link); 500 MB)  
4. items.txt, items_copy.txt, items2_copy.txt - files (backup) with column names --> items2_copy.txt was cleaned manually (in the end it consisted only of noise words) 
5. txt_to_lst.py - (the noise words from items2_copy.txt are turned into a list that can be exported to tokenization.py for cleaning step (cleaning of columns)  

### iris_examples  
6. model.py - pytorch tabular example with iris data (iris.csv) (high-level)  
7. bin_class.py & bin_class_eval.py - another pytorch tabular example with iris data (low-level)  
8. iris_random_forest.py - random forest example with iris data  

### models  
9. vegan_bin_class.py - train a pytorch tabular model 80% of food_matrix.csv to classify products as vegan/non-vegan based on the ingredients 
10. vegan_bin_class_continue_training.py - continue training the model  
11. vegan_bin_class_eval.py - evaluate the model  
12. vegan_random_forest.py - train a random forest on 80% of food_matrix.csv to classify products as vegan/non-vegan based on the ingredients  
  
### Features to implement in the future  
13. Adapt learning rate to get an even higher accuracy.  
14. Build an app that can scan barcodes, get the ingredients and deploy the model to classify the product as vegan/non-vegan.  
15. Implement other languages.  

## Credits  
Contributors: Stefan Smid (@StefanFSmid), Wanja Tolksdorf (@Wanja24)  
Mentor: Jennifer Matthiesen (@jjmatthiesen)     
This project was done as part of the Deep Learning track at TechLabs Hamburg (https://techlabs.org/location/hamburg).    
Data source: https://openfoodfacts.org (licensed under the Open Database License)  
Examples used as templates:   
https://www.kaggle.com/code/devishu14/tabulardata-pytorch-tabular#%F0%9F%8C%B9-iris-flower-dataset  
https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89  
https://www.kaggle.com/code/tcvieira/simple-random-forest-iris-dataset/notebook 
