# Side Project: Cookpad Food Classification

## Problem Description
This project aims to build a deep learning model to accurately classify 55 different food with various sizes. For more information, please refer to this [site](https://deepanalytics.jp/compe/59)

## Data 
1. Number of training set(.jpg): 11,995
2. Number of test set(.jpg): 3,937
3. Number of classes: 55

## Methology 
1. Transform the data with various sizes to (192, 256, 3)
2. Use VGG19 pretrained for feature extraction
3. Implement Data augmentation

## Result

<img src="https://github.com/lwkuant/Side_project_Cookpad_Food_Classification/blob/master/Accuracy_VGG19_1_layer_256.png">
<img src="https://github.com/lwkuant/Side_project_Cookpad_Food_Classification/blob/master/Loss_VGG19_1_layer_256.png">