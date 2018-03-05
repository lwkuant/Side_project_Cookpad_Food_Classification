# Side Project: Cookpad Food Classification

## Problem Description
This project aims to build a deep learning model to accurately classify 55 different food with various sizes. For more information, please refer to this [site](https://deepanalytics.jp/compe/59)

## Data 
1. Number of training set(.jpg): 11,995 (Training: 10,000 & Validation: 1,995)
2. Number of test set(.jpg): 3,937
3. Number of classes: 55

## Methology 
1. Transform the data with various sizes to (192, 256, 3)
2. Use pretrained model for feature extraction:
	(1) VGG19
	(2) InceptionV3
3. Implement Data augmentation
4. Retrain part of the layers of the VGG19 pretrained model

## Result

### VGG19 Pretrained Model + 1 Layer Dense Model (256, ) 
<img src="https://github.com/lwkuant/Side_project_Cookpad_Food_Classification/blob/master/Accuracy_VGG19_1_layer_256.png">
<img src="https://github.com/lwkuant/Side_project_Cookpad_Food_Classification/blob/master/Loss_VGG19_1_layer_256.png">

### VGG19 Pretrained Model + 1 Layer Dense Model (256, ) + Data Augmentation to 40,000 pictures
<img src="https://github.com/lwkuant/Side_project_Cookpad_Food_Classification/blob/master/Accuracy_VGG19_Data_Augmentation_40k_1_layer_256.png">
<img src="https://github.com/lwkuant/Side_project_Cookpad_Food_Classification/blob/master/Loss_VGG19_Data_Augmentation_40k_1_layer_256.png">

### VGG19 Pretrained Model + Retrain Block 5 of VGG19 + 1 Layer Dense Model (256, )
<img src="https://github.com/lwkuant/Side_project_Cookpad_Food_Classification/blob/master/Accuracy_VGG19_Retraining_Block5_1_layer_256.png">
<img src="https://github.com/lwkuant/Side_project_Cookpad_Food_Classification/blob/master/Loss_VGG19_Retraining_Block5_1_layer_256.png">

### InceptionV3 Pretrained Model + 1 Layer Dense Model (256, ) 
<img src="https://github.com/lwkuant/Side_project_Cookpad_Food_Classification/blob/master/Accuracy_InceptionV3_1_layer_256.png">
<img src="https://github.com/lwkuant/Side_project_Cookpad_Food_Classification/blob/master/Loss_InceptionV3_1_layer_256.png">