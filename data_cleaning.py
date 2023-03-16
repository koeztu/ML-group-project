import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#READ DATA AND SELECT FEATURES
data = pd.read_csv('./GroupProjectDataSet.csv', header=0, index_col=0)


#SELECT THE FEATURES WE WANT
#AND CONVERT DATA TO USEFUL FORMAT

#categorical data:
#use one-hot encoding
#what features we use is based on educated guesses about which features are predicitive of price and which ones are not
#why are years categorical? a property sold in 2006 will have had a higher price than a property sold in 2009 because of the housing crisis in 2008. likewise, a property sold in 2019 will have a higher price than one sold in 2010. in some years we have high prices, in some lower prices. and we cannot say that a higher year means a higher/lower price.

data['Alley'] = data['Alley'].fillna('NoAlleyAccess') #nan does not mean no value here
data['BsmtQual'] = data['BsmtQual'].fillna('NoBasement') #nan does not mean no value here

DUMMIES = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'Neighborhood', 'Condition1', 'RoofStyle', 'Exterior1st', 'Heating', 'CentralAir', 'PavedDrive', 'MoSold', 'SaleCondition', 'KitchenQual', 'OverallQual', 'ExterQual', 'ExterCond', 'BldgType', 'HouseStyle', 'BsmtQual', 'SaleType', 'SaleCondition']
dummies = pd.get_dummies(data[DUMMIES].astype(str))

print('\nnumber of nan dummies:', data[DUMMIES].isna().sum().sum())


#numerical data:

#add a composite feature that includes the age of the building when it was sold
data['AgeWhenSold'] = data['YrSold'] - data['YearBuilt']
#add a composite feature that includes the age of the building when it was sold, since last renovation
data['AgeSinceRemod'] = data['YrSold'] - data['YearRemodAdd']



#scale to numbers to be between 0 and 5
#the choice of 5 is more or less arbitrary
high = 5
data['GrLivArea'] = data['GrLivArea'] / data['GrLivArea'].max() * high
data['TotalBsmtSF'] = data['TotalBsmtSF'] / data['TotalBsmtSF'].max() * high
data['LotArea'] = data['LotArea'] / data['LotArea'].max() * high
data['AgeWhenSold'] = data['AgeWhenSold'] / data['AgeWhenSold'].max() * high
data['AgeSinceRemod'] = data['AgeSinceRemod'] / data['AgeSinceRemod'].max() * high


NUMERICALS =  ['LotArea', 'GrLivArea', 'BedroomAbvGr', 'TotalBsmtSF', 'KitchenAbvGr', 'GarageCars', 'AgeWhenSold', 'AgeSinceRemod']
numericals = data[NUMERICALS]

print('\nnumber of nan numericals:', data[NUMERICALS].isna().sum().sum())


#FUSE DATASETS
complete_dataset = pd.concat([numericals, dummies, data['Class']], axis=1)


#COUNT NAN
print('\nnumber of nan total:', complete_dataset.isna().sum().sum())
#conveniently there are no nans!


#DATA BALANCE
#FIXME: we are currently doing oversampling. but this creates a problem:
#the testing data will share entries with the training data because we have so many duplicate entries!
#hence our model will score highly
#we should only balance our training data, separate from testing data
print('')
class_counts = []
for i in range(5):
    count = np.sum(complete_dataset['Class'] == i)
    print(f'class {i}:', count)
    class_counts.append(count)
"""


dataframes = []
for i in range(5):
    df = complete_dataset[complete_dataset['Class'] == i]
    dataframes.append(df.sample(np.min(class_counts)))


complete_dataset = pd.concat(dataframes, axis=0)

print('')
for i in range(5):
    print(f'class {i}:', np.sum(complete_dataset['Class'] == i))
"""

#WRITE TO CSV FOR USE IN MODELS
complete_dataset.to_csv('./completeDataset.csv')