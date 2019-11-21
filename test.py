# Importing libraries
import pandas as pd
import seaborn as sns

# Reading the test dataset
df_test = pd.read_csv('test.csv')

# Preprocessing
df_test.drop(['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu'], axis=1, inplace=True)

df_test.isnull().sum()

df_test['LotFrontage'] = df_test['LotFrontage'].fillna(df_test['LotFrontage'].mean())
df_test['BsmtQual'] = df_test['BsmtQual'].fillna(df_test['BsmtQual'].mode()[0])
df_test['BsmtCond'] = df_test['BsmtCond'].fillna(df_test['BsmtCond'].mode()[0])
df_test['GarageType'] = df_test['GarageType'].fillna(df_test['GarageType'].mode()[0])

df_test.drop(['GarageYrBlt'], axis=1, inplace=True)

df_test['GarageFinish'] = df_test['GarageFinish'].fillna(df_test['GarageFinish'].mode()[0])
df_test['GarageQual'] = df_test['GarageQual'].fillna(df_test['GarageQual'].mode()[0])
df_test['GarageCond'] = df_test['GarageCond'].fillna(df_test['GarageCond'].mode()[0])

df_test['MasVnrType'].value_counts()
df_test['MasVnrType'] = df_test['MasVnrType'].fillna(df_test['MasVnrType'].mode()[0])
df_test['MasVnrArea'] = df_test['MasVnrArea'].fillna(0)

df_test['BsmtExposure'].value_counts()
df_test['BsmtExposure'].isnull().sum()
df_test['BsmtExposure'] = df_test['BsmtExposure'].fillna(df_test['BsmtExposure'].mode()[0])

df_test['BsmtFinType1'].value_counts()
df_test['BsmtFinType1'].isnull().sum()
df_test['BsmtFinType1'] = df_test['BsmtFinType1'].fillna(df_test['BsmtFinType1'].mode()[0])

df_test['BsmtFinType2'].value_counts()
df_test['BsmtFinType2'].isnull().sum()
df_test['BsmtFinType2'] = df_test['BsmtFinType2'].fillna(df_test['BsmtFinType2'].mode()[0])

sns.heatmap(df_test.isnull(), yticklabels=False, cbar=False)

df_test['BsmtFullBath'].value_counts()
df_test['BsmtFullBath'].isnull().sum()
df_test['BsmtFullBath'] = df_test['BsmtFullBath'].fillna(df_test['BsmtFullBath'].mode()[0])

df_test['BsmtHalfBath'].value_counts()
df_test['BsmtHalfBath'].isnull().sum()
df_test['BsmtHalfBath'] = df_test['BsmtHalfBath'].fillna(df_test['BsmtHalfBath'].mode()[0])

df_test.columns[df_test.isnull().any()]

df_test['MSZoning'].value_counts()
df_test['MSZoning'] = df_test['MSZoning'].fillna(df_test['MSZoning'].mode()[0])

df_test['Utilities'].value_counts()
df_test['Utilities'] = df_test['Utilities'].fillna(df_test['Utilities'].mode()[0])

df_test['Exterior1st'].value_counts()
df_test['Exterior1st'] = df_test['Exterior1st'].fillna(df_test['Exterior1st'].mode()[0])

df_test['Exterior2nd'].value_counts()
df_test['Exterior2nd'] = df_test['Exterior2nd'].fillna(df_test['Exterior2nd'].mode()[0])

df_test['KitchenQual'].value_counts()
df_test['KitchenQual'] = df_test['KitchenQual'].fillna(df_test['KitchenQual'].mode()[0])

df_test['Functional'].value_counts()
df_test['Functional'] = df_test['Functional'].fillna(df_test['Functional'].mode()[0])

df_test['GarageCars'].value_counts()
df_test['GarageCars'] = df_test['GarageCars'].fillna(df_test['GarageCars'].mode()[0])

df_test['GarageArea'].value_counts()
df_test['GarageArea'] = df_test['GarageArea'].fillna(df_test['GarageArea'].mean())

df_test['SaleType'].value_counts()
df_test['SaleType'] = df_test['SaleType'].fillna(df_test['SaleType'].mode()[0])

df_test['BsmtFinSF1'].value_counts()
df_test['BsmtFinSF1'] = df_test['BsmtFinSF1'].fillna(df_test['BsmtFinSF1'].mean())

df_test['BsmtFinSF2'].value_counts()
df_test['BsmtFinSF2'] = df_test['BsmtFinSF2'].fillna(df_test['BsmtFinSF2'].mean())

df_test['BsmtUnfSF'].value_counts()
df_test['BsmtUnfSF'] = df_test['BsmtUnfSF'].fillna(df_test['BsmtUnfSF'].mean())

df_test['TotalBsmtSF'].value_counts()
df_test['TotalBsmtSF'] = df_test['TotalBsmtSF'].fillna(df_test['TotalBsmtSF'].mean())

# Saving data to csv file
df_test.to_csv('preprocessed_test.csv',index=False)