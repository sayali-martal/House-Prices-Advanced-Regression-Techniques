# Importing libraries
import pandas as pd
import seaborn as sns

# Reading dataset
df = pd.read_csv('train.csv')

# Heatmap
sns.heatmap(df.isnull(), yticklabels=False, cbar=False)

# Preprocessing
df.isnull().sum()

df.drop(['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu'], axis=1, inplace=True)

df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())

df['BsmtQual'].value_counts()
df['BsmtQual'] = df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])

df['BsmtCond'].value_counts()
df['BsmtCond'] = df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])

df['GarageType'].value_counts()
df['GarageType'] = df['GarageType'].fillna(df['GarageType'].mode()[0])

df.drop(['GarageYrBlt'], axis=1, inplace=True)

df['GarageFinish'].value_counts()
df['GarageFinish'] = df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])

df['GarageQual'].value_counts()
df['GarageQual'] = df['GarageQual'].fillna(df['GarageQual'].mode()[0])

df['GarageCond'].value_counts()
df['GarageCond'] = df['GarageCond'].fillna(df['GarageCond'].mode()[0])

df['MasVnrType'].value_counts()
df['MasVnrType'] = df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])

df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

df['BsmtExposure'].value_counts()
df['BsmtExposure'].isnull().sum()
df['BsmtExposure'] = df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])

df['BsmtFinType1'].value_counts()
df['BsmtFinType1'].isnull().sum()
df['BsmtFinType1'] = df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])

df['BsmtFinType2'].value_counts()
df['BsmtFinType2'].isnull().sum()
df['BsmtFinType2'] = df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])

df.columns[df.isnull().any()]

df['Electrical'].value_counts()
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

# Read final est csv
df_test = pd.read_csv('preprocessed_test.csv')

# Concatinating test and train dataset
df_new = pd.concat([df, df_test], axis=0)

df_new = pd.get_dummies(df_new, prefix_sep='_', drop_first=True)

# Differentiating test and train data after preprocessing
df_Train = df_new.iloc[:1460, :]
df_Test = df_new.iloc[1460:, :]
df_Test.drop(['SalePrice'],axis=1,inplace=True)
X_train = df_Train.drop(['SalePrice'],axis=1)
y_train = df_Train['SalePrice']
 
# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 23)
X_train = lda.fit_transform(X_train, y_train)
explained_variance = lda.explained_variance_ratio_
df_Test = lda.transform(df_Test)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10)
regressor.fit(X_train, y_train)

# Saving model
import pickle
filename = 'model.pkl'
pickle.dump(regressor, open(filename, 'wb'))

# Predicting y values for test dataset
y_pred = regressor.predict(df_Test)

# Saving results into file
prediction = pd.DataFrame(y_pred)
df_final = pd.read_csv('sample_submission.csv')
datasets = pd.concat([df_final['Id'], prediction], axis=1)
datasets.columns = ['Id', 'SalePrice']
datasets.to_csv('sample_submission.csv', index=False)