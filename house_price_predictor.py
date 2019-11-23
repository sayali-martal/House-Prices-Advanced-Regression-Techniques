# Importing libraries
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from vecstack import stacking
from sklearn.model_selection import cross_val_score
import pickle

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
lda = LDA(n_components = 23)
X_train = lda.fit_transform(X_train, y_train)
explained_variance = lda.explained_variance_ratio_
df_Test = lda.transform(df_Test)

## Fitting Ridge Regression to the dataset
#tunning_Ridge = Ridge(alpha=5, solver='saga', tol=0.09, max_iter=100)
#tunning_Ridge.fit(X_train, y_train)
#
## Applying Grid Search to find the best model and the best parameters
#parameters = [{'max_iter' : [10, 100, 1000]}]
#grid_search = GridSearchCV(estimator = tunning_Ridge,
#                           param_grid = parameters,
#                           scoring = 'explained_variance',
#                           cv = 10,
#                           n_jobs = -1)
#grid_search = grid_search.fit(X_train, y_train)
#best_accuracy = grid_search.best_score_
#best_parameters = grid_search.best_params_
#
## Fitting LASSO Regression to the dataset
#tunning_Lasso = Lasso(max_iter=10, normalize=True, precompute=True, tol=0.001, warm_start=True)
#tunning_Lasso.fit(X_train, y_train)
#
## Applying Grid Search to find the best model and the best parameters
#parameters = [{'warm_start' : [True, False]}]
#grid_search = GridSearchCV(estimator = tunning_Lasso,
#                           param_grid = parameters,
#                           scoring = 'explained_variance',
#                           cv = 10,
#                           n_jobs = -1)
#grid_search = grid_search.fit(X_train, y_train)
#best_accuracy = grid_search.best_score_
#best_parameters = grid_search.best_params_
#
## Fitting XGBoost to the Training set
#tunning_XGB = XGBRegressor(base_score=0.43)
#tunning_XGB.fit(X_train, y_train)
#
## Applying Grid Search to find the best model and the best parameters
#parameters = [{'base_score': [0.42, 0.43, 0.44]}]
#grid_search = GridSearchCV(estimator = tunning_XGB,
#                           param_grid = parameters,
#                           scoring = 'explained_variance',
#                           cv = 10,
#                           n_jobs = -1)
#grid_search = grid_search.fit(X_train, y_train)
#best_accuracy = grid_search.best_score_
#best_parameters = grid_search.best_params_
#
## Fitting Random Forest Regression to the dataset
#tunning_RF = RandomForestRegressor(n_estimators = 400, warm_start=True)
#tunning_RF.fit(X_train, y_train)
#
## Applying Grid Search to find the best model and the best parameters
#parameters = [{'warm_start': [True, False]}]
#grid_search = GridSearchCV(estimator = tunning_RF,
#                           param_grid = parameters,
#                           scoring = 'explained_variance',
#                           cv = 10,
#                           n_jobs = -1)
#grid_search = grid_search.fit(X_train, y_train)
#best_accuracy = grid_search.best_score_
#best_parameters = grid_search.best_params_

# Stack predicted values to find optimal values
models = [RandomForestRegressor(n_estimators = 400, warm_start=True),
          XGBRegressor(base_score=0.43),
          Ridge(alpha=5, solver='saga', tol=0.09, max_iter=100),
          Lasso(max_iter=10, normalize=True, precompute=True, tol=0.001, warm_start=True)]
S_train, S_test = stacking(models, X_train, y_train, df_Test, regression = True, 
                           metric = 'mean_absolute_error', n_folds =5, stratified = True, shuffle = True)
regressor = LinearRegression()
regressor.fit(S_train, y_train)
y_pred = regressor.predict(S_test)

# Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator = regressor, X = S_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Saving model
filename = 'model.pkl'
pickle.dump(regressor, open(filename, 'wb'))

# Saving results into file
prediction = pd.DataFrame(y_pred)
df_final = pd.read_csv('sample_submission.csv')
datasets = pd.concat([df_final['Id'], prediction], axis=1)
datasets.columns = ['Id', 'SalePrice']
datasets.to_csv('sample_submission.csv', index=False)