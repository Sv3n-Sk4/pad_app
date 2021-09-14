import time
start_time = time.time()


import pandas as pd
import numpy as np
data = pd.read_csv('./model_data.csv')

# Encoding & Features Engineering
df = data.copy()

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Import pour la création de pipelines
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector


def feat_engi(df):
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH'] 
    return df

df = feat_engi(df)

y = df['TARGET']
X = df.drop(['TARGET'], axis=1)

all_features = X.columns.to_list()

numerical_features = make_column_selector(dtype_include = np.number)

categorical_features = make_column_selector(dtype_exclude= np.number)

numerical_pipeline = make_pipeline(SimpleImputer(strategy='mean'), 
                                   StandardScaler())
                                   
categorical_pipeline = make_pipeline(SimpleImputer(strategy = 'constant'),
                                    OneHotEncoder(handle_unknown = 'ignore'))

preprocessor = make_column_transformer((numerical_pipeline, numerical_features),
                                       (categorical_pipeline, categorical_features))

import lightgbm as lgb
lgbm = lgb.LGBMClassifier(learning_rate = 0.013359443291300465, max_depth = 22, n_estimators = 1400, num_leaves = 92, reg_alpha = 0.2, reg_lambda = 0.8, subsample = 0.6)


pipeline = Pipeline(steps= [
        ('preprocessor', preprocessor),
        ('classifier', lgbm)
        ])

pipeline.fit(X,y)

# Sauvegarde du pipeline
import pickle
pickle.dump(pipeline, open('data_clf.pkl', 'wb'))

# Sauvegarde du modèle declassifier uniquement
import pickle
pickle.dump(lgbm, open('model_lgbm.pkl', 'wb'))

print("--- %s seconds ---" % (time.time() - start_time))