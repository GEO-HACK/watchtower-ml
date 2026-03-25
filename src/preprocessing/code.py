from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
import numpy as np # Import numpy to use np.inf and np.nan

preprocessing_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', RobustScaler()),
])


X_train_cleaned = X_train.replace([np.inf, -np.inf], np.nan)
X_test_cleaned = X_test.replace([np.inf, -np.inf], np.nan)


X_train_processed = preprocessing_pipeline.fit_transform(X_train_cleaned)
X_test_processed = preprocessing_pipeline.transform(X_test_cleaned)