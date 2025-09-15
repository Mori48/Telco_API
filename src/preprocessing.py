import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def load_raw_file(file_path):
    df = pd.read_csv(file_path)
    return df 

def find_categotical (df):
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = [col for col in df.columns 
                        if col not in numerical_cols + ['customerID', 'Churn'] and df[col].dtype == 'object']
    return categorical_cols

def preprocessing (df):
    df_processig = df.copy()
    df_processig = df_processig.drop(['customerID'], axis = 1)
    df_processig['TotalCharges'] = pd.to_numeric(df_processig.TotalCharges, errors='coerce')
    df_processig.drop(labels=df_processig[df_processig['TotalCharges'].isna()].index, inplace=True, axis=0) #drop NaN TotalCharges (11)
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = find_categotical(df_processig) 


    X = df_processig.drop(columns=['Churn']) 
    y = df_processig['Churn']
    y = LabelEncoder().fit_transform(y)    


    numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # или 'median', 'constant'
    ('scaler', StandardScaler())
])

    categorical_transform = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer( transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat',categorical_transform, categorical_cols)
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14)
    return X_train, X_test, y_train, y_test, preprocessor
