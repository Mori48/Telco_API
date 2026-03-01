import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer



def load_raw_file(file_path):
    df = pd.read_csv(file_path)
    return df 

def preprocessing (df):
    df_processig = df.copy()

    service_cols = [
    "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies"
    ]
    df_processig.drop(columns=['PhoneService','customerID','Churn'], inplace=True)

    df_processig["num_services"] = (df_processig[service_cols] == "Yes").sum(axis=1)

    df_processig.drop(columns=['gender'],inplace=True)
    df_processig['TotalCharges'] = pd.to_numeric(df_processig.TotalCharges, errors='coerce')
    df_processig['TotalCharges'] = df_processig['TotalCharges'].fillna(0)
    df_processig['Is_a_new_customer'] = (df_processig['TotalCharges']==0).astype(int)


    cat_cols = df_processig.select_dtypes(include="object").columns.tolist()
    num_cols = df_processig.select_dtypes(include=["int64", "float64"]).columns.tolist()

    X = df_processig
    y = df['Churn']
    y = LabelEncoder().fit_transform(y)    

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat_cols",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                cat_cols
            ),
            (
                "num",
                StandardScaler(),
                num_cols
            )
        ],
        remainder="drop"
    )
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14)
    return X_train, X_test, y_train, y_test, preprocessor
