"""
Midterm project
COVID death prediction using Mexico data 
Dashel Ruiz Perez 10/30/2023
"""
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import pickle
import warnings
warnings.filterwarnings(action='ignore')

# Importing dataset
print('Importing dataset...')
data = pd.read_csv('COVID19MEXICO2021.csv')

# Dropping columns will not be used
df = data.drop(columns=['FECHA_ACTUALIZACION', 'ID_REGISTRO', 'ORIGEN','SECTOR', 'ENTIDAD_UM',
                        'ENTIDAD_NAC', 'ENTIDAD_RES', 'MUNICIPIO_RES', 'TIPO_PACIENTE',
                        'FECHA_INGRESO', 'FECHA_SINTOMAS', 'OTRA_COM', 'OTRO_CASO',
                        'HABLA_LENGUA_INDIG', 'INDIGENA', 'NACIONALIDAD', 'MIGRANTE',
                        'TOMA_MUESTRA_LAB', 'RESULTADO_LAB','TOMA_MUESTRA_ANTIGENO',
                        'RESULTADO_ANTIGENO', 'PAIS_NACIONALIDAD', 'PAIS_ORIGEN', 'INTUBADO', 'UCI'])



print('Preparing dataset...')
# Selecting rows where the patient tested positive for COVID-19
df_covid_post = df[df.CLASIFICACION_FINAL < 4]
df_covid_post = df_covid_post.reset_index(drop=True)

# Cleaning data and converting data
cols = ['NEUMONIA', 'EMBARAZO',
       'DIABETES', 'EPOC', 'ASMA', 'INMUSUPR', 'HIPERTENSION',
       'CARDIOVASCULAR', 'OBESIDAD', 'RENAL_CRONICA', 'TABAQUISMO']

df_covid_post['SEXO'] = df_covid_post.SEXO.apply(lambda x: 'female' if x == 1 else 'male')

for col in cols:
    df_covid_post[col] = df_covid_post[col].apply(lambda x: 'yes' if x == 1 else 'no')


# Making a new column name 'decease' to record the patientes that died.
df_covid_post['decease'] = df_covid_post.FECHA_DEF.apply(lambda x: 0 if x == '9999-99-99' else 1)

# Dropping columsn used to get the patients that died
df_covid_post.drop(columns=['FECHA_DEF', 'CLASIFICACION_FINAL'], inplace=True)

# Converting columns to lower case
df_covid_post.columns = df_covid_post.columns.str.lower()

# Making the encoder
cat_cols = ['sexo','neumonia','embarazo', 'diabetes', 'epoc',
       'asma', 'inmusupr', 'hipertension', 'cardiovascular', 'obesidad',
       'renal_cronica', 'tabaquismo']
encoder = OrdinalEncoder()
df_covid_post[cat_cols] = encoder.fit_transform(df_covid_post[cat_cols])

# Scaling the 'edad' column
scaler = MinMaxScaler()
df_covid_post.edad = scaler.fit_transform(df_covid_post[['edad']])
df_covid_post.head()


# We will drop 'tabaquismo', 'embarazo', 'asma' variables due that the risk ratio is almost null
df_covid_post = df_covid_post.drop(columns=['tabaquismo', 'embarazo', 'asma'])


# Dividing the data set in X and y
X = df_covid_post.iloc[:, :-1]
y = df_covid_post.iloc[:, -1]


X_full_train, X_test, y_full_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
X_train, X_val, y_train, y_val = train_test_split(X_full_train, y_full_train, test_size=0.25, random_state=123)


# Making the model
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',  # Set the objective function for a binary classification task
    max_depth=5,                 # Maximum depth of trees
    learning_rate=0.1,           # Learning rate (shrinkage)
    n_estimators=100,            # Number of boosting rounds (trees)
    subsample=0.8,               # Fraction of samples used for each boosting round
    colsample_bytree=0.8,        # Fraction of features used for each boosting round
    reg_alpha=0.0,               # L1 regularization term on weights
    reg_lambda=1.0,              # L2 regularization term on weights
    random_state=123              # Set a seed for reproducibility
)

print('Training the model...')
xgb_model.fit(X_train.values, y_train)

# Make predictions on the test set using the best model
print('Making predictions')
time.sleep(1)

y_pred_xgb = xgb_model.predict_proba(X_val)[:,1]


auc = roc_auc_score(y_val, y_pred_xgb)
print(f'Result of te prediction with validation set, roc_auc -> {auc}')
print()
print(f'Result of te prediction with testing set, roc_auc -> {roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:,1])}')


# Explorting the model
time.sleep(2)
with open('xgb_model.bin', 'wb') as f_out: 
    pickle.dump(xgb_model, f_out)
    print(f'Model exported as xgb_model.bin')

