import sys
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

def carrega_dados(file_name, separador=',', drop_cols=None):
    data = pd.read_csv(f"../data/{file_name}", sep=separador)
    print("Reorganizando os dados...")
    data = data.sample(frac=1)

    try:
        data = data if not drop_cols else data.drop(drop_cols, axis=1)
    except KeyError:
        print(f"Colunas {drop_cols} não encontradas no DataFrame.")
        sys.exit(1)

    print("Dados carregados")
    return data

def encoding_cols(data, cols):
    print("Transformando colunas categóricas em numéricas...")

    if not data.columns.isin(cols).any():
        print(f"Colunas {cols} não encontradas no DataFrame.")
        raise KeyError(f"Colunas {cols} não encontradas no DataFrame.")
    

    label_encoder = LabelEncoder()

    for col in cols:
        if data[col].dtype == 'object':
            data[col] = label_encoder.fit_transform(data[col])
        else:
            print(f"A coluna {col} não é categórica, não será transformada.")
            raise ValueError(f"A coluna {col} não é categórica, não será transformada.")
        
    print("Colunas categóricas transformadas em numéricas")
    
    return data

def standardize_data(X_train, X_test):
    print("Normalizandos os dados com z-score")
    scaler = StandardScaler()

    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
        raise ValueError("X_train e X_test devem ser DataFrames do pandas.")
    
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("X_train e X_test devem ter o mesmo número de colunas.")
    
    if X_train.isnull().values.any() or X_test.isnull().values.any():
        raise ValueError("X_train e X_test não podem conter valores nulos.")

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print("Dados normalizados com z-score")

    scaler_filename = 'scaler.pkl'
    with open(f"../artifacts/{scaler_filename}", 'wb') as file:
        pickle.dump(scaler, file)
    return X_train, X_test

def standardize_data_from_file(data, scaler_filename):
    print("Normalizando os dados com z-score")
    scaler = pickle.load(open(f"../artifacts/{scaler_filename}", 'rb'))

    if not isinstance(data, pd.DataFrame):
        raise ValueError("data devem ser DataFrames do pandas.")
    
    if data.isnull().values.any():
        raise ValueError("data não pode conter valores nulos.")

    data = scaler.transform(data)
    print("Dados normalizados com z-score")

    return data

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print("Modelo salvo")

def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def get_x_y(data, y_label, x_cols=None):
    print("Preparando amostras de treino e validação")
    try:
        X = data.drop(y_label, axis=1)
        X = X if not x_cols else X[x_cols]
        y = data[y_label]
    except KeyError:
        print(f"Colunas {y_label} ou {x_cols} não encontradas no DataFrame.")
        sys.exit(1)

    print("Amostras de treino e validação preparadas")

    return X, y
