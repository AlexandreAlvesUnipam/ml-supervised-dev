import sys
sys.path.append('/workspaces/ml-supervised-dev')

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import tree
from src.config.properties import DATA_FOLDER

def carrega_dados(file_path, separador=',', drop_cols=None):
    data = pd.read_csv(f"../{DATA_FOLDER}/{file_path}", sep=separador)
    return data if not drop_cols else data.drop(drop_cols, axis=1)

def encoding_cols(col):
    label_encoder = LabelEncoder()
    coluna_transformada = label_encoder.fit_transform(col)
    return coluna_transformada

## Salvar modelo

## Carregar o modelo

## Amostragem

## Treino

## Medidas

## Plotar árvore

if __name__=="__main__":
    df = carrega_dados("Iris.csv", drop_cols=['Id'])
    print(df.head())