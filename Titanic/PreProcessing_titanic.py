# Este arquivo é referente a etapa de pré processamento do problema do titanic.
# Ele tem como objetivo, resumir toda esta etapa em uma única função, de modo que a entrada seja um dataframe original e a saída seja um outro dataframe processado.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def PP_titanic_function(df_model):
    # Executando uma limpeza básica dos dados, excluindo as colunas que não auxiliam na predição:
    df_model.drop(columns=["PassengerId", "Name", "Ticket"], inplace=True)

    # Tratando valores nulos com base no dtype da coluna:
    for col in df_model.columns:
        if pd.api.types.is_numeric_dtype(df_model[col]):
            # Preencher nulos com -1 se a coluna for numérica
            df_model[col].fillna(-1, inplace=True)
        else:
            # Preencher nulos com 'NA' se a coluna não for numérica
            df_model[col].fillna("NA", inplace=True)

    # Criando uma nova feature
    df_model["Taxa_x_SibSp"] = df_model["Fare"] * df_model["SibSp"]

    # Corrigindo o dtype da coluna 'Pclass' para object:
    df_model['Pclass'] = df_model['Pclass'].astype(str)

    return df_model
