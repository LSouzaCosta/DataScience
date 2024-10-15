# Este arquivo é referente a etapa de pré processamento do problema do titanic.
# Ele tem como objetivo, resumir toda esta etapa em uma única função, de modo que a entrada seja um dataframe original e a saída seja um outro dataframe processado.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def PP_titanic_function(df_model):
    # Executando uma limpeza básica dos dados, excluindo as colunas que não auxiliam na predição:
    df_model.drop(columns=["PassengerId", "Name", "Ticket"], inplace= True)

    # Tratando valores nulos com base no dtype da coluna:
    # Criando um dicionário para preencher os valores de acordo com o tipo da coluna:
    fill_values = {col: -1 if df_model[col].dtype in ['int64', 'float64'] else 'NA' for col in df_model.columns}
    
    # Usando fillna() com o dicionário
    df_model.fillna(value=fill_values, inplace=True)
    """df_model["Age"] = df_model["Age"].fillna(-1)
    df_model["Cabin"] = df_model["Cabin"].fillna('NA')
    df_model["Embarked"] = df_model["Embarked"].fillna("NA")"""

    # Criando uma nova feature
    df_model["Taxa_x_SibSp"] = df_model["Fare"] * df_model["SibSp"]

    # Corrigindo o dtype da coluna 'Pclass' para object:
    df_model['Pclass'] = df_model['Pclass'].astype(str)

    return df_model