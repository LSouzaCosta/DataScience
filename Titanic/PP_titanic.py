# Este arquivo é referente a etapa de pré processamento do problema do titanic.
# Ele tem como objetivo, resumir toda esta etapa em uma única função, de modo que a entrada seja um dataframe original e a saída seja um outro dataframe processado.

def PP_titanic_function(df_model):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Executando uma limpeza básica dos dados, excluindo as colunas que não auxiliam na predição:
    df_model.drop(columns=["PassengerId", "Name", "Ticket"], inplace= True)

    # Dropando as colunas desnecessárias:
    df_model["Age"] = df_model["Age"].fillna(-1)
    df_model["Cabin"] = df_model["Cabin"].fillna('NA')
    df_model["Embarked"] = df_model["Embarked"].fillna("NA")

    # Criando uma nova feature
    df_model["Taxa_x_SibSp"] = df_model["Fare"] * df_model["SibSp"]

    # Corrigindo o dtype da coluna 'Pclass' para object:
    df_model['Pclass'] = df_model['Pclass'].astype(str)

    return df_model