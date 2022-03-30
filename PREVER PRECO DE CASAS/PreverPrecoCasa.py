#Aprendizado Supervisionado 
#Regressao --> Prever quantidade  ou valores continuos ex: Salario, idade, preço
#Classificação -->Prever uma classe ou valores discretos ex: Masc, Fem, Verd, False

#1-->Conseguir os dados
import numpy as np
import pandas as pd

#plotar graficos
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
#Modelos
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

house_Price = sklearn.datasets.load_boston()
print(house_Price)

#2-->Pre-Processamento os dados
house_Price_dataframe = pd.DataFrame(house_Price.data, columns= house_Price.feature_names)
print(house_Price_dataframe.head())

#add Price
house_Price_dataframe['price'] = house_Price.target
print(house_Price_dataframe.head())

print(house_Price_dataframe.shape)

#Verificar se ha valores ausentes
print(house_Price_dataframe.isnull().sum())

print(house_Price_dataframe.describe())


#3-->Analise dos dados
#Correlação
#Positiva -->se um valor aumenta o outro tambem aumenta
#Negativa -->se um valor aumenta ou esta diminuindo
correlacao = house_Price_dataframe.corr()

#Mapa de calor para encontrar a correlacao
plt.figure(figsize=(10,10))
sns.heatmap(correlacao, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')

#3-->Treinamento e Teste
X = house_Price_dataframe.drop(['price'], axis=1)
Y = house_Price_dataframe['price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


#4-->Modelo regressor(XGBoost Regressor - Arvore)
modelo = XGBRegressor()
modelo.fit(X_train, Y_train)

#5-->Avaliacao do modelo
#Previsao do treinamento dos dados
previsao_Train = modelo.predict(X_train)

#R erro quadrado
score_1 = metrics.r2_score(Y_train, previsao_Train)
print('\nR2: ',score_1)

#Mean Absolute Error
score_2 = metrics.mean_absolute_error(Y_train, previsao_Train)
print('MAE: ',score_2,'\n')

#Plotar um grafico com os precos atual e previsao
plt.scatter(Y_train, previsao_Train)
plt.xlabel('Real Preco')
plt.ylabel('Previsao Preco')
plt.title("Preco real vs Previsao")
plt.show()

#Previsao do teste dos dados
previsao_Test = modelo.predict(X_test)

#R erro quadrado
score_3 = metrics.r2_score(Y_test, previsao_Test)
print('R2: ',score_3)

#Mean Absolute Error
score_4 = metrics.mean_absolute_error(Y_test, previsao_Test)
print('MAE: ',score_4)