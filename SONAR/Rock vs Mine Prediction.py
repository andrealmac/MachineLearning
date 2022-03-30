#importar as blibliotecas a serem usadas
import numpy as np #Criar matrizes
import pandas as pd #Carregar nossos dados
from sklearn.model_selection import train_test_split #Dividir os dados em Treinamento e teste
from sklearn.linear_model import LogisticRegression #Modelo de aprendizado de maquinas
from sklearn.metrics import accuracy_score #Modelo para realizar as metricas do modelo


#Coletando os dados e processando 

#Carregando os dados
sonar = pd.read_csv('C:/Users/andre/OneDrive/Documentos/GitHub/Machine_Learning/SONAR/sonar.all-data.csv', header=None)
sonar.head()

#Numeros de colunas e linhas
sonar.shape

#Descrição dos dados --> Medidas Estatisticas dos dados
sonar.describe()

#Descobrir quantos exemplos existem para R e M que esta na coluna 60
sonar[60].value_counts()

#M = MINE
#R = ROCK
#Valores medios de cada coluna de M e R
sonar.groupby(60).mean()

#Separar os dados em Labels 
X = sonar.drop(columns=60, axis=1)
Y = sonar[60]

#DIVIDR OS DADOS EM TRAIN E TEST
#train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

print(X.shape, X_train.shape, X_test.shape)

#Treinar nosso modelo -->Logistic Regression
model = LogisticRegression()

#Treinar o Modelo com os dados
model.fit(X_train, Y_train)

#Avaliar o modelo
#1-->Precisao do treinamento dos dados
X_train_prediction = model.predict(X_train)
train_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Precisao dos dados de treinamento: ',train_accuracy)

#2-->Precisao do teste dos dados
X_test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Precisao dos dados de teste: ',test_accuracy)

#REALIZAR UM SISTEMA PREDICTIVE
input_dt = (0.0286,0.0453,0.0277,0.0174,0.0384,0.0990,0.1201,0.1833,0.2105,0.3039,0.2988,0.4250,0.6343,0.8198,1.0000,0.9988,0.9508,0.9025,0.7234,0.5122,0.2074,0.3985,0.5890,0.2872,0.2043,0.5782,0.5389,0.3750,0.3411,0.5067,0.5580,0.4778,0.3299,0.2198,0.1407,0.2856,0.3807,0.4158,0.4054,0.3296,0.2707,0.2650,0.0723,0.1238,0.1192,0.1089,0.0623,0.0494,0.0264,0.0081,0.0104,0.0045,0.0014,0.0038,0.0013,0.0089,0.0057,0.0027,0.0051,0.0062)

#alterando os dados para matriz numpy
input_dt_as_numpy_array = np.asarray(input_dt)

#Prevendo uma instancia
input_reshaped = input_dt_as_numpy_array.reshape(1,-1)

previsao = model.predict(input_reshaped)
print(previsao)

#Condicao para verificar se e ROCK OU MINE e retornar R ou M
if (previsao[0]=='R'):
    print('O obj é uma ROCK')
else:
    print('O obj é uma MINE')