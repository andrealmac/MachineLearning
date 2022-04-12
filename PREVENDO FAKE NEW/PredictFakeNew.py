#1--> Coleta de daods
import numpy as np
import pandas as pd
#Expressao regular usado para pesquisar texto
import re
#Linguagem natural
#Palavras que nao tras valor para a pesquisa
from nltk.corpus import stopwords
#Pegue prefixo e sufixo das palavras e retorne a palavra raiz
from nltk.stem.porter import PorterStemmer
#Converter o texto em vetores
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')
print(stopwords.words('english'))

#2--> Pre Processamento dos dados
news_dataset = pd.read_csv('C:/Users/andre/OneDrive/Documentos/GitHub/Machine_Learning/PREVENDO FAKE NEW/train.csv')
print(news_dataset.shape)
#0 para REAL NEWS e 1 para FAKE NEWS
print(news_dataset.head())
#verificar valores faltantes
print(news_dataset.isnull().sum())

#3--> Treinamento e teste
#4--> Modelo de regressao logistica -- classificacao binaria(True and False)
#5--> Modelo treinado, Avaliacao do modelo
#6--> Novos dados
#7--> Prever se e falsa ou verdadeira a noticia