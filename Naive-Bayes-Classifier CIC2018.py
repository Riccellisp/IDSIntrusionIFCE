#BIBLIOTECAS UTILIZADAS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#LENDO O DATASET
df = pd.read_csv('C:/Users/mathe/OneDrive/Documentos/Estágio/DataSets/CIC2018ClusterCentroidUnderSampled.csv')

#SEPARA O UMA PARTE DO DATASET DE ACORDO COM O QUE VOCE QUER COLOCANDO AS LINHAS OU ATE MESMO UM ARRAY DE LINHAS.
#print(df.loc[[0,1,2,3,4,5]])
#O ILOC JA FUNCIONAN COLOCANDO A LINHA E A COLUNA QUE VC DESEJA QUE ELE PEGUE.
#df.iloc[<DADOS> 0:6,<FEATURES> 0:4]

#CRIANDO UMA VARIAVEL X E ALOCANDO UMA PARTE DO DATASET
X = df.iloc[:,0:70]
#CRIANDO UMA VARIAVEL Y E ALOCANDO OUTRA PARTE DO DATASET
y = df.iloc[:,-1]

#CRIANDO AS VARIAVEIS DE TREINO E TESTES QUE SERÃO USADAS PELO CLASSIFICADOR
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)
#print(pd.DataFrame(X_train))
#print(pd.DataFrame(X_test))

#CLASSIFICAÇÃO USANDO NAIVE-BAYES
clf = GaussianNB()
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)

#METRICA Accuracy_Score
print('Accyracy: ',accuracy_score(y_test, predicted))
#print(accuracy_score(y, predicted, normalize=False))

#METRICA F1_Score
#f1_score(y, predicted, average='macro')
#f1_score(y, predicted, average='micro')
print('F1: ',f1_score(y_test, predicted, average='weighted'))
#f1_score(y, predicted, average=None)
#f1_score(y, predicted, zero_division=1)

#METRICA Precision_Score
#precision_score(y, predicted, average='macro')
#precision_score(y, predicted, average='macro')
print('Precision: ',precision_score(y_test, predicted, average='weighted'))
#precision_score(y, predicted, average=None)
#precision_score(y, predicted, average=None)
#precision_score(y, predicted, average=None, zero_division=1)

#METRICA Recal_Score
#recall_score(y, predicted, average='macro')
#recall_score(y, predicted, average='micro')
print('Recal: ',recall_score(y_test, predicted, average='weighted'))
#recall_score(y, predicted, average=None)
#recall_score(y, predicted, average=None)
#recall_score(y, predicted, average=None, zero_division=1)


