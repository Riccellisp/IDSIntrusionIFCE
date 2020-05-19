#BIBLIOTECAS UTILIZADAS
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

excel = pd.ExcelWriter('Support-Vector-Machines-Classifier.xlsx', engine='xlsxwriter')

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

#LISTA PARA ARMAZENAR AS METRICAS
Metricas = []

for x in range(30):
    #CRIANDO AS VARIAVEIS DE TREINO E TESTES QUE SERÃO USADAS PELO CLASSIFICADOR
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    #print(pd.DataFrame(X_train))
    #print(pd.DataFrame(X_test))
    
    #CLASSIFICAÇÃO USANDO SUPORT VECTOR MACHINES
    clf = svm.SVC()
    inicio = time.time()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    fim = time.time()
    tempo = fim - inicio
    #print("%.0f" % tempo,'Segundos')
    Metricas.append(tempo)
    
    #METRICA Accuracy_Score
    Metricas.append(accuracy_score(y_test, y_pred))
    #print(accuracy_score(y_test, y_pred, normalize=False))
    
    #METRICA F1_Score
    #f1_score(y_test, y_pred, average='macro')
    #f1_score(y_test, y_pred, average='micro')
    Metricas.append(f1_score(y_test, y_pred, average='weighted'))
    #f1_score(y_test, y_pred, average=None)
    #f1_score(y_test, y_pred, zero_division=1)
    
    #METRICA Precision_Score
    #precision_score(y_test, y_pred, average='macro')
    #precision_score(y_test, y_pred, average='macro')
    Metricas.append(precision_score(y_test, y_pred, average='weighted'))
    #precision_score(y_test, y_pred, average=None)
    #precision_score(y_test, y_pred, average=None)
    #precision_score(y_test, y_pred, average=None, zero_division=1)
    
    #METRICA Recal_Score
    #recall_score(y_test, y_pred, average='macro')
    #recall_score(y_test, y_pred, average='micro')
    Metricas.append(recall_score(y_test, y_pred, average='weighted'))
    #recall_score(y_test, y_pred, average=None)
    #recall_score(y_test, y_pred, average=None)
    #recall_score(y_test, y_pred, average=None, zero_division=1)

#print(Metricas)
Dados = pd.DataFrame(Metricas)
Dados.to_excel(excel, sheet_name='Planilha de Dados')
excel.save() 
