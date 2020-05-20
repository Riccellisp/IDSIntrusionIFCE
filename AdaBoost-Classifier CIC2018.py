#BIBLIOTECAS UTILIZADAS
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

excel = pd.ExcelWriter('AdaBoost-Classifier.xlsx', engine='xlsxwriter')

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

Metricas = []
Tem = []
Acu = []
F1  = []
Pre = []
Rec = []

for x in range(30):
    print('Teste:',x,'\n')
    #CRIANDO AS VARIAVEIS DE TREINO E TESTES QUE SERÃO USADAS PELO CLASSIFICADOR
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    #print(pd.DataFrame(X_train))
    #print(pd.DataFrame(X_test))
        
    #CLASSIFICAÇÃO USANDO ADABOOST-CLASSIFIER
    clf = RandomForestClassifier()
    inicio = time.time()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    fim = time.time()
    tempo = fim - inicio
    #print("%.0f" % tempo,'Segundos')
    Tem.append(tempo)
        
    #METRICA Accuracy_Score
    Acu.append(accuracy_score(y_test, y_pred))
    #print(accuracy_score(y, predicted, normalize=False))
        
    #METRICA F1_Score
    #f1_score(y, predicted, average='macro')
    #f1_score(y, predicted, average='micro')
    F1.append(f1_score(y_test, y_pred, average='weighted'))
    #f1_score(y, predicted, average=None)
    #f1_score(y, predicted, zero_division=1)
        
    #METRICA Precision_Score
    #precision_score(y, predicted, average='macro')
    #precision_score(y, predicted, average='macro')
    Pre.append(precision_score(y_test, y_pred, average='weighted'))
    #precision_score(y, predicted, average=None)
    #precision_score(y, predicted, average=None)
    #precision_score(y, predicted, average=None, zero_division=1)
        
    #METRICA Recal_Score
    #recall_score(y, predicted, average='macro')
    #recall_score(y, predicted, average='micro')
    Rec.append(recall_score(y_test, y_pred, average='weighted'))
    #recall_score(y, predicted, average=None)
    #recall_score(y, predicted, average=None)
    #recall_score(y, predicted, average=None, zero_division=1)

print('MedTemp: ',np.mean(Tem))
print('MedAcu: ',np.mean(Acu))
print('MedF1: ',np.mean(F1))
print('MedPre: ',np.mean(Pre))
print('MedRec: ',np.mean(Rec),'\n')

print('DesvTemp: ',np.std(Tem))
print('DesvAcu: ',np.std(Acu))
print('DesvF1: ',np.std(F1))
print('DesvPre: ',np.std(Pre))
print('DesvRec: ',np.std(Rec))

#print(Metricas)
Metricas = {'Tempo':Tem, 'Accuracy':Acu, 'F1':F1, 'Pressision':Pre, 'Recal':Rec}
Dados = pd.DataFrame(Metricas)
Dados.to_excel(excel, sheet_name='Planilha de Dados')
excel.save()


