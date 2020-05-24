#BIBLIOTECAS UTILIZADAS
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#LISTAS PARA GUARDAS VALORES ANTES DE SEREM EXPORTANDO PARA ARQUIVO .XLSX
Metricas = []
Tem = []
Acu = []
F1  = []
F1Mac  = []
F1Mic  = []
Pre = []
PreMac = []
PreMic = []
Rec = []
RecMac = []
RecMic = []

#LENDO O DATASET
df = pd.read_csv('C:/Users/mathe/OneDrive/Documentos/Estágio/DataSets/CIC2018ClusterCentroidUnderSampled.csv')

#CRIANDO UMA VARIAVEL X E ALOCANDO UMA PARTE DO DATASET
X = df.iloc[:,0:70]

#CRIANDO UMA VARIAVEL Y E ALOCANDO OUTRA PARTE DO DATASET
y = df.iloc[:,-1]

#repeticoes = int(input('Digite a quantidade de repeticoes:\n'))
#splits = int(input('Digite a quantidade de Splints:\n'))
print('\n========CLASSIFICADORES========\n')
print('(1)==AdaBoost==')
print('(2)==Kneighbors==')
print('(3)==Naive-Bayes==')
print('(4)==Nearest-Centroid==')
print('(5)==Random-Forest==')
print('(6)==Support-Vector-Machines==\n')
print('==============================\n')
classif = input('Digite qual Classificador Deseja Utilizar?\n')

if classif == '1':
    clf = AdaBoostClassifier()
    excelNome = 'AdaBoost-Classifier.xlsx'
elif classif == '2':
    clf = KNeighborsClassifier()
    excelNome = 'KNeighbors-Classifier.xlsx'
elif classif == '3':
    clf = GaussianNB()
    excelNome = 'Naive-Bayes-Classifier.xlsx'
elif classif == '4':
    clf = NearestCentroid()
    excelNome = 'Nearest-Centroid-Classifier.xlsx'
elif classif == '5':
    clf = RandomForestClassifier() 
    excelNome = 'Random-Forest-Classifier.xlsx'
elif classif == '6':
    clf = svm.SVC()
    excelNome = 'Support-Vector-Machines-Classifier.xlsx'
else:
    print('Erro!') 

excel = pd.ExcelWriter(excelNome, engine='xlsxwriter')

#DEFININDO A DIVISÃO
rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=2652124)
#RETORNA O NUMERO DE INTERAÇÕES DE DIVISÃO NO VALIDADOR CRUZADO
rkf.get_n_splits(X, y)
#print(rkf)
#print(X.columns)

#SEPARAND VARIAVEIS DE TREINO E TESTE
for train_index, test_index in rkf.split(X, y):
    print(train_index, test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #CLASSIFICAÇÃO USANDO NEAREST-CENTROID
    inicio = time.time()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    fim = time.time()
    tempo = fim - inicio
    Tem.append(tempo)
    
    #METRICA Accuracy_Score
    Acu.append(accuracy_score(y_test, y_pred))
    #print(accuracy_score(y_test, y_pred, normalize=False))

    #METRICA F1_Score
    F1Mac.append(f1_score(y_test, y_pred, average='macro'))
    F1Mic.append(f1_score(y_test, y_pred, average='micro'))
    F1.append(f1_score(y_test, y_pred, average='weighted'))
    #f1_score(y_test, y_pred, average=None)
    #f1_score(y_test, y_pred, zero_division=1)
    
    #METRICA Precision_Score
    PreMac.append(precision_score(y_test, y_pred, average='macro'))
    PreMic.append(precision_score(y_test, y_pred, average='micro'))
    Pre.append(precision_score(y_test, y_pred, average='weighted'))
    #precision_score(y_test, y_pred, average=None)
    #precision_score(y_test, y_pred, average=None, zero_division=1)
    
    #METRICA Recal_Score
    RecMac.append(recall_score(y_test, y_pred, average='macro'))
    RecMic.append(recall_score(y_test, y_pred, average='micro'))
    Rec.append(recall_score(y_test, y_pred, average='weighted'))
    #recall_score(y_test, y_pred, average=None)
    #recall_score(y_test, y_pred, average=None, zero_division=1)

print('MedTemp: ',np.mean(Tem))
print('MedAcu: ',np.mean(Acu))
print('MedF1: ',np.mean(F1))
print('MedF1Mac: ',np.mean(F1Mac))
print('MedF1Mic: ',np.mean(F1Mic))
print('MedPre: ',np.mean(Pre))
print('MedPreMac: ',np.mean(PreMac))
print('MedPreMic: ',np.mean(PreMic))
print('MedRec: ',np.mean(Rec))
print('MedRecMac: ',np.mean(RecMac))
print('MedRecMic: ',np.mean(RecMic),'\n')

print('DesvAcu: ',np.std(Acu))
print('DesvF1: ',np.std(F1))
print('DesvF1Mac: ',np.std(F1Mac))
print('DesvF1Mic: ',np.std(F1Mic))
print('DesvPre: ',np.std(Pre))
print('DesvPreMac: ',np.std(PreMac))
print('DesvPreMic: ',np.std(PreMic))
print('DesvRec: ',np.std(Rec))
print('DesvRecMac: ',np.std(RecMac))
print('DesvRecMic: ',np.std(RecMic))

#print(Metricas)
Metricas = {'Tempo':Tem, 
            'Accuracy':Acu,
            'F1':F1, 'F1Macro':F1Mac, 'F1Micro':F1Mic,
            'Pressision':Pre, 'PressisionMacro':PreMac, 'PressisionMicro':PreMic,
            'Recal':Rec, 'RecalMacro':RecMac, 'RecalMicro':RecMic}
Dados = pd.DataFrame(Metricas)
Dados.to_excel(excel, sheet_name='Planilha de Dados')
excel.save()   