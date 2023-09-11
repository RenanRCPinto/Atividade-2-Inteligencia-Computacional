import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

#importando a base
base = pd.read_csv("car.data”)

#estatistica descritiva basica
estatistica = base.describe()

#separação do conjunto de input e target (previsores e classe)
Previsores = base.iloc[:,0:6].values
classe = base.iloc[:,6].values

#transformando dados categoricos em valores numericos
LEprevisores = LabelEncoder()

previsores[:,0] = LEprevisores.fit_transform(previsores[:,0])
previsores[:,1] = LEprevisores.fit_transform(previsores[:,1])
previsores[:,2] = LEprevisores.fit_transform(previsores[:,2])
previsores[:,3] = LEprevisores.fit_transform(previsores[:,2])
previsores[:,4] = LEprevisores.fit_transform(previsores[:,4])
previsores[:,5] = LEprevisores.fit_transform(previsores[:,5])

#normalização dos dados
scaler = StandardScaler()
previsores_normalizado = scaler.fit_transform(previsores)

#classifícador KNN
knn1 = KNeighborsClassifier(n_neighbors=11)
knn1.fit(previsores,classe)
res1 = knn1.predict(previsores)

knn2 = KNeighborsClassifier(n_neighbors=11)
knn2.fit(previsores_normalizado,classe)
res2 = knn1.predict(previsores_normalizado)

#matriz de confusao
CM_res1 = confusion_matrix(classe, res1)
CM_res2 = confusion_matrix(classe, res2)

