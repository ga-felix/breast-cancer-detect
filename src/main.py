import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Leitura do dataset e separação das colunas de interesse
df = pd.read_csv("recursos/breast-cancer.csv")
cdf = df.drop(["id"], axis=1)

# Filtragem de missing data e análise exploratória
cdf = cdf[(cdf["clump_thickness"] != "?")
        & (cdf["uniformity_of_cell_size"] != "?")
        & (cdf["uniformity_of_cell_shape"] != "?")
        & (cdf["marginal_adhesion"] != "?")
        & (cdf["single_epithelial_cell_size"] != "?")
        & (cdf["bare_nuclei"] != "?")
        & (cdf["bland_chromatin"] != "?")
        & (cdf["normal_nucleoli"] != "?")
        & (cdf["mitoses"] != "?")
        & (cdf["class"] != "?")]
cdf = cdf.replace({2: 0}).replace({4: 1})
report = open('relatorios/relatorio.txt', 'w+')
report.write("Dimensões do dataframe (Linhas x Colunas): " + str(cdf.shape) + "\n")
report.write("\nDescrição geral: " + str(cdf.describe()))

# Visualização da frequência de tumores benignos e malignos
arr = np.array(cdf["class"].replace({0: "Benigno"}).replace({1: "Maligno"}))
labels, counts = np.unique(arr, return_counts=True)
plt.bar(labels, counts, align='center')
plt.gca().set_xticks(labels)
plt.title("Tumores na mama")
plt.ylabel("Ocorrências")
plt.show()

# Separação das variáveis
x = cdf.drop(["class"], axis=1)
y = cdf[["class"]]

# Padronizar as escalas numéricas das variáveis independentes
x_norm = StandardScaler().fit_transform(x)

# Divir os datasets de treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Treino da regressão logística
logit = LogisticRegression()
y_train = np.array(y_train["class"])
logit.fit(x_train, y_train)

# Predição
y_pred = logit.predict(x_test)

# Medições de desempenho
## Acurácia
report.write("\nAcurácia do modelo: " + str(logit.score(x_test, y_test)) + "\n")
## Matriz de confusão
report.write("\nMatriz de confusão\n" + str(confusion_matrix(y_test, y_pred)) + "\n")
## Acurácia, precisão, recall, F1-score
report.write("\nAcurácia, precisão, recall, F1-score\n" + str(classification_report(y_test, y_pred)))