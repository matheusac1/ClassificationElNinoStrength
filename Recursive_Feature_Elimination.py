from sklearn.feature_selection import RFECV
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
import numpy as np

dados = pd.read_csv('results_corr_065_sem_modulo_mensal_sempolo_com_qtde_links_ajustado.csv')


X = pd.DataFrame()

X['modularity'] = dados['modularity']
X['global_average_link_distance'] = dados['global_average_link_distance']
X['eigenvector'] = dados['eigenvector']
X['coreness'] = dados['coreness']
X['transitivity'] = dados['transitivity']
X['average_path_length'] = dados['average_path_length']
X['eccentricity'] = dados['eccentricity']
X['pagerank'] = dados['pagerank']
X['grauMedio'] = dados['grauMedio']
X['links'] = dados['links']

y = dados['flag']
array_colunas = list(X.columns.values)

print("Resultados Para o SVM:")

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)
features = [array_colunas,list(rfecv.get_support())]
features = np.asarray(features)
print(np.transpose(features))
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

print("Resultados Para o Decision Tree:")

# Create the RFE object and compute a cross-validated score.
tree_clf = tree.DecisionTreeClassifier()
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=tree_clf, step=1, cv=StratifiedKFold(10),
              scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)
features = [array_colunas,list(rfecv.get_support())]
features = np.asarray(features)
print(np.transpose(features))
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

print("Resultados Para o Random Forest:")

# Create the RFE object and compute a cross-validated score.
clf_random_forest = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=clf_random_forest, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)
features = [array_colunas,list(rfecv.get_support())]
features = np.asarray(features)
print(np.transpose(features))
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


