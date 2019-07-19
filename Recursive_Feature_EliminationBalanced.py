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
from sklearn.preprocessing import MinMaxScaler

import numpy as np

data = pd.read_csv('results_corr_065_sem_modulo_mensal_sempolo_com_qtde_links_ajustado.csv')

#shuffle the data
from sklearn.utils import shuffle
data = shuffle(data)
print(data)
columns = []

# iterating the columns 
for col in data.columns: 
    columns.append(col)


# Creating an empty Dataframe with column names only
data_balanced = pd.DataFrame(columns=columns)
print(data_balanced)

N = 44#total of samples with flag equal 2

#get N samples with flag equal 0
count = 0 
for index, row in data.iterrows():
    if(row['flag'] == 0 and count<N):
        data_balanced.loc[len(data_balanced)] = row
        count += 1

#get N samples with flag equal 1
count = 0 
for index, row in data.iterrows():
    if(row['flag'] == 1 and count<N):
        data_balanced.loc[len(data_balanced)] = row
        count += 1


#get N samples with flag equal 2
count = 0 
for index, row in data.iterrows():
    if(row['flag'] == 2 and count<N):
        data_balanced.loc[len(data_balanced)] = row
        count += 1
        
print(data_balanced)

X, y = pd.DataFrame(), pd.DataFrame()

#shuffle the data
from sklearn.utils import shuffle
data_shuffle = shuffle(data_balanced)

#Selection of the used features
X['modularity'] = data_shuffle['modularity']
X['global_average_link_distance'] = data_shuffle['global_average_link_distance']
X['eigenvector'] = data_shuffle['eigenvector']
X['coreness'] = data_shuffle['coreness']
X['transitivity'] = data_shuffle['transitivity']
X['average_path_length'] = data_shuffle['average_path_length']
X['eccentricity'] = data_shuffle['eccentricity']
X['grauMedio'] = data_shuffle['grauMedio']
X['links'] = data_shuffle['links']

array_colunas = list(X.columns.values)

#normalize the features
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

#get the ONI flag as Y                     
y['flag'] = data_shuffle['flag']

X = np.asarray(X)
y = np.asarray(y)
y = y.astype('int')



print("Resultados Para o SVM:")

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
rfecv.fit(X, y.ravel())

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
rfecv.fit(X, y.ravel())

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
rfecv.fit(X, y.ravel())

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


