from sklearn.model_selection import KFold
from sklearn import tree
import pandas as pd
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler
import numpy as np

data = pd.read_csv('results_corr_065_sem_modulo_mensal_sempolo_com_qtde_links_ajustado.csv')

columns = []

# iterating the columns 
for col in data.columns: 
    columns.append(col)


# Creating an empty Dataframe with column names only
data_balanced = pd.DataFrame(columns=columns)
data_class0 = pd.DataFrame(columns=columns)
data_class1 = pd.DataFrame(columns=columns)
data_class2 = pd.DataFrame(columns=columns)
 
N = 44#sampling size

#get the samples with flag equal 0

for index, row in data.iterrows():
    if(row['flag'] == 0):
        data_class0.loc[len(data_class0)] = row


#get the samples with flag equal 1

for index, row in data.iterrows():
    if(row['flag'] == 1):
        data_class1.loc[len(data_class1)] = row


#get the samples with flag equal 2

for index, row in data.iterrows():
    if(row['flag'] == 2):
        data_class2.loc[len(data_class2)] = row

data_balanced = data_balanced.append(data_class0.sample(N))
data_balanced = data_balanced.append(data_class1.sample(N))
data_balanced = data_balanced.append(data_class2.sample(N))

X, Y = pd.DataFrame(), pd.DataFrame()

#shuffle the data
from sklearn.utils import shuffle
data_shuffle = shuffle(data_balanced)

print(data_shuffle)

#Selection of the used features
X['modularity'] = data_shuffle['modularity']
X['global_average_link_distance'] = data_shuffle['global_average_link_distance']
#X['eigenvector'] = data_shuffle['eigenvector']
#X['coreness'] = data_shuffle['coreness']
X['transitivity'] = data_shuffle['transitivity']
#X['average_path_length'] = data_shuffle['average_path_length']
#X['eccentricity'] = data_shuffle['eccentricity']
#X['grauMedio'] = data_shuffle['grauMedio']
#X['links'] = data_shuffle['links']


#normalize the features
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

#get the ONI flag as Y                     
Y['flag'] = data_shuffle['flag']

X = np.asarray(X)
Y = np.asarray(Y)
Y = Y.astype('int')
unique_elements, counts_elements = np.unique(Y, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))

kf = KFold(n_splits=10)#divided the dataset in 10 parts and use  9 to train and 1 to teste

a = 0
f = 0
p = 0
r = 0
i = 0

for train_index, test_index in kf.split(X):
    i+=1
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    clf = tree.DecisionTreeClassifier().fit(X_train,y_train.ravel())
    y_pred = clf.predict(X_test)
    print(y_pred)
    print(np.asarray(y_test).reshape(1,-1))
    print("-------------------------------------------------------")
    a += clf.score(X_test,y_test)
    f += metrics.f1_score(y_test, y_pred, average='macro')
    p += metrics.precision_score(y_test, y_pred,average='macro')
    r += metrics.recall_score(y_test, y_pred,average='macro')

average_accuracy = a/i
average_f1_score = f/i
average_precision = p/i
average_recall = r/i

print('Accuracy: ')
print(average_accuracy)

print('F1 - Score:')
print(average_f1_score)

print('Precision:')
print(average_precision)

print('Recall:')
print(average_recall)




