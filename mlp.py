import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import time

data_train = pd.read_csv('./processed_data/train_5.csv')
test = pd.read_csv('./processed_data/test_5.csv')

data_train_y = data_train.iloc[:, -1]
data_train.drop(columns=['exceeds50K'], inplace=True)
data_train = (data_train-data_train.mean()) / data_train.std()
data_train = pd.concat([data_train, data_train_y], axis=1)
test = (test-test.mean()) / test.std()

train = data_train.sample(frac=0.75, axis=0, random_state=0)
validate = data_train[~data_train.index.isin(train.index)]

train = np.array(train)
validate = np.array(validate)
test = np.array(test)

x_train = train[:, 0:-1]
y_train = train[:, -1]
x_validate = validate[:, 0:-1]
y_validate = validate[:, -1]

clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(20, 2), random_state=1)
clf.fit(x_train, y_train)
y_validate_pred = clf.predict(x_validate)
print('Accuracy score:', metrics.accuracy_score(y_validate, y_validate_pred))
# train_1 Accuracy score: 0.8370188370188371
# train_2 Accuracy score: 0.8357084357084357
# train_3 Accuracy score: 0.8343980343980344
# train_4 Accuracy score: 0.843079443079443
# train_5 Accuracy score: 0.8586404586404587


# # find the best hyper-parameters
# train_all = np.array(data_train)
# x_train_all = train_all[:, 0:-1]
# y_train_all = train_all[:, -1]
# max_score = 0
# hyperpara = 0
# size_hidden_layer1 = np.arange(60)+1
# for i in size_hidden_layer1:
#     clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(i, 2), random_state=1)
#     score = np.mean(cross_val_score(clf, x_train_all, y_train_all, cv=10))
#     if score > max_score:
#         max_score = score
#         hyperpara = i
# print('Max accuracy score:', max_score)
# print('hyperpara:', hyperpara)

train_all = np.array(data_train)
x_train_all = train_all[:, 0:-1]
y_train_all = train_all[:, -1]
clf2 = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(20, 2), random_state=1)
clf2.fit(x_train_all, y_train_all)
test_pred = np.int_(clf2.predict(test))
print(test_pred)
idx = np.arange(len(test_pred))+1
result = np.vstack([idx, test_pred]).T
df_result = pd.DataFrame(data=result, columns=['id', 'prediction'])
df_result.to_csv('./result/mlp_res_process5.csv', index=None)
