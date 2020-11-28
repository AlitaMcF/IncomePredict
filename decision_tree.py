import numpy as np
import pandas as pd
import sklearn
from sklearn import tree
import sklearn.metrics as metrics


data_train = pd.read_csv('./processed_data/train_5.csv')
test = pd.read_csv('./processed_data/test_5.csv')

# normalize
data_train_y = data_train.iloc[:, -1]
data_train.drop(columns=['exceeds50K'], inplace=True)
data_all = pd.concat([data_train, test], axis=0)
data_all = (data_all-data_all.mean()) / data_all.std()
data_train = data_all[0:data_train_y.shape[0]]
test = data_all[data_train_y.shape[0]:]
data_train = pd.concat([data_train, data_train_y], axis=1)

train = data_train.sample(frac=0.75, random_state=0, axis=0)
validate = data_train[~data_train.index.isin(train.index)]

train = np.array(train)
validate = np.array(validate)
test = np.array(test)

x_train = train[:, 0:-1]
y_train = train[:, -1]
x_validate = validate[:, 0:-1]
y_validate = validate[:, -1]

clf = tree.DecisionTreeClassifier()
clf.fit(x_train, y_train)
y_validate_pred = clf.predict(x_validate)
print('accuracy score:', metrics.accuracy_score(y_validate, y_validate_pred))
# train_1 81%
# train_2 accuracy score: 0.8203112203112203
# train_3 accuracy score: 0.8144144144144144
# train_4 accuracy score: 0.8144144144144144
# train_5 accuracy score: 0.8388206388206388


train_all = np.array(data_train)
x_train_all = train_all[:, 0:-1]
y_train_all = train_all[:, -1]
clf2 = tree.DecisionTreeClassifier()
clf2.fit(x_train_all, y_train_all)
test_pred = np.int_(clf2.predict(test))
print(test_pred)
idx = np.arange(len(test_pred))+1
result = np.vstack([idx, test_pred]).T
df_result = pd.DataFrame(data=result, columns=['id', 'prediction'])
df_result.to_csv('./result/decision_tree_res_process5.csv', index=None)
