import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics


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

clf = AdaBoostClassifier(n_estimators=50, random_state=0)
clf.fit(x_train, y_train)
y_validate_pred = clf.predict(x_validate)
print('Accuracy score:', metrics.accuracy_score(y_validate, y_validate_pred))
# train_1 83%
# train_2 Accuracy score: 0.8375102375102376
# train_3 Accuracy score: 0.8353808353808354
# train_4 Accuracy score: 0.8383292383292383
# train_5 Accuracy score: 0.8632268632268633


train_all = np.array(data_train)
x_train_all = train_all[:, 0:-1]
y_train_all = train_all[:, -1]
clf2 = AdaBoostClassifier(n_estimators=50, random_state=0)
clf2.fit(x_train_all, y_train_all)
test_pred = np.int_(clf2.predict(test))
print(test_pred)
idx = np.arange(len(test_pred))+1
result = np.vstack([idx, test_pred]).T
df_result = pd.DataFrame(data=result, columns=['id', 'prediction'])
df_result.to_csv('./result/adaboost_res_process5.csv', index=None)
