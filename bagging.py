import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


data_train = pd.read_csv('./processed_data/train_5.csv')
test = pd.read_csv('./processed_data/test_5.csv')

data_train_y = data_train.iloc[:, -1]
data_train.drop(columns=['exceeds50K'], inplace=True)
data_train = (data_train-data_train.mean()) / data_train.std()
data_train = pd.concat([data_train, data_train_y], axis=1)
test = (test-test.mean()) / test.std()

train = data_train.sample(frac=0.75, random_state=0, axis=0)
validate = data_train[~data_train.index.isin(train.index)]

train = np.array(train)
validate = np.array(validate)
test = np.array(test)

x_train = train[:, 0:-1]
y_train = train[:, -1]
x_validate = validate[:, 0:-1]
y_validate = validate[:, -1]

clf = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=50, bootstrap=True)
clf.fit(x_train, y_train)
y_validate_pred = clf.predict(x_validate)
print('Accuracy score:', metrics.accuracy_score(y_validate, y_validate_pred))
# KNN model, train_1 Accuracy score: 0.8283374283374283
# KNN, train_2 Accuracy score: 0.8337428337428338
# KNN, train_3 Accuracy score: 0.8298116298116298
# KNN, train_4 Accuracy score: 0.8314496314496315
# KNN, train_5 Accuracy score: 0.8384930384930385

train_all = np.array(data_train)
x_train_all = train_all[:, 0:-1]
y_train_all = train_all[:, -1]
clf2 = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=50, bootstrap=True)
clf2.fit(x_train_all, y_train_all)
test_pred = np.int_(clf2.predict(test))
print(test_pred)
idx = np.arange(len(test_pred))+1
result = np.vstack([idx, test_pred]).T
df_result = pd.DataFrame(data=result, columns=['id', 'prediction'])
df_result.to_csv('./result/bagging_res_process5.csv', index=None)
