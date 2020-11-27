import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


data_train = pd.read_csv('./processed_data/train_5.csv')
test = pd.read_csv('./processed_data/test_5.csv')

data_train_y = data_train.iloc[:, -1]
data_train.drop(columns=['exceeds50K'], inplace=True)
data_train = (data_train-data_train.mean()) / data_train.std()
data_train = pd.concat([data_train, data_train_y], axis=1)
test = (test-test.mean()) / test.std()

x_train_all = np.array(data_train)[:, 0:-1]
y_train_all = np.array(data_train)[:, -1]

train = data_train.sample(frac=0.75, random_state=0, axis=0)
validate = data_train[~data_train.index.isin(train.index)]

train = np.array(train)
validate = np.array(validate)
test = np.array(test)

x_train = train[:, 0:-1]
y_train = train[:, -1]
x_validate = validate[:, 0:-1]
y_validate = validate[:, -1]

clf = RandomForestClassifier(n_estimators=50)
clf.fit(x_train, y_train)
y_validate_pred = clf.predict(x_validate)
print('Accuracy score:', metrics.accuracy_score(y_validate, y_validate_pred))
# train_1 Accuracy score: 0.8283374283374283
# train_2 Accuracy score: 0.8321048321048321
# train_3 Accuracy score: 0.828009828009828
# train_4 Accuracy score: 0.8294840294840294
# train_5 Accuracy score: 0.8512694512694513


# # use cv and plot the error curve of n_estimator
# # The result shows that n_estimator around 25 is best enough, and accuracy score only 82.4%.
# superpa = []
# for i in range(200):
#     rfc = RandomForestClassifier(n_estimators=i+1, n_jobs=-1)
#     rfc_s = cross_val_score(rfc, x_train_all, y_train_all, cv=10).mean()
#     superpa.append(rfc_s)
# print(max(superpa), superpa.index(max(superpa)))
# plt.figure(figsize=[20, 5])
# plt.plot(range(1, 201), superpa)
# plt.show()


train_all = np.array(data_train)
x_train_all = train_all[:, 0:-1]
y_train_all = train_all[:, -1]
clf2 = RandomForestClassifier(n_estimators=50)
clf2.fit(x_train_all, y_train_all)
test_pred = np.int_(clf2.predict(test))
print(test_pred)
idx = np.arange(len(test_pred))+1
result = np.vstack([idx, test_pred]).T
df_result = pd.DataFrame(data=result, columns=['id', 'prediction'])
df_result.to_csv('./result/random_forest_res_process5.csv', index=None)