# IncomePredict
CS5228 Final Project.

Predict whether the income exceeds 50K.

### Preprecessing
1. Age: divided into 17-20, 21-30, 31-40, 41-50, 51-60, 61-70, 71-80, 81-90, encoded with simple continuous number 1-8.
2. Work class: divided into four part: private, government, self, other. One-hot encoding.
3. Fnlwgt: abandon
4. Education: abandon, because it’s related to education_num.
5. Education num: use directly.
6. Marital status: divided into five class: married, never-married, divorced, separated, widowed. One-hot encoding.
7. Occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. One-hot encoding.
8. Relationship: simply divided into 6 class and one-hot encoding.
9. Sex: simply label encoding into 1 and 2.
10. Capital-gain: Normalize.
11. Capital-loss: Normalize.
12. Hours-per-week: divided into 3 parts: x<35, 35<=x<=45 and x>45. Label encoding.
13. Native country:abandon, due to 90% are American.

### Model
1. AdaBoost
2. Linear Regression
3. Decision Tree
4. SVC
5. MLP
6. Bagging with KNN base model
7. Random Forest
