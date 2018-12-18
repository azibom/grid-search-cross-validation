# grid-search-cross-validation
how find optimize parameters in our model?

for example it is good question about the amount of "K" in knn model
how i should choose it

in this case you should write some codes like it to find the best value :pushpin:
 
```python
# import requirement
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
# load data
bcd = datasets.load_breast_cancer()
x = bcd.data
y = bcd.target
# divide data and fit model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

param_grid = {'n_neighbors':np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv = 5)
knn_cv.fit(x,y)
print(knn_cv.best_params_)
print(knn_cv.best_score_)

# result
# {'n_neighbors': 12}
# 0.9332161687170475
```

I hope this article will be useful to you.
