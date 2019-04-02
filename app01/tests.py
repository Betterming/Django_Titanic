from django.test import TestCase

# Create your tests here.

# titanic_LR
import pandas as pd
import numpy as np
from sklearn.model_selection import  GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


import warnings
warnings.filterwarnings("ignore")


#==========定义通用框架=============
def models(alg,parameters):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    # from sklearn.xgboost import XGBClassifier

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score, recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import roc_auc_score

    import warnings
    warnings.filterwarnings("ignore")

    train = pd.read_csv(r'media\train_processed.csv', dtype={"Age": np.float64})
    test = pd.read_csv(r'media\titanic\test_processed.csv', dtype={"Age": np.float64})

    X = train.values[:, 1:]
    y = train.values[:, 0]
    scorer = make_scorer(roc_auc_score)
    grid = GridSearchCV(alg,parameters,scoring=scorer,cv=5)
    grid.fit(X,y)
    y_pred = grid.predict(X)

    parameters = {'tol': [1 / pow(10, i) for i in range(4, 6, 1)], 'C': [i / 10 for i in range(1, 10, 2)],
                  'max_iter': range(5, 20, 5)}
    grid = GridSearchCV(LR, parameters, scoring='accuracy', cv=5)
    grid.fit(X, y)
    y_pred = grid.predict(X)
    # print(grid.best_params_,grid.best_score_)
    cvres = grid.cv_results_
    for mean_train_score, params in zip(cvres['mean_train_score'], cvres['params']):
        print(mean_train_score, params)


    #============列出所需要的算法=================
    alg1 = DecisionTreeClassifier(random_state=29)
    alg2 = SVC(probability=True,random_state=29)
    alg3 = RandomForestClassifier(random_state=29)
    alg4 = AdaBoostClassifier(random_state=29)
    alg5 = KNeighborsClassifier(n_jobs=-1)
    alg6 = LogisticRegression(random_state=29)

    #============参数范围========================
    parameters1 = {'max_depth':range(1,10),'min_samples_split':range(2,10)}
    parameters2 = {"C":range(1,20),"gamma":[0.05,0.10]}
    parameters3 = {'n_estimators':range(10,200,10),'max_depth':range(1,10),'min_samples_split':range(2,10)}
    parameters4 = {'n_estimators':range(10,200,10),'learning_rate':[i/10 for i in range(5,15)]}
    parameters5 = {'n_neighbors':range(2,10),'leaf_size':range(10,80,10)}
    parameters6 = {'tol':[1/pow(10,i) for i in range(4,6,1)],'C':[i/10 for i in range(1,10,2)],'max_iter':range(5,20,5)}

    # print("using model is {}:".format(alg.__class__.__name__))
    #print(grid.best_params_, grid.best_score_)
    dic = {'best_params': grid.best_params_, 'best_score': grid.best_score_}
    return dic

