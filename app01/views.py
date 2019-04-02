from django.shortcuts import render, redirect,HttpResponse

from app01 import models
# Create your views here.
from app01 import tests
import os
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


def register(req):
    if req.method == "POST":
        user_name = req.POST.get('name', None)
        user_pwd = req.POST.get('pwd', None)
        user_type = req.POST.get('type', None)
        user_email = req.POST.get('email', None)
        print(user_name,user_pwd)

        if models.userInfo.objects.filter(name=user_name).values():  # 如果用户名存在，直接转到登陆
            return redirect('/login')
        else:
            models.userInfo.objects.create(     # 否则将注册信息写到数据库
                name=user_name,
                pwd=user_pwd,
                type=user_type,
                email=user_email
            )

        return redirect('/login')              # 注册完成后重定向到主页
    return render(req, 'register.html')


def login(req):
    if req.method == "POST":
        user_name = req.POST.get('name', None)
        user_pwd = req.POST.get('pwd', None)
        user_type = req.POST.get('type', None)

        if models.userInfo.objects.filter(name=user_name, pwd=user_pwd, type="普通用户").values():

            return redirect('/home_ordinary')  # 验证通过转到主页
        elif models.userInfo.objects.filter(name=user_name, pwd=user_pwd, type='管理员').values():

            return redirect('/home_manager')  # 验证通过转到主页
        elif models.userInfo.objects.filter(name=user_name, pwd=user_pwd, type='开发者用户').values():

            return redirect('/home_coder')  # 验证通过转到主页
        else:
            return render(req, 'login.html')  # 验证不通过重新登陆
    return render(req,'login.html')


def search(req):
    if req.method == "POST":
        user_name = req.POST.get('name', None)
        print(user_name)


def home_ordinary(req):

    return render(req, 'home_ordinary.html')


def home_coder(req):

    return render(req, 'home_coder.html')


def home_manager(req):
    if req.method == "POST":
        if 'add' in req.POST:
            return redirect('/home_manager_add')
        elif 'find' in req.POST:
            return redirect('/home_manager_find')
        else:
            return redirect('/home_manager')
    return render(req, 'home_manager.html')


def home_manager_add(req):
    if req.method == "POST" and 'return' in req.POST:
        return redirect('/home_manager')
    elif req.method == "POST":
        user_name = req.POST.get('name', None)
        user_pwd = req.POST.get('pwd', None)
        user_type = req.POST.get('type', None)
        user_email = req.POST.get('email', None)
        if models.userInfo.objects.filter(name=user_name, email=user_email, type=user_type).values():
            return redirect('/home_manager_add')
        else:
            models.userInfo.objects.create(     # 否则将注册信息写到数据库
                name=user_name,
                pwd=user_pwd,
                type=user_type,
                email=user_email
            )
        return redirect('/home_manager')
    return render(req, 'home_manager_add.html')


def home_manager_find(req):
    if req.method == "POST" and 'return' in req.POST:
        return redirect('/home_manager')
    elif req.method == "POST":
        user_name = req.POST.get('name', None)
        if models.userInfo.objects.filter(name=user_name).values():
            home_manager_find = models.userInfo.objects.filter(name=user_name)
            return render(req, 'home_manager_find.html', {'home_manager_find': home_manager_find})
        else:
            wrong = '用户名不存在'
            return render(req, 'home_manager_find.html', {'info':wrong})
    return render(req, 'home_manager_find.html')


def delete(req):
    if req.method == "POST" and 'del' in req.POST:
        user_name = req.POST.get('line.name', None)
        models.userInfo.objects.filter(name=user_name).delete()
        return render(req, 'home_manager_find.html')
    return render(req, 'home_manager_find.html')


def home_manager_file(req):
    if req.method == "GET":
        return render(req, 'home_manager.html')
    elif req.method == "POST":
        obj = req.FILES.get('txt_file')
        f = open(os.path.join('media', obj.name), 'wb')
        for line in obj.chunks():
            f.write(line)
        f.close()

        filename_list = os.listdir(r'C:\Users\chenyuan\Desktop\django_templates3.13\django_templates3.13\media')
        data_set = []
        for i in filename_list:
            if i[-3:] == 'csv':
                data_set.append(i)
        return render(req, 'home_manager.html', {'data':data_set})
    return render(req, 'home_manager.html')


def home_manager_model(req):
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(BASE_DIR, 'media')

    if req.method == "GET":
        return render(req, 'home_manager.html')
    elif req.method == "POST":
        obj = req.FILES.get('model')
        f = open(os.path.join('media', obj.name), 'wb')
        for line in obj.chunks():
            f.write(line)
        f.close()
        model_list = os.listdir(path)
        model_set = []
        for i in model_list:
            if i[-3:] == 'csv':
                model_set.append(i)
        return render(req, 'home_manager.html', {'model': model_set})
    return render(req, 'home_manager.html')


def model1(req):
    return render(req, '逻辑回归模型.html')


def model2(req):
    return render(req, '随机森林模型.html')


def model3(req):
    return render(req, '决策树模型.html')


def model4(req):
    return render(req, 'K近邻模型.html')


def model5(req):
    return render(req, 'SVC模型.html')


def model6(req):
    return render(req, 'AdaBoostClassifier模型.html')


def Ajax_recv(req):

    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(BASE_DIR, 'media')
    #   =========导入模块===========
    import json
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    alg_name = req.POST.get('alg', None)   # 判断使用哪个算法

    #   =========读入数据=======================
    file_path = path + r'\train_processed.csv'
    train = pd.read_csv(file_path, dtype={"Age": np.float64})
    X = train.values[:, 1:]
    y = train.values[:, 0]

    if alg_name == "LR":
        dataset_num = req.POST.get('dataset_num', None)
        splite_rate = req.POST.get('splite_rate', None)

        C = req.POST.get('C', None)
        fit_intercept = req.POST.get('fit_intercept', None)
        tol = req.POST.get('tol', None)
        max_iter = req.POST.get('max_iter', None)
        print(alg_name, dataset_num, splite_rate, C, fit_intercept, tol, max_iter)

    #   =========确定算法======================
        if(fit_intercept == 1):
            fit_intercept = 'true'
        tol = pow(10, -4)
        C = float(C)
        max_iter = int(max_iter)

        alg = LogisticRegression(random_state=29, C=C, fit_intercept=fit_intercept, tol=tol, max_iter=max_iter)

    #   =========返回准确率=====================
        lst = cross_val_score(alg, X, y, cv=3, scoring='accuracy')
        precision = np.mean(lst)
        print(precision)
        dic = {'best_score': precision}
        return HttpResponse(json.dumps(dic))

    if alg_name == 'RF':
        #   =========获取超参数=======================
        dataset_num = req.POST.get('dataset_num', None)
        splite_rate = req.POST.get('splite_rate', None)

        n_estimators = int(req.POST.get('n_estimators', None))
        min_samples_split = int(req.POST.get('min_samples_split', None))
        max_depth = int(req.POST.get('max_depth', None))

        alg = RandomForestClassifier(random_state=29, n_estimators=n_estimators,
                                     min_samples_split=min_samples_split, max_depth=max_depth)

        #   =========返回准确率=====================
        lst = cross_val_score(alg, X, y, cv=3, scoring='accuracy')
        precision = np.mean(lst)
        print(precision)
        dic = {'best_score': precision}
        print("RF OK")
        return HttpResponse(json.dumps(dic))

    if alg_name == 'DT':
        #   =========获取超参数=======================
        dataset_num = req.POST.get('dataset_num', None)
        splite_rate = req.POST.get('splite_rate', None)

#        n_estimators = int(req.POST.get('n_estimators', None))
        min_samples_split = int(req.POST.get('min_samples_split', None))
        max_depth = int(req.POST.get('max_depth', None))

        alg =DecisionTreeClassifier(random_state=29,
                                    min_samples_split=min_samples_split, max_depth=max_depth)

        #   =========返回准确率=====================
        lst = cross_val_score(alg, X, y, cv=3, scoring='accuracy')
        precision = np.mean(lst)
        print(precision)
        dic = {'best_score': precision}
        print("DT OK")
        return HttpResponse(json.dumps(dic))

    if alg_name == 'KNN':
        #   =========获取超参数=======================
        dataset_num = req.POST.get('dataset_num', None)
        splite_rate = req.POST.get('splite_rate', None)

        leaf_size = int(req.POST.get('leaf_size', None))
        n_neighbors = int(req.POST.get('n_neighbors', None))

        alg = KNeighborsClassifier(n_jobs=-1, n_neighbors=n_neighbors, leaf_size=leaf_size)

        #   =========返回准确率=====================
        lst = cross_val_score(alg, X, y, cv=3, scoring='accuracy')
        precision = np.mean(lst)
        print(precision)
        dic = {'best_score': precision}
        print("KNN OK")
        return HttpResponse(json.dumps(dic))

    if alg_name == 'SVC':
        #   =========获取超参数=======================
        dataset_num = req.POST.get('dataset_num', None)
        splite_rate = req.POST.get('splite_rate', None)

        C = int(req.POST.get('C', None))
        gamma = float(req.POST.get('gamma', None))

        alg = SVC(probability=True, random_state=29, gamma=gamma, C=C)

        #   =========返回准确率=====================
        lst = cross_val_score(alg, X, y, cv=3, scoring='accuracy')
        precision = np.mean(lst)
        print(precision)
        dic = {'best_score': precision}
        print("SVC OK")
        return HttpResponse(json.dumps(dic))

    if alg_name == 'AdaBootClassifier':
        #   =========获取超参数=======================
        dataset_num = req.POST.get('dataset_num', None)
        splite_rate = req.POST.get('splite_rate', None)

        n_estimators = int(req.POST.get('n_estimators', None))
        learning_rate = float(req.POST.get('learning_rate', None))

        alg = AdaBoostClassifier(random_state=29, n_estimators=n_estimators, learning_rate=learning_rate)

        #   =========返回准确率=====================
        lst = cross_val_score(alg, X, y, cv=3, scoring='accuracy')
        precision = np.mean(lst)
        print(precision)
        dic = {'best_score': precision}
        print("AdaBootClassifier OK")
        return HttpResponse(json.dumps(dic))


def grid_search(req):
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(BASE_DIR, 'media')
    #   =========导入模块===========
    import json
    from sklearn.model_selection import GridSearchCV
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression

    file_path = path + r'\train_processed.csv'
    train = pd.read_csv(r'D:\train_processed.csv', dtype={"Age": np.float64})
    X = train.values[:, 1:]
    y = train.values[:, 0]

    alg_name = req.POST.get('alg', None)   # 判断使用哪个算法

    if alg_name == 'LR':
        LR = LogisticRegression(random_state=29)

        tol_start = req.POST.get('tol_start', None)
        tol_end = req.POST.get('tol_end', None)
        tol_interval = req.POST.get('tol_interval', None)

        C_start = req.POST.get('C_start', None)
        C_end = req.POST.get('C_start', None)
        C_interval = req.POST.get('C_start', None)

        parameters = {'tol': [1 / pow(10, i) for i in range(tol_start, tol_end, tol_interval)],
                      'C': [i / 10 for i in range(C_start, C_end, C_interval)],
                      'max_iter': range(5, 20, 5)}
        grid = GridSearchCV(LR, parameters, scoring='accuracy', cv=5)
        grid.fit(X, y)

        cvres = grid.cv_results_
        result = []
        for mean_train_score, params in zip(cvres['mean_train_score'], cvres['params']):
            result.append({'precision': mean_train_score, 'params': params})
            print(result)

        dic = {'table': result}
        return HttpResponse(json.dumps(dic))

    if alg_name == 'RF':
        begin_n_estimators = int(req.POST.get('begin_n_estimators', None))
        end_n_estimators = int(req.POST.get('end_n_estimators', None))
#        tol_interval = int(req.POST.get('tol_interval', None)))

        begin_max_depth = int(req.POST.get('begin_max_depth', None))
        end_max_depth = int(req.POST.get('end_max_depth', None))
#        C_interval = int(req.POST.get('C_start', None))

        begin_min_samples_split = int(req.POST.get('begin_min_samples_split', None))
        end_min_samples_split = int(req.POST.get('end_min_samples_split', None))
#         C_interval = int(req.POST.get('C_start', None))

        RF = RandomForestClassifier(random_state=29)
        parameters = {'n_estimators': range(begin_n_estimators, end_n_estimators, 50),
                      'max_depth': range(begin_max_depth, end_max_depth),
                      'min_samples_split': range(begin_min_samples_split, end_min_samples_split)}

#        parameters = {'n_estimators': range(10, 200, 50), 'max_depth': range(1, 5), 'min_samples_split': range(2, 3)}
        grid = GridSearchCV(RF, parameters, scoring='accuracy', cv=5)
        grid.fit(X, y)

        cvres = grid.cv_results_

        global grid_result
        grid_result = []
        for mean_train_score, params in zip(cvres['mean_train_score'], cvres['params']):
            grid_result.append({
                           'n_estimators': params['n_estimators'],
                           'max_depth': params['max_depth'],
                           'min_samples_split': params['min_samples_split'],
                           'precision': mean_train_score})
        print(grid_result)
        return HttpResponse(json.dumps(grid_result))

    if alg_name == 'DT':
        begin_max_depth = int(req.POST.get('begin_max_depth', None))
        end_max_depth = int(req.POST.get('end_max_depth', None))
        #        tol_interval = int(req.POST.get('tol_interval', None)))

        begin_min_samples_split = int(req.POST.get('begin_min_samples_split', None))
        end_min_samples_split = int(req.POST.get('end_min_samples_split', None))
        #        C_interval = int(req.POST.get('C_start', None))

        DT = DecisionTreeClassifier(random_state=29,)
        parameters = {'min_samples_split': range(begin_min_samples_split, end_min_samples_split),
                      'max_depth': range(begin_max_depth, end_max_depth)}

        grid = GridSearchCV(DT, parameters, scoring='accuracy', cv=5)

        grid.fit(X, y)

        cvres = grid.cv_results_
        result = []
        for mean_train_score, params in zip(cvres['mean_train_score'], cvres['params']):
            result.append({'precision': mean_train_score, 'params': params})
            print(result)

        dic = {'table': result}
        return HttpResponse(json.dumps(dic))

    if alg_name == 'KNN':
        begin_leaf_size = int(req.POST.get('begin_leaf_size', None))
        end_leaf_size = int(req.POST.get('end_leaf_size', None))
        #        tol_interval = int(req.POST.get('tol_interval', None)))

        begin_n_neighbors = int(req.POST.get('begin_n_neighbors', None))
        end_n_neighbors = int(req.POST.get('end_n_neighbors', None))
        #        C_interval = int(req.POST.get('C_start', None))

        KNN = KNeighborsClassifier(n_jobs=-1)
        parameters = {'leaf_size': range(begin_leaf_size, end_leaf_size),
                      'n_neighbors': range(begin_n_neighbors, end_n_neighbors)}

        grid = GridSearchCV(KNN, parameters, scoring='accuracy', cv=5)

        grid.fit(X, y)

        cvres = grid.cv_results_
        result = []
        for mean_train_score, params in zip(cvres['mean_train_score'], cvres['params']):
            result.append({'precision': mean_train_score, 'params': params})
            print(result)

        dic = {'table': result}
        return HttpResponse(json.dumps(dic))

    if alg_name == 'SVC':
        from decimal import Decimal
        import decimal

        begin_C = int(req.POST.get('begin_C', None))
        end_C = int(req.POST.get('end_C', None))
        #        tol_interval = int(req.POST.get('tol_interval', None)))

        with decimal.localcontext() as ctx:
            ctx.prec = 2
            begin_gamma = Decimal(req.POST.get('begin_gamma', None))
            end_gamma = Decimal(req.POST.get('end_gamma', None))
            #        C_interval = int(req.POST.get('C_start', None))
            print(begin_gamma, end_gamma)
            gamma_list = []
            interval = 0.02     # 此处表示间隔
            while Decimal(begin_gamma) <= Decimal(end_gamma):
                gamma_list.append(float(begin_gamma))
                begin_gamma += Decimal(interval)
            print(gamma_list)

        SVC = SVC(random_state=29)
        parameters = {'C': list(range(begin_C, end_C)),
                      'gamma': gamma_list}   # 暂时写死

        grid = GridSearchCV(SVC, parameters, scoring='accuracy', cv=5)
        grid.fit(X, y)

        cvres = grid.cv_results_
        result = []
        for mean_train_score, params in zip(cvres['mean_train_score'], cvres['params']):
            result.append({'precision': mean_train_score, 'params': params})
            print(result)

        dic = {'table': result}
        return HttpResponse(json.dumps(dic))

    if alg_name == 'AdaBootClassifier':
        begin_n_estimators = int(req.POST.get('begin_n_estimators', None))
        end_n_estimators = int(req.POST.get('end_n_estimators', None))
        #        tol_interval = int(req.POST.get('tol_interval', None)))

        # 设置 小数列表模块
        from decimal import Decimal
        import decimal
        with decimal.localcontext() as ctx:
            ctx.prec = 2
            begin_learning_rate = Decimal(req.POST.get('begin_learning_rate', None))
            end_learning_rate = Decimal(req.POST.get('end_learning_rate', None))
            #        C_interval = int(req.POST.get('C_start', None))
            print(begin_learning_rate, end_learning_rate)
            learning_rate_list = []
            interval = 0.2     # 此处表示间隔
            while Decimal(begin_learning_rate) <= Decimal(end_learning_rate):
                learning_rate_list.append(float(begin_learning_rate))
                begin_learning_rate += Decimal(interval)

        print(learning_rate_list)
        ABC = AdaBoostClassifier(random_state=29)
        parameters = {'n_estimators': list(range(begin_n_estimators, end_n_estimators, 10)),
                      'learning_rate': learning_rate_list}  # 小数不能用 range

        grid = GridSearchCV(ABC, parameters, scoring='accuracy', cv=5)
        grid.fit(X, y)

        cvres = grid.cv_results_
        result = []
        for mean_train_score, params in zip(cvres['mean_train_score'], cvres['params']):
            result.append({'precision': mean_train_score, 'params': params})
            print(result)

        dic = {'table': result}
        return HttpResponse(json.dumps(dic))


import json
def test(req):
    if req.method == "GET":
        return render(req, '随机森林模型.html')
    print("hello")
    list = [{'n_estimators': 1, 'max_depth': 2, 'min_samples_split': 3, 'score': '44'}]
    print(grid_result)
    return HttpResponse(json.dumps(list))








