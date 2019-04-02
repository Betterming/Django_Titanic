import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
import sklearn.preprocessing as preprocessing
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingRegressor
from PIL import Image
import imageio


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
data_train = pd.read_csv(r'..\static\media\train.csv')
data_test = pd.read_csv(r'..\static\media\test.csv')


def fig_demo():  # 直方图生成
    fig = plt.figure()
    fig.set_size_inches(12, 12)  # 设置画布尺寸

    plt.subplot2grid((2, 2), (0, 0))
    data_train.Survived.value_counts().plot(kind='bar')
    plt.title(u"获救情况 (1为获救)")
    plt.ylabel(u"人数")

    plt.subplot2grid((2, 2), (0, 1))
    data_train.Pclass.value_counts().plot(kind="bar")
    plt.ylabel(u"人数")
    plt.title(u"乘客等级分布")

    plt.subplot2grid((2, 2), (1, 0))
    data_train.Embarked.value_counts().plot(kind='bar')
    plt.title(u"各登船口岸上船人数")
    plt.ylabel(u"人数")

    plt.subplot2grid((2, 2), (1, 1))
    data_train.Age[data_train.Pclass == 1].plot(kind='kde')
    data_train.Age[data_train.Pclass == 2].plot(kind='kde')
    data_train.Age[data_train.Pclass == 3].plot(kind='kde')
    plt.xlabel(u"年龄")  # plots an axis lable
    plt.ylabel(u"密度")
    plt.title(u"各等级的乘客年龄分布")
    plt.legend((u'头等舱', u'2等舱', u'3等舱'), loc='best')
    plt.savefig(r'..\static\media\relation.png')
    plt.show()


def age_demo():  # 年龄与生存率关系
    fig = plt.figure()
    fig.set_size_inches(12, 12)  # 设置画布尺寸

    plt.subplot2grid((2, 2), (0, 0))
    plt.scatter(data_train.Survived, data_train.Age)
    plt.ylabel(u"年龄")  # 设定纵坐标名称
    plt.grid(b=True, which='major', axis='y')
    plt.title(u"按年龄看获救分布 (1为获救)")
    plt.savefig(r'..\static\media\age_relation.png')
    plt.show()


def pclass_demo():
    # 各舱位等级的获救情况
    fig = plt.figure()

    Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
    Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
    df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
    df.plot(kind='bar', stacked=True)
    plt.title(u"各乘客等级的获救情况")
    plt.xlabel(u"乘客等级")
    plt.ylabel(u"人数")
    plt.savefig(r'..\static\media\pclass_relation.png')
    plt.show()


def sex_demo():
    # 看看各性别的获救情况

    fig = plt.figure()

    Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
    Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
    df = pd.DataFrame({u'男性': Survived_m, u'女性': Survived_f})
    df.plot(kind='bar', stacked=True)

    plt.title(u"按性别看获救情况")
    plt.xlabel(u"性别")
    plt.ylabel(u"人数")
    plt.savefig(r'..\static\media\sex_relation.png')
    plt.show()


def sex_pclass():
    # 各种舱级别情况下各性别的获救情况
    fig = plt.figure()
    fig.set_size_inches(14, 7)
    plt.title(u"根据舱等级和性别的获救情况")

    ax1 = fig.add_subplot(141)
    data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar',
                                                                                                label="female highclass",
                                                                                                color='#FA2479')
    ax1.set_xticklabels([u"获救", u"未获救"], rotation=45)
    ax1.legend([u"女性/高级舱"], loc='best')

    ax2 = fig.add_subplot(142, sharey=ax1)
    data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar',
                                                                                                label='female, low class',
                                                                                                color='pink')
    ax2.set_xticklabels([u"未获救", u"获救"], rotation=45)
    plt.legend([u"女性/低级舱"], loc='best')

    ax3 = fig.add_subplot(143, sharey=ax1)
    data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar',
                                                                                              label='male, high class',
                                                                                              color='lightblue')
    ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
    plt.legend([u"男性/高级舱"], loc='best')

    ax4 = fig.add_subplot(144, sharey=ax1)
    data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar',
                                                                                              label='male low class',
                                                                                              color='steelblue')
    ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
    plt.legend([u"男性/低级舱"], loc='best')
    plt.savefig(r'..\static\media\sex_pclass.png')
    plt.show()


def embarked():
    #登陆港口和获救关系

    fig = plt.figure()

    Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
    Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
    df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
    df.plot(kind='bar', stacked=True)

    plt.title(u"各登录港口乘客的获救情况")
    plt.xlabel(u"登录港口")
    plt.ylabel(u"人数")
    plt.savefig(r'..\static\media\embarked_demo.png')
    plt.show()


def sibsp_demo():
    # （堂）兄弟姐妹与父母子女数量对获救的影响
    g = data_train.groupby(['SibSp', 'Survived'])
    df = pd.DataFrame(g.count()['PassengerId'])
    print(df)

    g = data_train.groupby(['Parch', 'Survived'])
    df = pd.DataFrame(g.count()['PassengerId'])
    df.to_csv(r'..\static\media\sibsp.csv')
    print(df)


def cabin():
    # 船舱信息对获救的影响
    # ticket是船票编号，应该是unique的，和最后的结果没有太大的关系，先不纳入考虑的特征范畴
    # cabin只有204个乘客有值，我们先看看它的一个分布

    count = data_train.Cabin.value_counts()
    count.to_csv(r'..\static\media\cabin.csv')


# 注意，若第二次运行本程序，会报"ValueError: Found array with 0 sample(s) (shape=(0, 4)) while a minimum of 1 is required."，
# 这是因为在上次运行本段程序时，data_train已经发生了变化
# 解决方案：不要连续运行本程序，在再次运行本程序之前，要先运行上面第一段程序，以获得原data_train的值
### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages():
    df = data_train
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1:])


    # 用得到的预测结果填补原缺失数据
    df.loc[df.Age.isnull(), 'Age'] = predictedAges

    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df


def feature_factorization(df):
    dummies_Cabin = pd.get_dummies(df['Cabin'], prefix='Cabin')

    dummies_Embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')

    dummies_Sex = pd.get_dummies(df['Sex'], prefix='Sex')

    dummies_Pclass = pd.get_dummies(df['Pclass'], prefix='Pclass')

    df = pd.concat([df, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

    scaler = preprocessing.StandardScaler()
    age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))
    df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1), age_scale_param)
    fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1, 1))
    df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1, 1), fare_scale_param)

    df.to_csv(r'..\static\media\feature_factorization.csv')

    return df, scaler, age_scale_param, fare_scale_param


def pre_processed(df, dt, rfr, scaler, age_scale_param, fare_scale_param):
    # 用正则取出我们要的属性值
    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.as_matrix()

    # y即Survival结果
    y = train_np[:, 0]

    # X即特征属性值
    X = train_np[:, 1:]

    # fit到LogisticRegression之中
    clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(X, y)

    dt.loc[(dt.Fare.isnull()), 'Fare'] = 0

    # 接着我们对test_data做和train_data中一致的特征变换
    # 首先用同样的RandomForestRegressor模型填上丢失的年龄
    tmp_df = dt[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    null_age = tmp_df[dt.Age.isnull()].as_matrix()
    # 根据特征属性X预测年龄并补上
    X = null_age[:, 1:]
    predictedAges = rfr.predict(X)
    dt.loc[(dt.Age.isnull()), 'Age'] = predictedAges

    data_test = set_Cabin_type(dt)
    dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')

    df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1, 1), age_scale_param)
    df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1, 1), fare_scale_param)

    df_test.to_csv(r'..\static\media\df_test.csv')
    return train_df, df_test, clf, X, y


def prediction(df_test, clf):
    test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictions = clf.predict(test)
    result = pd.DataFrame(
        {'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
    result.to_csv(r'..\static\media\predicted_result.csv', index=False)


# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
def plot_learning_curve(i,j, estimator, title, X, y, train_sizes, ylim=None, cv=None, n_jobs=1,  verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        plt.xlim((0, 600))
        plt.ylim((0.65, 1))
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")
        print(i, train_sizes)

        plt.legend(loc="best")

        plt.draw()
        plt.gca().invert_yaxis()
        picture_name = r'..\static\media\learn_curve\lc'+str(j) +r'\learning_curve' + str(i) + r'.png'
        plt.savefig(picture_name)
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])

    return midpoint, diff


def bagging(df, df_test):

    train_df = df.filter(
        regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
    train_np = train_df.as_matrix()

    # y即Survival结果
    y = train_np[:, 0]

    # X即特征属性值
    X = train_np[:, 1:]

    # fit到BaggingRegressor之中
    clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True,
                                   bootstrap_features=False, n_jobs=-1)
    bagging_clf.fit(X, y)

    test = df_test.filter(
        regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
    predictions = bagging_clf.predict(test)
    result = pd.DataFrame(
        {'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
    result.to_csv("predicted_bagging_result.csv", index=False)


def create_img(j, alg, X, y):
    ts = []
    for i, train in enumerate(np.linspace(.05, 1., 20)):
        ts.append(train)
        train_sizes = np.array(ts)
        print(train_sizes)

        midpoint, diff = plot_learning_curve(i, j, alg, u"learn_curve", X, y, train_sizes)

    return midpoint, diff


def img2gif(i):  # i为对应模型
    outfilename = r'..\static\media\learning_curve' + str(i) +r'.gif'  # 转化的GIF图片名称
    filenames = []
    for item in range(20):
        filename = r'..\static\media\learn_curve\lc' + str(i) + r'\learning_curve' + str(item) + r'.png'
        filenames.append(filename)
    frames = []
    for image_name in filenames:
        im = Image.open(image_name)  # 读取方式上存在略微区别，由于是直接读取数据，并不需要后续处理
        frames.append(im)
    imageio.mimsave(outfilename, frames, 'GIF', duration=0.1)  # 生成方式也差不多


def main():

    # fig_demo()
    # age_demo()
    # sex_demo()
    # sex_pclass()
    # embarked()
    # print(data_train)
    # sibsp_demo()
    # cabin()

    data_train, rfr = set_missing_ages()
    data_train = set_Cabin_type(data_train)
    # data_train.to_csv(r'..\static\media\set_missing_ages.csv')
    df, scaler, age_scale_param, fare_scale_param = feature_factorization(data_train)
    train_df, df_test, clf, X, y = pre_processed(df, data_test, rfr, scaler, age_scale_param, fare_scale_param)
    prediction(df_test, clf)

    connect = pd.DataFrame({"columns": list(train_df.columns)[1:], "coef": list(clf.coef_.T)})
    connect.to_csv(r'..\static\media\connect.csv')

    all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    X = all_data.as_matrix()[:, 1:]
    y = all_data.as_matrix()[:, 0]

    # 分割数据，按照 训练数据:cv数据 = 7:3的比例
    split_train, split_cv = cross_validation.train_test_split(df, test_size = 0.3, random_state = 0)
    train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    # 生成模型
    clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(train_df.as_matrix()[:, 1:], train_df.as_matrix()[:, 0])
    # svm = SVC()
    # 对cross validation数据进行预测

    cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictions = clf.predict(cv_df.as_matrix()[:, 1:])

    origin_data_train = pd.read_csv(r'..\static\media\train.csv')

    bad_cases = origin_data_train.loc[
        origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.as_matrix()[:, 0]]['PassengerId'].values)]
    bad_cases.to_csv(r'..\static\media\bad_cases.csv')

    # plot_learning_curve(1, clf, u"learning_curve",X, y, train_sizes=np.linspace(0.05, 1, 20))

    create_img(0, clf, X, y)  # 创建图片在lc0文件夹下
    img2gif(0)  # 生成动态图, 0表示第一个模型的动态图
    # bagging(df, df_test)


if __name__ == '__main__':
    main()
