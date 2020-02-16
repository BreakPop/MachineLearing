#导入日志库
import logging

#分类模型数据 回归模型数据  二分类模型评估
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.metrics import roc_auc_score

#矩阵随机划分为训练子集和测试子集，并返回训练集测试集样本和样本标签
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

#Classifier 分类器   regressor 回归 均方误差
from mla.ensemble.random_forest import RandomForestClassifier,RandomForestRegressor
from mla.metrics.metrics import mean_squared_error


logging.basicConfig(level=logging.DEBUG)


#分类函数
def classofication():
    X,y = make_classification(
        n_samples=500, n_features=10, n_informative=10, random_state=1111, n_classes=2, class_sep=2.5,n_redundant=0
        )
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=1111)

    model = RandomForestClassifier(n_estimators=10, max_depth=4)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)[:,1]
    #print(predictions)
    print("classification, roc auc score: %s" % roc_auc_score(y_test,predictions))
    

#回归函数
def regression():
    X,y = make_regression(
        n_samples=500, n_features=5, n_informative=5, n_targets=1, noise=0.05, random_state=1111,bias=0.5
        )
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state=1111)
    model = RandomForestRegressor(n_estimators=50, max_depth=10, max_features=3)
    model.fit(X_train,y_train)
    predictions = model.predict(X_train, y_train)
    #print(predictions)
    print("regressor, mse: %s" % mean_squared_error(y_test.flatten(), predictions.flatten()))


#机器学习之分类性能度量指标:ROC曲线、AUC值、正确率、召回率
#机器学习之回归性能度量指标:MSE、RMSE、MAE、R-Squared

if __name__ == "__main__":
    classification()
    # regression()
    
