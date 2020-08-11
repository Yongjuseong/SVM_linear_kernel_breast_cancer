import sklearn.datasets as data # breast_cancer 데이터 셋 = > 유방암
import pandas as pd
import sklearn.svm as svm # support vertor machine
import sklearn.metrics as metric
from sklearn.model_selection import cross_val_score, cross_validate
import sklearn.model_selection as ms
from sklearn.preprocessing import StandardScaler

x= data.load_breast_cancer()
cancer = pd.DataFrame(data=x.data,columns=x.feature_names)
cancer['target'] = x.target

cancer.info()
cancer.describe()
cancer.target.value_counts()

# SVM, kernel=> 선형분리.  linear kernel classification
svm_clf = svm.SVC(kernel='linear')

#Cross validation
scores = pd.DataFrame(cross_val_score(svm_clf,x.data,x.target,cv=5))
print('The Average of Cross Validation by using linear kernel SVM:',scores.mean())

#scaling -StandardScaling  aveerage = 0, standard deviation = 1 평균 0, 표준 편차 1
X=cancer.iloc[:,:-1]
y=cancer.iloc[:,-1]

scaler=StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# Dividing transfomred data into data / 변환된 데이터 x를 분할
X_train,X_test,y_train,y_test = ms.train_test_split(X_scaled,y,test_size=0.3,random_state=1)

#SVM(using standard scaler) linear classification / 스케일러 사용 svm 선형분리
sscaler_svm_clf=svm.SVC(kernel='linear',random_state=100)
scores2=pd.DataFrame(cross_val_score(sscaler_svm_clf,X_scaled,y,cv=5))

print('The Average of Cross Validation by using Linear kernel SVM with StandardScaler:',scores2.mean())

# Tuning SVM parameter by using GridSearchCV / svm파라미터 튜닝 => GridSearch CV 활용

# Define values to be tested as dictionary types /테스트 하고자하는 파라미터 값들을 사전타입으로 정의

GS_svm_clf = svm.SVC(kernel='linear',random_state=100) #random_state = the seed value when making a numver for random / 숫자 생성시 seed 값
parameters = {'C': [0.001,0.01,0.1,1,10,225,50,100]}
grid_svm=ms.GridSearchCV(GS_svm_clf,param_grid=parameters,cv=5)
grid_svm.fit(X_train,y_train)

result=pd.DataFrame(grid_svm.cv_results_['params'])
result['mean_test_score'] = grid_svm.cv_results_['mean_test_score']
result.sort_values(by='mean_test_score',ascending=False)
print(result)