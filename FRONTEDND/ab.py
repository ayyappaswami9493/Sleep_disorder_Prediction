import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

data=pd.read_csv("sleep_disorder.csv")
print(data.columns)
print(data.isna().sum())
print(data.info())

data['Sleep Disorder']=data['Sleep Disorder'].replace('NaN',0)

print(data['Gender'].unique())
print(data['Sleep Disorder'].value_counts().values)
print(data['Sleep Disorder'].unique)

lab=LabelEncoder()
for i in data.select_dtypes(include='object').columns.values:
    data[i]=lab.fit_transform(data[i])

print(data['Gender'].value_counts())
print(data['Sleep Disorder'].value_counts().values)

x=[]
corr=data.corr()['Sleep Disorder']
corr=corr.drop(['Sleep Disorder'])
for i in corr.index:
    if corr[i]>0:
        x.append(i)
x=data[x]
y=data['Sleep Disorder']
smote=SMOTE()
x,y=smote.fit_resample(x,y)

print(x.columns)

x_train,x_test,y_train,y_test=train_test_split(x,y)

print(x_test.values[0])
print(y_test.values[0])

xgb=XGBClassifier()
xgb.fit(x_train,y_train)
print(xgb.score(x_test,y_test))

rf=RandomForestClassifier(criterion='gini',max_depth=9)
rf.fit(x_train,y_train)
print(rf.score(x_test,y_test))