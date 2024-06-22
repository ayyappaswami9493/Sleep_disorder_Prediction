from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

app=Flask(__name__)
data=pd.read_csv("sleep_disorder.csv")
print(data.columns)
print(data.isna().sum())
print(data.info())

lab=LabelEncoder()
for i in data.select_dtypes(include='object').columns.values:
    data[i]=lab.fit_transform(data[i])

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

x_train,x_test,y_train,y_test=(train_test_split(x,y))

@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form
    gender = float(request.form['gender'])
    sleep_duration = float(request.form['sleep_duration'])
    quality_of_sleep = float(request.form['quality_of_sleep'])
    physical_activity_level = float(request.form['physical_activity_level'])
    daily_steps = float(request.form['daily_steps'])

    input_data = np.array([[gender, sleep_duration, quality_of_sleep,
                            physical_activity_level, daily_steps]])


    rf=RandomForestClassifier()
    rf.fit(x_train,y_train)
    prediction = rf.predict(input_data)

    return render_template('result.html', prediction=prediction[0])


if __name__ == '__main__':
    app.run(debug=True)
