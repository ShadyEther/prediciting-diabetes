

import streamlit as st
import pandas as pd
# from PIL import ImAge
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc


df = pd.read_csv("diabetes.csv")


st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())



x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)



def user_report():
  Pregnancies = st.sidebar.slider('Pregnancies', 0,17, 3 )
  Glucose = st.sidebar.slider('Glucose', 0,200, 120 )
  BloodPressure = st.sidebar.slider('Blood Pressure', 0,122, 70 )
  SkinThickness = st.sidebar.slider('Skin Thickness', 0,100, 20 )
  Insulin = st.sidebar.slider('Insulin', 0,846, 79 )
  BMI = st.sidebar.slider('BMI', 0,67, 20 )
  DiabetesPedigreeFunction = st.sidebar.slider('Diabetes Pedigree Function', 0.0,2.4, 0.47 )
  Age = st.sidebar.slider('Age', 21,88, 33 )

  user_report_data = {
      'Pregnancies':Pregnancies,
      'Glucose':Glucose,
      'BloodPressure':BloodPressure,
      'SkinThickness':SkinThickness,
      'Insulin':Insulin,
      'BMI':BMI,
      'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
      'Age':Age
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data




user_data = user_report()
# user_data.columns = user_data.columns.str.upper()
st.subheader('Patient Data')
st.write(user_data)




# # MODEL
# rf  = RandomForestClassifier()
# rf.fit(x_train, y_train)
# user_result = rf.predict(user_data)

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
user_result = knn.predict(user_data)




st.title('Visualised Patient Report')




if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'



st.header('Pregnancy count Graph (Others vs Yours)')
fig_preg = plt.figure()
ax1 = sns.scatterplot(x = 'Age', y = 'Pregnancies', data = df, hue = 'Outcome', palette = 'Greens')
ax2 = sns.scatterplot(x = user_data['Age'], y = user_data['Pregnancies'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_preg)




st.header('Glucose Value Graph (Others vs Yours)')
fig_Glucose = plt.figure()
ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = df, hue = 'Outcome' , palette='magma')
ax4 = sns.scatterplot(x = user_data['Age'], y = user_data['Glucose'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,220,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Glucose)



st.header('Blood Pressure Value Graph (Others vs Yours)')
fig_BloodPressure = plt.figure()
ax5 = sns.scatterplot(x = 'Age', y = 'BloodPressure', data = df, hue = 'Outcome', palette='Reds')
ax6 = sns.scatterplot(x = user_data['Age'], y = user_data['BloodPressure'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,130,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_BloodPressure)


st.header('Skin Thickness Value Graph (Others vs Yours)')
fig_st = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'SkinThickness', data = df, hue = 'Outcome', palette='Blues')
ax8 = sns.scatterplot(x = user_data['Age'], y = user_data['SkinThickness'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,110,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_st)



st.header('Insulin Value Graph (Others vs Yours)')
fig_i = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = df, hue = 'Outcome', palette='rocket')
ax10 = sns.scatterplot(x = user_data['Age'], y = user_data['Insulin'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,900,50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_i)



st.header('BMI Value Graph (Others vs Yours)')
fig_BMI = plt.figure()
ax11 = sns.scatterplot(x = 'Age', y = 'BMI', data = df, hue = 'Outcome', palette='rainbow')
ax12 = sns.scatterplot(x = user_data['Age'], y = user_data['BMI'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,70,5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_BMI)



st.header('DiabetesPedigreeFunction Value Graph (Others vs Yours)')
fig_DiabetesPedigreeFunction = plt.figure()
ax13 = sns.scatterplot(x = 'Age', y = 'DiabetesPedigreeFunction', data = df, hue = 'Outcome', palette='YlOrBr')
ax14 = sns.scatterplot(x = user_data['Age'], y = user_data['DiabetesPedigreeFunction'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,3,0.2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_DiabetesPedigreeFunction)




st.subheader('Your Report: ')
output=''
if user_result[0]==0:
  output = 'You are not Diabetic'
else:
  output = 'You are Diabetic'
st.title(output)
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, knn.predict(x_test))*100)+'%')



y_proba = knn.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)


roccurve=plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
st.pyplot(roccurve)

#draft accu
accuracy = accuracy_score(y_test, knn.predict(x_test))
st.write(str("Accuracy: {:.2f}%".format(accuracy * 100)))
