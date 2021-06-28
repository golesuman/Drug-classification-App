import streamlit as st
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
df=pd.read_csv('drug200.csv')
df.head()

df.describe()

st.write('''
# Drug Classifier
## This web app classifies what kind of drug a patient needs

''')
st.sidebar.header("User Input")
def user_input():
    Age=st.sidebar.slider("Age",15,25,80)
    Sex=st.sidebar.selectbox("Gender",("M","F"))
    BP=st.sidebar.selectbox("Blood Pressure",("HIGH","LOW", "NORMAL"))
    Cholesterol=st.sidebar.selectbox("Cholesterol",("HIGH","NORMAL"))
    Na_to_K=st.sidebar.slider("SodiumtoPotassium",0.0,10.5,40.5)
    data={
        'Age':Age,
        'Sex':Sex,
        'BP':BP,
        'Cholesterol':Cholesterol,
        'Na_to_K':Na_to_K,
    }
    drug=pd.DataFrame(data,index=[0])
    return drug
df_drug=user_input()
st.subheader("User Input Parameters")
st.write(df_drug)
df.info()

features_category=[feature for feature in df.columns if df[feature].dtypes=='O']
# features_category
# for feature in features_category:
#     sns.countplot(feature,data=df)
#     plt.show()

num_feature=[feature for feature in df.columns if df[feature].dtypes!='O']
# num_feature

for feature in num_feature:
    print(f'{feature}:{len(df[feature].unique())}')
    
# ## Since both of the given features are continuous variable we use histogram to see the property of this features
# for feature in num_feature:
#     sns.kdeplot(df[feature])
#     plt.show()
# ## Since the data is normally distributed we now do feature engineering 
# features_category
from sklearn.preprocessing import LabelEncoder
enco=LabelEncoder()
for feature in features_category:
    df[feature]=enco.fit_transform(df[feature])

for feature in features_category:
    if feature!='Drug':
        df_drug[feature]=enco.fit_transform(df_drug[feature])

# df.head()
X=df.drop('Drug',axis=1)
y=df['Drug']
from pyforest import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold,cross_val_score
models=[RandomForestClassifier(),DecisionTreeClassifier(),SVC()]
for model in models:
    print(f'{model}:{cross_val_score(model,X,y,cv=5)}')
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=3)
model=DecisionTreeClassifier()
from sklearn.pipeline import Pipeline


model.fit(x_train,y_train)
model.score(x_test,y_test)
pre=model.predict(df_drug)
st.subheader("You need")
if pre==0:
    st.write('DrugA')

elif pre==1:
    st.write('DrugB')
elif pre==2:
    st.write('DrugC')
elif pre==3:
    st.write('DrugX')
else:
    st.write('DrugY')
