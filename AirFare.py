#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split


df=pd.read_excel('D:/python/Datasets/Flight Data/Data_Train.xlsx')
df.head()
df.info()
df.dropna(inplace=True)
b=df.isnull().sum()
print(b)

df['Journey Date hour']=pd.to_datetime(df['Date_of_Journey'],format='%d/%m/%Y').dt.day #converting datatype into int from object
df['Journey Date month']=pd.to_datetime(df['Date_of_Journey'],format='%d/%m/%Y').dt.month

df.drop('Date_of_Journey',inplace=True,axis=1)

df['dep_hour']=pd.to_datetime(df['Dep_Time']).dt.hour
df['dep_minute']=pd.to_datetime(df['Dep_Time']).dt.minute
df.drop('Dep_Time',inplace=True,axis=1)

df['arr_hour']=pd.to_datetime(df['Arrival_Time']).dt.hour
df['arr_minute']=pd.to_datetime(df['Arrival_Time']).dt.minute
df.drop('Arrival_Time',inplace=True,axis=1)

df['Duration'].value_counts()
df['duration time']=  df['Duration'].str.replace("h", '*60').str.replace(' ','+').str.replace('m','*1').apply(eval)
df.drop(['Duration'],inplace=True,axis=1)

fig=df['Price'].hist(bins=50)
fig.set_title('price hist')
fig.set_xlabel('Price Range')
fig.set_ylabel('Count')

df['Airline'].value_counts()
sns.countplot('Airline',data=df)
plt.figure(figsize=(20,6))
sns.catplot('Airline','Price',data=df)
plt.show()
airline=pd.get_dummies(df['Airline'],drop_first=True)

df['Source'].value_counts()
sns.catplot('Source','Price',data=df,kind='strip')
source=pd.get_dummies(df['Source'],drop_first=True)

df['Destination'].value_counts()
sns.catplot('Destination','Price',data=df)
destination=pd.get_dummies(df['Destination'],drop_first=True)

df['Total_Stops'].value_counts()
df['stops']=df['Total_Stops'].replace({'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4})

df=pd.concat([df,airline,source,destination],axis=1)
df.head()

df.drop(['Airline','Source','Destination','Route','Total_Stops','Additional_Info'],axis=1,inplace=True)

x=df.drop('Price',axis=1)
y=df['Price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

model=ExtraTreesRegressor()
model.fit(x,y)

imp_features=pd.DataFrame(model.feature_importances_,index=x.columns)
imp_features
imp_features.nlargest(25,columns=imp_features.columns).plot(kind='bar')


from sklearn.model_selection import RandomizedSearchCV

max_depth=[3,4,7,11,15,20]
n_estimators=[int(x) for x in np.linspace(100,1000,10)]
min_samples_leaf=[2,5,10,20]
min_samples_split=[2,5,10,15,100]

params={'max_depth':max_depth,'n_estimators':n_estimators,'min_samples_leaf':min_samples_leaf,
        'min_samples_split':min_samples_split}

tuned_model=RandomizedSearchCV(RandomForestRegressor(),params,n_iter=15,n_jobs=1,cv=10,verbose=2,
                               scoring='neg_mean_squared_error')
tuned_model.fit(x_train,y_train)
tuned_model.best_params_
tuned_model.best_estimator_

#applying best estimators to RF Regressor model

new_model=RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                      max_depth=11, max_features='auto', max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=2,
                      min_samples_split=10, min_weight_fraction_leaf=0.0,
                      n_estimators=900, n_jobs=None, oob_score=False,
                      random_state=None, verbose=0, warm_start=False)
new_model.fit(x_train,y_train)
pred=new_model.predict(x_test)

score=new_model.score(x_train,y_train)
score1=new_model.score(x_test,y_test)
print(score,score1)

sns.distplot(y_test-pred)
sns.scatterplot(y_test,pred)
plt.xlabel('y_test')
plt.ylabel('predicted')

print('Root MSE',np.sqrt(mean_squared_error(y_test,pred)))
print('MSE',mean_squared_error(y_test,pred))

comparison=pd.DataFrame({'Original':y_test,'Predicted':pred})

file=open('Flight-Predictor.pkl','wb')
pickle.dump(new_model,file)


# In[ ]:




