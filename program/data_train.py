#!/usr/bin/env python
# coding: utf-8


#import logmodel as logmodel

import joblib
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
# In[2]:
from datetime import datetime

# datetime object containing current date and time
now = datetime.now()

print("now =", now)

data = pd.read_csv('data.csv', sep=",", encoding="ISO-8859-1")

# In[3]:


data.drop(['stn_code', 'agency', 'sampling_date', 'location_monitoring_station'], axis=1, inplace=True)

# In[4]:


total = data.isnull().sum().sort_values(ascending=False)

# In[5]:


percent = (data.isnull().sum() / data.isnull().count() * 100).sort_values(
    ascending=False)  # count(returns Non-NAN value)

# In[6]:


missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

# In[7]:


data.groupby('state')[['spm', 'pm2_5', 'rspm', 'so2', 'no2']].mean()

# In[8]:


grp_state = data.groupby('state')


# In[9]:


def impute_mean_by_state(series):
    return series.fillna(series.mean())


# In[10]:


data['rspm'] = grp_state['rspm'].transform(impute_mean_by_state)  # fill value with mean value group by state
data['so2'] = grp_state['so2'].transform(impute_mean_by_state)
data['no2'] = grp_state['no2'].transform(impute_mean_by_state)
data['spm'] = grp_state['spm'].transform(impute_mean_by_state)
data['pm2_5'] = grp_state['pm2_5'].transform(impute_mean_by_state)


# In[11]:


def cal_SOi(so2):
    si = 0
    if (so2 <= 40):
        si = so2 * (50 / 40)
    elif (so2 > 40 and so2 <= 80):
        si = 50 + (so2 - 40) * (50 / 40)
    elif (so2 > 80 and so2 <= 380):
        si = 100 + (so2 - 80) * (100 / 300)
    elif (so2 > 380 and so2 <= 800):
        si = 200 + (so2 - 380) * (100 / 420)
    elif (so2 > 800 and so2 <= 1600):
        si = 300 + (so2 - 800) * (100 / 800)
    elif (so2 > 1600):
        si = 400 + (so2 - 1600) * (100 / 800)
    return si


data['SOi'] = data['so2'].apply(cal_SOi)
df = data[['so2', 'SOi']]


# In[12]:


def cal_Noi(no2):
    ni = 0
    if (no2 <= 40):
        ni = no2 * 50 / 40
    elif (no2 > 40 and no2 <= 80):
        ni = 50 + (no2 - 40) * (50 / 40)
    elif (no2 > 80 and no2 <= 180):
        ni = 100 + (no2 - 80) * (100 / 100)
    elif (no2 > 180 and no2 <= 280):
        ni = 200 + (no2 - 180) * (100 / 100)
    elif (no2 > 280 and no2 <= 400):
        ni = 300 + (no2 - 280) * (100 / 120)
    else:
        ni = 400 + (no2 - 400) * (100 / 120)
    return ni


data['Noi'] = data['no2'].apply(cal_Noi)
df = data[['no2', 'Noi']]


# In[13]:


def cal_RSPMi(rspm):
    rpi = 0
    if (rspm <= 100):
        rpi = rspm
    elif (rspm >= 101 and rspm <= 150):
        rpi = 101 + (rspm - 101) * ((200 - 101) / (150 - 101))
    elif (rspm >= 151 and rspm <= 350):
        ni = 201 + (rspm - 151) * ((300 - 201) / (350 - 151))
    elif (rspm >= 351 and rspm <= 420):
        ni = 301 + (rspm - 351) * ((400 - 301) / (420 - 351))
    elif (rspm > 420):
        ni = 401 + (rspm - 420) * ((500 - 401) / (420 - 351))
    return rpi


data['RSPMi'] = data['rspm'].apply(cal_RSPMi)
df = data[['rspm', 'RSPMi']]


# In[14]:


def cal_SPMi(spm):
    spi = 0
    if (spm <= 50):
        spi = spm * 50 / 50
    elif (spm > 50 and spm <= 100):
        spi = 50 + (spm - 50) * (50 / 50)
    elif (spm > 100 and spm <= 250):
        spi = 100 + (spm - 100) * (100 / 150)
    elif (spm > 250 and spm <= 350):
        spi = 200 + (spm - 250) * (100 / 100)
    elif (spm > 350 and spm <= 430):
        spi = 300 + (spm - 350) * (100 / 80)
    else:
        spi = 400 + (spm - 430) * (100 / 430)
    return spi


data['SPMi'] = data['spm'].apply(cal_SPMi)
df = data[['spm', 'SPMi']]


# In[15]:


def cal_pmi(pm2_5):
    pmi = 0
    if (pm2_5 <= 50):
        pmi = pm2_5 * (50 / 50)
    elif (pm2_5 > 50 and pm2_5 <= 100):
        pmi = 50 + (pm2_5 - 50) * (50 / 50)
    elif (pm2_5 > 100 and pm2_5 <= 250):
        pmi = 100 + (pm2_5 - 100) * (100 / 150)
    elif (pm2_5 > 250 and pm2_5 <= 350):
        pmi = 200 + (pm2_5 - 250) * (100 / 100)
    elif (pm2_5 > 350 and pm2_5 <= 450):
        pmi = 300 + (pm2_5 - 350) * (100 / 100)
    else:
        pmi = 400 + (pm2_5 - 430) * (100 / 80)
    return pmi


data['PMi'] = data['pm2_5'].apply(cal_pmi)
df = data[['pm2_5', 'PMi']]


# In[16]:


def cal_aqi(si, ni, rspmi, spmi):
    aqi = 0
    if (si > ni and si > rspmi and si > spmi):
        aqi = si
    if (ni > si and ni > rspmi and ni > spmi):
        aqi = ni
    if (rspmi > si and rspmi > ni and rspmi > spmi):
        aqi = rspmi
    if (spmi > si and spmi > ni and spmi > rspmi):
        aqi = spmi
    return aqi


data['AQI'] = data.apply(lambda x: cal_aqi(x['SOi'], x['Noi'], x['RSPMi'], x['SPMi']), axis=1)
df = data[['state', 'SOi', 'Noi', 'RSPMi', 'SPMi', 'AQI']]


# In[17]:


def AQI_Range(x):
    if x <= 50:
        return "Good"
    elif x > 50 and x <= 100:
        return "Moderate"
    elif x > 100 and x <= 200:
        return "Poor"
    elif x > 200 and x <= 300:
        return "Unhealthy"
    elif x > 300 and x <= 400:
        return "Very unhealthy"
    elif x > 400:
        return "Hazardous"


data['AQI_Range'] = data['AQI'].apply(AQI_Range)

# In[18]:


data = data.dropna(subset=['spm'])

# In[19]:


data = data.dropna(subset=['pm2_5'])

# In[20]:


from sklearn.model_selection import train_test_split

# In[21]:


X1 = data[['so2', 'no2', 'rspm', 'spm']]
y1 = data['AQI']

# In[22]:


X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.33, random_state=101)

# In[ ]:


CX = data[['so2', 'no2', 'rspm', 'spm']]
Cy = data['AQI_Range']

# In[ ]:


CX_train, CX_test, Cy_train, Cy_test = train_test_split(CX, Cy, test_size=0.33, random_state=42)

# In[ ]:


#Ly_pred = logmodel.predict(CX_test)

# In[ ]:


#logmodel.score(CX_test, Cy_test)

# In[ ]:


# X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.33, random_state=101)

# LR = LinearRegression()
# LR.fit(X_train1, y_train1)
# joblib.dump(LR,'linear_regression.joblib')


################################SVR#################################


from sklearn.svm import SVR

# In[ ]:


SVModel = SVR()
SVModel.fit(X_train1, y_train1)
joblib.dump(SVModel,'svmodel.joblib')



y_pred2=SVModel.predict(X_test1)


# In[ ]:


SVModel.score(X_test1,y_test1)


# In[ ]:


print('R^2_Square:%.2f '% r2_score(y_test1, y_pred2))
print('MSE:%.2f ' % pd.np.sqrt(mean_squared_error(y_test1, y_pred2)))





##########################################Random Regresor###########################




# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


random_regressor=RandomForestRegressor(n_estimators=30,random_state=0)
random_regressor.fit(X_train1,y_train1)


# In[ ]:


y_pred4=random_regressor.predict(X_test1)


# In[ ]:


random_regressor.score(X_test1,y_test1)


# In[ ]:


print('R^2_Square:%.2f '% r2_score(y_test1, y_pred4))
print('MSE:%.2f ' % pd.np.sqrt(mean_squared_error(y_test1, y_pred4)))



joblib.dump(random_regressor,'randomregressor.jobib')


##############################################random classifier ######################################


# In[3]:


#Random forest classifiction
from sklearn.ensemble import RandomForestClassifier


# In[4]:


model = RandomForestClassifier(n_estimators=30)
model.fit(CX_train,Cy_train)

joblib.dump(model,'randomclassifier.joblib')

# In[ ]:


Ry_pred = model.predict(CX_test)


# In[ ]:


model.score(CX_test,Cy_test)


#####################################################LR1###################################


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# In[ ]:


LR1 = LinearRegression()
LR1.fit(X_train1, y_train1)
joblib.dump(LR1,'linear_regression.joblib')

# In[ ]:


print('Intercept',LR1.intercept_)
print('Coefficients',LR1.coef_)


# In[ ]:


y_pred = LR1.predict(X_test1)


# In[ ]:


LR1.score(X_test1,y_test1)


# In[ ]:


print('R^2_Square:%.2f '% r2_score(y_test1, y_pred))
print('MSE:%.2f ' % pd.np.sqrt(mean_squared_error(y_test1, y_pred)))


########################################Logistic#######################


# In[ ]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression


# In[ ]:


logisticmodel = LogisticRegression()
logisticmodel.fit(CX_train,Cy_train)

joblib.dump(logisticmodel,'logisticregression.joblib')
#from datetime import datetime

# datetime object containing current date and time
now = datetime.now()

print("now =", now)