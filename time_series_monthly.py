# imports
import pymysql
import pymysql.cursors
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

# connection to db
connection = pymysql.connect(host = '192.168.1.91',
                             user = 'root',
                             password = 'mysql',
                             db = 'Retail_database',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

# fetching tables from db
query = 'SELECT * from Time_Series_Data;'
df = pd.read_sql(query, connection)
# df_temp = pd.read_sql(query, connection)

# datetime manipulation
df['Date'] = pd.to_datetime(df['Date'])
# df_temp['Date'] = pd.to_datetime(df_temp['Date'])

# fetching current year and fetching data before the current year
now = datetime.datetime.now()
now = now.year
df = df.loc[df['Date'].dt.year<now]

df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

df1 = df.groupby(['Year', 'Month'], as_index=False).sum()
df1['Month_Year'] = df1['Month'].astype(str)+'-'+df1['Year'].astype(str)
df1 = df1[['Month_Year','GrandTotal']]

df1 = df1.set_index(df1.Month_Year, drop=True, append=False)
df1.drop(columns=['Month_Year'],axis=1,inplace = True)

# ARIMA model
model = ARIMA(df1, order=(0,1,1))
model_fit = model.fit(disp=0)
print(model_fit.summary())

# plotting residuals
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

model_fit.plot_predict(dynamic=False)
plt.show()

model_fit.predict()