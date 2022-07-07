import inline as inline
import matplotlib
import pandas as pd
import numpy as np 
import seaborn as sb 
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

data=pd.read_csv("Stores.csv")
print(data)

print(data.shape)

print(data.describe())

print(data.dtypes)

print(data.info)

print(data.isnull().sum())

print(data.columns)

sb.heatmap(data.corr(), annot = True)
plt.show()

features = ['Store_Area', 'Items_Available', 'Daily_Customer_Count','Store_Sales']
fig, axs = plt.subplots(2, 2, figsize=(8,10))
fig.tight_layout(pad=4.0)

for f,ax in zip(features,axs.ravel()):
    ax=sb.boxplot(ax=ax,data=data,y=data[f])
    ax.set_title('Feature:'+ f)

plt.show()

x = data.iloc[:,0:-1].to_numpy()
y = data.iloc[:,-1].to_numpy()

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor()
reg_list = [rf_reg]
for reg in reg_list:
    reg.fit(x, y)

    y_pred = reg.predict(x)

    mse = mean_squared_error(y_pred, y)
    rmse = np.sqrt(mean_squared_error(y_pred, y))
    mae = mean_absolute_error(y_pred, y)
    score = reg.score(x, y)

    print('Regressor:{}\nMSE:{:.2f}\nRMSE:{:.2f}\nMAE:{:.2f}\nScore:{:.4f}\n\n'.format(str(reg), mse, rmse, mae, score))
