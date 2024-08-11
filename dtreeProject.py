



import pandas as pd
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor
#import modules
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

import plotly.express as px

df = pd.read_csv("C:\\Users\\Amit\\Downloads\\dtreeProjectFolder\\melb_data.csv")
filtered = df.dropna(axis=0)
filtered = filtered.drop(columns=['num', 'Suburb', 'Address', 'Date', 'CouncilArea', 'Type', 'Method', 'Regionname', 'Distance', 'Postcode', 'Propertycount'])






#density heatmap
melbMap = filtered[['Lattitude', 'Longtitude', 'Price']]
fig = px.density_mapbox(melbMap, lat = 'Lattitude', lon = 'Longtitude',
                        z = 'Price', radius = 10, title = "Housing Prices", mapbox_style = "carto-positron")
#fig.show()



targetdf = filtered["Price"]
datadf = filtered[['Rooms', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude', 'TypeNum', 'MethodNum']]
X = datadf
y = targetdf
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#initial decision tree
clf = DecisionTreeRegressor()
clf.fit(X_train, y_train)
initialy_predict = clf.predict(X_test)
r2 = r2_score(y_test, initialy_predict)
print(r2)





#hyperparameter optimized decisiontree
clf = DecisionTreeRegressor(max_depth=8, criterion='friedman_mse')
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
r2 = r2_score(y_test, y_predict)
print(r2)

#import modules
import matplotlib.pyplot as plt
import pandas as pd

#feature imporatnces
def sortSecond(val):
    return val[1]
values = clf.feature_importances_
features = list(X)
importances = [[features[i], values[i]] for i in range(len(features))]
importances.sort(reverse=False, key=sortSecond)
fix, ax = plt.subplots(figsize = (16,9))
for i in importances:
    ax.barh(i[0], i[1])







X_train=X_train[[col[0] for col in importances[5:]]]
X_test=X_test[[col[0] for col in importances[5:]]]
cut_clf = DecisionTreeRegressor(max_depth=8, criterion='friedman_mse')
cut_clf.fit(X_train, y_train)
cut_predict = cut_clf.predict(X_test)
r2 = r2_score(y_test, cut_predict)
print(r2)



#ada boost
adaBoost = AdaBoostRegressor(cut_clf)
adaBoost.fit(X_train, y_train)
ada_predict = adaBoost.predict(X_test)
r2 = r2_score(y_test, ada_predict)
print(r2)

#xgboost
model = xgb.XGBRegressor()
model.fit(X_train, y_train)
xgb_predict = model.predict(X_test)
r2 = r2_score(y_test, xgb_predict)
print(r2)




plt.figure()
plt.plot(y_test, y_test, label="Actual Prices")
plt.xlabel("actual prices")
plt.ylabel("predicted prices")
plt.scatter(y_test, cut_predict, label="DecisionTreeRegressor", color="black")
plt.scatter(y_test, ada_predict, label="AdaBoostRegressor", color="purple")
plt.scatter(y_test, xgb_predict, label="XGBoostRegressor", color="orange")
plt.legend(fontsize="10")
plt.show()


