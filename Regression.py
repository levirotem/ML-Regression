
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 15:13:23 2021

@author: levir
"""

# Importing the libraries
import  numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
import  statsmodels.api as  sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.pipeline import Pipeline


warnings.filterwarnings('ignore')

dataset = pd.read_csv('COVID-19_Daily_Testing_-_By_Test.csv')
X = dataset[dataset.columns[5:23]]

y = dataset.iloc[:, 2].values
y2 = dataset.iloc[:, 3].values

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state = 1)
sc = StandardScaler()

#Using Pearson Correlation
#plt.figure(figsize=(12,10))
#Create correlation matrix
corr_matrix = X.corr().abs()
#sns.heatmap(corr_matrix, annot=True, cmap=plt.cm.Reds)
#plt.show()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X.drop(to_drop, axis=1, inplace=True)


X = np.array(X, dtype=float)


def metrix(X,y):
    lean_reg_mae,lean_reg_mse,lean_reg_rmse,lean_reg_r2 =lean_reg(X,y)
    poli_deg_mae,poli_deg_mse,poli_deg_rmse,poli_deg_r2,poli_deg_results,poli_deg_find,X_test, y_test= poli_deg(X,y,'neg_mean_squared_error')
    ridge_reg_mae,ridge_reg_mse,ridge_reg_rmse,ridge_reg_deg_r2,ridge_reg_results,ridge_reg_find= ridge_regression(X,y,'neg_mean_squared_error')
    lasso_reg_mae,lasso_reg_mse,lasso_reg_rmse,lasso_reg_deg_r2,lasso_reg_results,lasso_reg_find= lasso_regression(X,y,'neg_mean_squared_error')
    random_forect_mae,random_forect_mse,random_forect_rmse,random_forect_r2,random_forect_results,random_forect_find= random_forect(X,y,'neg_mean_squared_error')
    KNN_mae,KNN_mse,KNN_rmse,KNN_r2,KNN_results,KNN_find= KNN(X,y,'neg_mean_squared_error')
    
    
    data = {'paramter':['noparametr', poli_deg_find, ridge_reg_find, lasso_reg_find,random_forect_find,KNN_find ],
        'neg_mean_squared_error': ['noparametr',poli_deg_results.best_score_,ridge_reg_results.best_score_,lasso_reg_results.best_score_,random_forect_results.best_score_,KNN_results.best_score_],
        'the best':['noparametr',poli_deg_results.best_params_[poli_deg_find], ridge_reg_results.best_params_[ridge_reg_find], lasso_reg_results.best_params_[lasso_reg_find], random_forect_results.best_params_[random_forect_find],KNN_results.best_params_[KNN_find]], 
        'mae': [lean_reg_mae,poli_deg_mae ,ridge_reg_mae,lasso_reg_mae,random_forect_mae,KNN_mae],
        'mse': [lean_reg_mse,poli_deg_mse,ridge_reg_mse,lasso_reg_mse,random_forect_mse,KNN_mse],
        'rmse': [lean_reg_rmse,poli_deg_rmse,ridge_reg_rmse,lasso_reg_rmse,random_forect_rmse,KNN_rmse],
        'r2': [lean_reg_r2,poli_deg_r2,ridge_reg_deg_r2,lasso_reg_deg_r2,random_forect_r2,KNN_r2]}
    
    df = pd.DataFrame(data, index =['Linear Regression', 'Polynomial Regression', 'ridge Regression', 'lasso Regression','random forect','KNN' ])  
    return df

    
def evaluation ( model, X_test, y_test):
    y_pred= model.predict(X_test)
    mae= mean_absolute_error(y_test,y_pred)
    mse= mean_squared_error(y_test,y_pred)
    rmse= np.sqrt(mse)
    r2=  r2_score(y_test, y_pred)

    return mae,mse,rmse,r2

def backward_elimination(x,y_dependent,sl):
    var=np.arange(x.shape[1])
    x_ols_array=x[:,var]
    regressor=sm.OLS(y_dependent,x_ols_array).fit()
    for i in range(sum(regressor.pvalues>0)):
        if sum(regressor.pvalues>=sl)>0:
            arg=regressor.pvalues.argmax()
            var=np.delete(var,arg)
            x_ols_array=x[:,var]
            regressor=sm.OLS(y_dependent,x_ols_array).fit()
    return (x_ols_array)


def lean_reg(t_X,t_y):
   t_X= sc.fit_transform(t_X) 
   t_X=np.append(arr=np.ones((635,1)).astype(int),values=t_X ,axis=1)
   t_X=backward_elimination(t_X,t_y,0.05)
   X_train, X_test, y_train, y_test = train_test_split(t_X, t_y, test_size = 0.3,random_state = 1)
   regressor = LinearRegression()
   model=regressor.fit(X_train, y_train)
   mae,mse,rmse,r2= evaluation (model, X_test, y_test)
   return(mae,mse,rmse,r2)


def poli_deg(t_X,t_y,score):
    t_X= sc.fit_transform(t_X) 
    t_X=np.append(arr=np.ones((635,1)).astype(int),values=t_X ,axis=1)#בחרתי 635 שורות לפי השורות של הדאטה
    t_X=backward_elimination(t_X,t_y,0.05)
    X_train, X_test, y_train, y_test = train_test_split(t_X, t_y, test_size = 0.3,random_state = 1 )
    pipe = Pipeline(steps=[('poly', PolynomialFeatures(include_bias=False)), ('model', LinearRegression()),])
    poly_d = GridSearchCV(estimator=pipe,param_grid={'poly__degree': np.arange(2,6)},scoring=score,cv=cv)   
    results=poly_d.fit(X_train,y_train) 
    mae,mse,rmse,r2 = evaluation (results, X_test, y_test)
    return(mae,mse,rmse,r2,results,'poly__degree',X_test, y_test)


def ridge_regression(t_X,t_y,score):
    t_X= sc.fit_transform(t_X)
    X_train, X_test, y_train, y_test = train_test_split(t_X, t_y, test_size = 0.2,random_state = 1)
    parameters = {'alpha':[1, 10]}
    model = Ridge()
    Ridge_reg= GridSearchCV(model, parameters, scoring=score,cv=cv)
    results = Ridge_reg.fit(X_train, y_train)
    mae,mse,rmse,r2 = evaluation (results, X_test, y_test)
    return(mae,mse,rmse,r2,results,'alpha')   


def lasso_regression(t_X,t_y,score):
    t_X= sc.fit_transform(t_X)
    X_train, X_test, y_train, y_test = train_test_split(t_X, t_y, test_size = 0.2,random_state = 1)
    parameters = {'alpha':[3, 10]}
    model = Lasso()
    lasso_reg= GridSearchCV(model, parameters, scoring=score,cv=cv)
    results = lasso_reg.fit(X_train, y_train)
    mae,mse,rmse,r2 = evaluation (results, X_test, y_test)
    return(mae,mse,rmse,r2,results,'alpha') 


def random_forect(t_X,t_y,score):
    t_X= sc.fit_transform(t_X)
    X_train, X_test, y_train, y_test = train_test_split(t_X, t_y, test_size = 0.2,random_state = 1)
    rfc=RandomForestRegressor(random_state = 1)
    param_grid = { 'n_estimators': [100],'max_features': ['auto'],'max_depth' : [4,5,6,7,8,9],'criterion' :['mse']}
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, scoring=score, cv= cv)
    results = CV_rfc.fit(X_train, y_train)
    mae,mse,rmse,r2 = evaluation (results, X_test, y_test)
    return(mae,mse,rmse,r2,results,'max_depth') 

      
def KNN(t_X,t_y,score):
    t_X= sc.fit_transform(t_X)
    X_train, X_test, y_train, y_test = train_test_split(t_X, t_y, test_size = 0.2,random_state = 1)
    pipeline=KNeighborsClassifier()
    param_grid = dict(n_neighbors=( range(1, 30)))
    grid_search = GridSearchCV(pipeline,param_grid=param_grid, scoring=score,cv=cv)    
    results=grid_search.fit(X_train, y_train)
    mae,mse,rmse,r2 = evaluation (results, X_test, y_test)
    return(mae,mse,rmse,r2,results,'n_neighbors') 
    
 
print("#########for Positive Tests:########## ")
metrix_Positive=metrix(X,y)
print(metrix_Positive)

print("########for Not Positive Tests:####### ")
metrix_not_Positive=metrix(X,y2)
print(metrix_not_Positive)


    
