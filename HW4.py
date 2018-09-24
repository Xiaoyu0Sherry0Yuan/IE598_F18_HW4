import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
'python-machine-learning-book-2nd-edition'
'/master/code/ch10/housing.data.txt',sep='\s+')
##################################EDA

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS'
              , 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
plt.show()
#################CORELATION HEATMAP
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f'
                 ,annot_kws={'size': 15},yticklabels=cols,xticklabels=cols)
plt.show()
###################################SPLIT
X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

#######################residual plot
plt.scatter(y_train_pred, y_train_pred - y_train,c='steelblue'
            , marker='o', edgecolor='white',label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test,c='limegreen'
            , marker='s', edgecolor='white',label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()
#########################slop intercept mse and r2
Xb = np.hstack((np.ones((X.shape[0], 1)), X))
w = np.zeros(X.shape[1])
z = np.linalg.inv(np.dot(Xb.T, Xb))
w = np.dot(z, np.dot(Xb.T, y))
print(w)
print('Intercept: %.3f' % w[0])
slr = LinearRegression()
slr.fit(X, y)
print('MSE train: %.3f, test: %.3f' % ( mean_squared_error(y_train
                                                            , y_train_pred)
    ,mean_squared_error(y_test, y_test_pred)))

print('R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),
                                      r2_score(y_test, y_test_pred)))


###########################Ridge
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []
ridge = Ridge(normalize=True)

for alpha in alpha_space:
    ridge.alpha = Ridge(alpha,normalize=True)
    ridge_cv_scores = cross_val_score(ridge.alpha,X,y,cv=10)
    ridge_scores.append(np.mean(ridge_cv_scores))
    ridge_scores_std.append(np.std(ridge_cv_scores))

def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)
    std_error = cv_scores_std / np.sqrt(10)
    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()
display_plot(ridge_scores, ridge_scores_std)

alpha_space =np.arange(0,10)
MSE_test=np.empty(len(alpha_space))
r2_test=np.empty(len(alpha_space))

for a in alpha_space:
    
    ridge_alpha=Ridge(alpha=a,normalize=True)
    ridge_alpha.fit(X_train,y_train)
    y_test_pred=ridge_alpha.predict(X_test)
    MSE_test[a]=mean_squared_error(y_test, y_test_pred)
    r2_test[a]=r2_score(y_test, y_test_pred)
    

plt.plot(alpha_space, MSE_test)
_=plt.xlabel('alpha')
_=plt.ylabel('MSE')
plt.show()

plt.plot(alpha_space, r2_test)
_=plt.xlabel('alpha')
_=plt.ylabel('r2')
plt.show()

ridge=Ridge(alpha=0,normalize=True)
ridge.fit(X_train,y_train)
y_test_predict1=ridge.predict(X_test)
plt.scatter(y_test_predict1, y_test_predict1 - y_test,label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()

print('MSE test:')
print(mean_squared_error(y_test, y_test_predict1))
print('R^2 test:')
print(r2_score(y_test, y_test_predict1))
###########################lasso
alpha_space =np.arange(0,10)
MSE_test=np.empty(len(alpha_space))
r2_test=np.empty(len(alpha_space))

for k in alpha_space:
    
    lasso_alpha=Lasso(alpha=k,normalize=True)
    lasso_alpha.fit(X_train,y_train)
    y_test_pred=lasso_alpha.predict(X_test)
    MSE_test[k]=mean_squared_error(y_test, y_test_pred)
    r2_test[k]=r2_score(y_test, y_test_pred)
    

plt.plot(alpha_space, MSE_test)
_=plt.xlabel('alpha')
_=plt.ylabel('MSE')
plt.show()

plt.plot(alpha_space, r2_test)
_=plt.xlabel('alpha')
_=plt.ylabel('r2')
plt.show()

lasso_op=Lasso(alpha=0,normalize=True)
lasso_op.fit(X_train,y_train)
y_test_predict1=lasso_op.predict(X_test)
plt.scatter(y_test_predict1, y_test_predict1 - y_test,label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()

print('MSE test:')
print(mean_squared_error(y_test, y_test_predict1))
print('R^2 test:')
print(r2_score(y_test, y_test_predict1))
##################################################t reElastic


l1_ratio_space =np.arange(0,10)
MSE_test=np.empty(len(l1_ratio_space))
r2_test=np.empty(len(l1_ratio_space))

for j in alpha_space:
    
    elanet = ElasticNet(alpha=1.0, l1_ratio=j)
    elanet.fit(X_train,y_train)
    y_test_pred=elanet.predict(X_test)
    MSE_test[j]=mean_squared_error(y_test, y_test_pred)
    r2_test[j]=r2_score(y_test, y_test_pred)
    

plt.plot(l1_ratio_space, MSE_test)
_=plt.xlabel('l1_ratio')
_=plt.ylabel('MSE')
plt.show()

plt.plot(l1_ratio_space, r2_test)
_=plt.xlabel('l1_ratio')
_=plt.ylabel('r2')
plt.show()

elanet_op=Lasso(alpha=0,normalize=True)
elanet_op.fit(X_train,y_train)
y_test_predict1=elanet_op.predict(X_test)
plt.scatter(y_test_predict1, y_test_predict1 - y_test,label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()

print('MSE test:')
print(mean_squared_error(y_test, y_test_predict1))
print('R^2 test:')
print(r2_score(y_test, y_test_predict1))


print("My name is Xiaoyu Yuan")
print("My NetID is: 664377413")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
