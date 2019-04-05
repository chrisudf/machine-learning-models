from sklearn.datasets import load_boston
import numpy as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

boston = load_boston()
# print(boston.DESCR)

x = boston.data
y = boston.target

# print(x,file=task)
# print(len(x))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

print(x_test)
print(len(x_test))
ss_x = StandardScaler()
ss_y = StandardScaler()

x_train = ss_x.fit_transform(x_train)
print(x_train.shape)
x_test = ss_x.transform(x_test)
print(x_test.shape)
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
print(y_train.shape())
y_test = ss_y.transform(y_test.reshape(-1, 1))
print(y_test.shape())



linear_svr = SVR(kernel='linear')
linear_svr.fit(x_train, y_train.ravel())
linear_svr_predict = linear_svr.predict(x_test)

poly_svr = SVR(kernel='poly')
poly_svr.fit(x_train, y_train.ravel())
poly_svr_predict = poly_svr.predict(x_test)

rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(x_train, y_train.ravel())
rbf_svr_predict = rbf_svr.predict(x_test)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print('The value of default measurement of linear SVR is', linear_svr.score(x_test, y_test))
print('R-squared value of linear SVR is', r2_score(y_test, linear_svr_predict))
print('The mean squared error of linear SVR is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_predict)))
print('The mean absolute error of linear SVR is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_predict)))

print('\nThe value of default measurement of poly SVR is', poly_svr.score(x_test, y_test))
print('R-squared value of poly SVR is', r2_score(y_test, poly_svr_predict))
print('The mean squared error of poly SVR is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_predict)))
print('The mean absolute error of poly SVR is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_predict)))

print('\nThe value of default measurement of rbf SVR is', rbf_svr.score(x_test, y_test))
print('R-squared value of rbf SVR is', r2_score(y_test, rbf_svr_predict))
print('The mean squared error of rbf SVR is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_predict)))
print('The mean absolute error of rbf SVR is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_predict)))

print(len(ss_y.inverse_transform(rbf_svr_predict)))
print((ss_y.inverse_transform(rbf_svr_predict)))