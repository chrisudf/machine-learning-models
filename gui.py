import tkinter as tk
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

window = tk.Tk()
window.title('my window')
window.geometry('400x400')


ll1 = tk.Label(window, bg='green', width=50, text='R2')
ll2 = tk.Label(window, bg='green', width=50, text='MSE')
ll3 = tk.Label(window, bg='green', width=50, text='MAE')
ll4 = tk.Label(window, bg='green', width=50, text='MSE%')
ll1.pack()
ll2.pack()
ll3.pack()
ll4.pack()


tsize=0.0034
para = [0]
var1 = tk.IntVar()
var1.set(1)
var2 = tk.IntVar()
var3 = tk.IntVar()
var4 = tk.IntVar()
var5 = tk.IntVar()
var6 = tk.IntVar()
var6.set(1)
var7 = tk.IntVar()
var8 = tk.IntVar()
varlist=[var1,var2,var3,var4,var5]


def change_para():
    global tsize
    if (var1.get() == 1):
        if (0 not in para):
            para.append(0)
    if (var1.get() == 0):
        if (0 in para):
            para.remove(0)
    if (var2.get() == 1):
        if (1 not in para):
            para.append(1)
    if (var2.get() == 0):
        if (1 in para):
            para.remove(1)
    if (var3.get() == 1):
        if (2 not in para):
            para.append(2)
    if (var3.get() == 0):
        if (2 in para):
            para.remove(2)
    if (var4.get() == 1):
        if (3 not in para):
            para.append(3)
    if (var4.get() == 0):
        if (3 in para):
            para.remove(3)
    if (var5.get() == 1):
        if (4 not in para):
            para.append(4)
    if (var5.get() == 0):
        if (4 in para):
            para.remove(4)
    if (var6.get() == 1) or ((var7.get() == 0) and (var8.get() == 0)) :
        tsize = 0.0034
    if (var7.get() == 1):
        tsize = 0.01
    if (var8.get() == 1):
        tsize = 0.1
    print(tsize)
    model()


def model():
    dataset = pd.read_csv("C:/Users/Qiu/Desktop/cc.csv")
    datasize = 8000
    pnum = 4
    split = 0.1
    x = dataset.iloc[0:datasize,para].values
    print(x[0])
    y = dataset.iloc[0:datasize, 5].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =tsize, random_state=33)
    # print(y_test)
    ss_x = StandardScaler()
    ss_y = StandardScaler()

    x_train = ss_x.fit_transform(x_train)
    x_test = ss_x.transform(x_test)
    y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
    y_test = ss_y.transform(y_test.reshape(-1, 1))

    rbf_svr = SVR(kernel='rbf')
    rbf_svr.fit(x_train, y_train.ravel())
    rbf_svr_predict = rbf_svr.predict(x_test)

    ttest = ss_y.inverse_transform(y_test)
    ppre = ss_y.inverse_transform(rbf_svr_predict)
    mmax = max(ss_y.inverse_transform(y_test))

    print(len(ttest))
    aaa = 0
    for i in range(len(ttest)):
        aaa += ((ttest[i] - ppre[i]) / mmax) * ((ttest[i] - ppre[i]) / mmax)
    percent=float(aaa/len(ttest))
    print(percent)
    ttest = np.array(ttest)
    ppre = np.array(ppre)

    R2=r2_score(y_test, rbf_svr_predict)
    MSE= mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_predict))
    MAE = mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_predict))

    # ll1.config(text=R2)
    ll1.config(text="R2 value: "+str(R2))
    ll2.config(text="MSE value: "+str(MSE))
    ll3.config(text="MAE value: "+str(MAE))
    ll4.config(text="MSE %: " + str(percent))


    print('\nThe value of default measurement of rbf SVR is', rbf_svr.score(x_test, y_test))
    print('R-squared value of rbf SVR is', r2_score(y_test, rbf_svr_predict))
    print('The mean squared error of rbf SVR is',
          mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_predict)))
    print('The mean absolute error of rbf SVR is',
          mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_predict)))

    print(np.square(np.subtract(ttest, ppre)).mean())  # print(ss_y.inverse_transform(y_test))
    # print(ss_y.inverse_transform(rbf_svr_predict))
    print(np.square(np.subtract(ttest, ppre) / mmax).mean())


c1 = tk.Checkbutton(window, text='time', variable=var1, onvalue=1, offvalue=0,
                    command=change_para)
c2 = tk.Checkbutton(window, text='airtemp', variable=var2, onvalue=1, offvalue=0,
                    command=change_para)
c3 = tk.Checkbutton(window, text='humidity', variable=var3, onvalue=1, offvalue=0,
                    command=change_para)
c4 = tk.Checkbutton(window, text='insolation', variable=var4, onvalue=1, offvalue=0,
                    command=change_para)
c5 = tk.Checkbutton(window, text='windspeed', variable=var5, onvalue=1, offvalue=0,
                    command=change_para)


c6 = tk.Checkbutton(window, text='20 mins forecast', variable=var6, onvalue=1, offvalue=0,
                    command=change_para)

c7 = tk.Checkbutton(window, text='60 mins forecast', variable=var7, onvalue=1, offvalue=0,
                    command=change_para)

c8 = tk.Checkbutton(window, text='10 hours forecast', variable=var8, onvalue=1, offvalue=0,
                    command=change_para)
c1.pack()
c2.pack()
c3.pack()
c4.pack()
c5.pack()
c6.pack()
c7.pack()
c8.pack()
model()


window.mainloop()

