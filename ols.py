import statsmodels.api as sm
import numpy as np
import pandas as pd 
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from tabulate import *

def run():
    opc = menu()
    if opc == 1:
        x, y= read_xlsx()
        
    else:
        x, y = read_keyboard()

    ols(x, y)

def menu():
    print('\n\t\t Ordinary least squares regression')
    print('\n\t Enter a option: ') 
    print('\n 1. Read data from file, must be in same directory, must be named ols.xlsx, do not include index, independent variable must be first and data must be vertically')
    print('\n 2. Enter data from keyboard.') 
    while True:
        try:
            opc = int(input())
        except ValueError:
            print('Not a number try again.\n')
        else:
            if( opc == 1 or opc ==2):
                break
            else:
                print('Not an option. Try again.')
    return opc

def read_keyboard():
    while True:
        print('Enter the number of points in the table')
        try:
            n = int(input())
        except ValueError:
            print('Not a number try again\n')
        else: 
            break
    x=[]
    y =[]
    for i in range(n):
        while True:
            print('Enter the x value of the point {}'.format(i+1))
            try:
                a = float(input())
            except ValueError:
                print("Not a number. Try again")
            else: 
                break
        print('Enter the y value of the point {}'.format(i+1))
        while True:
            try:
                b = float(input())
            except ValueError:
                print('Not a number. Try again')
            else:
                break
        x.append(a)
        y.append(b)

    while True:
        headers= ["i", "x","y"]
        table1 = zip(x, y)
        row = list(range(1,n+1))
        print(tabulate(table1, headers=headers, showindex=row))
        print("Is the table correct?(Y/n)")
        a = input()
        if( a == 'y' or a=='Y'):
            break
        elif( a=='n' or a=='N'):
            while True:
                print("Enter the line to modify:")
                try:
                    line = int(input())
                except ValueError:
                    print("Not a number. Try again")
                else:
                    if(line > n or line <= 0 ):
                        print("Not a valid line. Try again")
                    else: 
                        break
            line = line - 1
            while True:
                print('Enter the new x value of the point {}'.format(line+1))
                try:
                    opc_1 = float(input())
                except ValueError:
                    print("Not a number. Try again")
                else: 
                    break
            print('Enter the new y value of the point {}'.format(line+1))
            while True:
                try:
                    opc_2 = float(input())
                except ValueError:
                    print('Not a number. Try again')
                else:
                    break
            x[line] = opc_1
            y[line] = opc_2
        else:
            print('Not a option. Try again')

    x = np.array(x)
    y = np.array(y)
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    return x, y

def read_xlsx():
    df = pd.read_excel("ols.xlsx")
    while True:

        print(df)
        print('Is the table correct?(Y/n)')
        a = input()
        if a == 'y' or a == 'Y':
            break;
        elif a == 'n' or a == 'N':

            print('Enter index to modify: ')
            while True:
                try:
                    b = int(input())
                except ValueError:
                    print('Not a number. Try again')
                else:
                    if b < 0 or b >= len(x):
                        print('Out of range. Try again')
                    else:
                        break

            print('Enter the first value: ')
            while True:
                try:
                    value_x = float(input())
                except ValueError:
                    print('Not a number. Try again')
                else:
                    break
                
            print('Enter the second value: ')
            while True:
                try:
                    value_y = float(input())
                except ValueError:
                    print('Not a number. Try again')
                else:
                    break

            df.iloc[b, 0] = value_x
            df.iloc[b, 1] = value_y
        else:
            print('Not an option. Try again')


    x = df.iloc[:, 0]
    y = df.iloc[:, 1]

    x = np.asarray(x)
    y = np.asarray(y)
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]

    return x, y

def ols(x, y):
    max_degree = len(x)-1
    while True:
        print('\nEnter de degree of the polynomial, max degree = ', max_degree)
        try:
            n = int(input())
        except ValueError:
            print('Not a number. Try again\n')
        else:
            if n <= 0 or n > max_degree:
                print('Out of range. Try again')
            else:
                break
    polynomial_features = PolynomialFeatures(degree=n)
    xp = polynomial_features.fit_transform(x)
    #x = sm.add_constant(x) # adding a constant
    model = sm.OLS( y, xp).fit() # fitting the model
    ypred = model.predict(xp)
    #print(model.summary())
    print('\nCoeficients: [b0, b1, b2,..., bn]')
    print(model.params)
    #print('\n Etandard error: b0, b1, b2, ..., bn')
    #print(model.bse)
    #print(model.fittedvalues)
    #print(model.resid)
    #print(model.predict(xp))
    suma = 0
    for i in range(len(model.resid)):
        suma += model.resid[i]**2
    
    print('\nModel error: ',suma)
    print('\n Wait for plot...')
    fig = plt.figure('OLS')
    plt.scatter(x, y)
    plt.plot(x, ypred)
    plt.show()

if __name__ =='__main__':
    run()
