import numpy as np
import math 
import sympy as smp
from tabulate import *
import statsmodels.api as sm
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy
import os
import tkinter
######### Newton-Raphon functions
def func1(x):  
    f1 = x[0][0]**2 + x[0][0]*x[1][0] -10
    f2 = x[1][0] + 3*x[0][0]*x[1][0]**2 -50
    return np.array([[f1],[f2]])

def J_func1(x):
    return np.array([[2*x[0][0] + x[1][0], x[0][0]],
                    [3*x[1][0]**2, 6*x[0][0]*x[1][0] + 1]])

def func2(x):    
    f1 = x[0][0]**2 + x[1][0]**2 -9
    f2 = -np.exp(x[0][0])-2*x[1][0]-3
    return np.array([[f1],[f2]])

def J_func2(x):
    return np.array([[2*x[0][0], 2*x[1][0]],
                    [-np.exp(x[0][0]), -2]])

def func3(x):
    f1 = 2*x[0][0]**2 - 4*x[0][0] + x[1][0]**2 + 3*x[2][0]**2 + 6*x[2][0] + 2
    f2 = x[0][0]**2 + x[1][0]**2 - 2*x[1][0] + 2*x[2][0]**2 -5
    f3 = 3*x[0][0]**2 - 12*x[0][0] + x[1][0]**2 - 3*x[2][0]**2 + 8
    return np.array([[f1],[f2],[f3]])

def J_func3(x):
    return np.array([[4*x[0][0] -4, 2*x[1][0], 6*x[2][0] + 6],
                    [2*x[0][0], 2*x[1][0] -2, 4*x[2][0]],
                    [6*x[0][0] -12, 2*x[1][0], -6*x[2][0]]])

def func4(x):
    f1 = x[0][0]**2 -4*x[0][0] + x[1][0]**2
    f2 = x[0][0]**2 -x[0][0] -12*x[1][0] +1
    f3 = 3*x[0][0]**2 -12*x[0][0] + x[1][0]**2 -3*x[2][0]**2 + 8
    return np.array([[f1],[f2],[f3]])

def J_func4(x):
    return np.array([[2*x[0][0] -4, 2*x[1][0], 0],
                    [2*x[0][0] -1, -12, 0],
                    [6*x[0][0] -12, 2*x[1][0], -6*x[2][0]]])

def menu():
    while(True):
            
        print('Program that calculates the root of the following systems of equations using Newton-Raphson method\n\n')
        print('\t\t\t Menu:')
        x, y, z =smp.symbols('x y z', real=True)
        smp.init_printing()    

        f1 = x**2 + x*y- 10
        g1 = y + 3*x*y**2 -50 
        print('1.\n')
        smp.pprint(f1)
        smp.pprint(g1)
        f2 = x**2 + y**2 -9
        g2 = -smp.exp(1)**x -2*y -3
        print('\n2.\n')
        smp.pprint(f2)
        smp.pprint(g2,use_unicode=True)
        f3 = 2*x**2 -4*x + y**2 + 3*z**2 + 6*z +2
        g3 = x**2 + y**2 -2*y +2*z**2 -5
        t3 = 3*x**2 -12*x + y**2 -3*z**2 + 8
        print('\n3.\n')
        smp.pprint(f3)
        smp.pprint(g3)
        smp.pprint(t3)
        f4 = x**2 -4*x +y**2
        g4 = x**2 -x -12*y +1
        t4 = 3*x**2 -12*x +y**2 -3*z**2 +8
        print('\n4.\n')
        smp.pprint(f4)
        smp.pprint(g4)
        smp.pprint(t4)
        print('''\n5. Main menu
        Enter a option: ''')
        try: 
            opc = int(input())
        except ValueError:
            print("Not a number. Try again\n")
        else:
            if 1 <= opc <= 5:
                break
            else:
                print("Out of range. Try again\n")
    return opc

def menu2():
    while True:
        print("Enter the maximum number of iterations")
        try:
            iteration = int(input())
        except ValueError:
            print("Not a number. Try again\n")
        else:
            break
    while True:
        print("Enter the maximum Ea(Absolute Error)")
        try:
            error = float(input())
        except ValueError:
            print("Not a number. Try again\n")
        else:
            break
    return iteration, error

def values1():
    while True:
        print("Enter the initial value of x")
        try:
            x = float(input())
        except ValueError:
            print("Not a number. Try again \n")
        else: 
            break
    while True:
        print("Enter the initial value of y")
        try:
            y = float(input())
        except ValueError:
            print('Not a number. Try again\n')
        else:
            break
    return np.array([[x],[y]])

def values2():
    while True:
        print("Enter the initial value of x")
        try:
            a  = float(input())
        except ValueError:
            print("Not a number. Try again\n")
        else:
            break
    while True:
        print("Enter the initial value of y")
        try:
            b = float(input())
        except ValueError:
            print("Not a number. Try again\n")
        else:
            break
    while True:
        print("Enter the initial values of z")
        try:
            c = float(input())
        except ValueError:
            print('Not a number. Try again\n')
        else:
            break
    return np.array([[a],[b],[c]])

def newton_rapshon_method():
    headers1 =["i", "x", "y", "Ea"]
    headers2 =["i", "x", "y", "z", "Ea"]
    while True: 
        opc = menu() 
        if opc == 1:
            iteration, error = menu2()
            x = values1()
            rows=[]
            for i in range(iteration):
                xold = x
                J_inv = np.linalg.inv(J_func1(x))
                product = np.dot(J_inv, func1(x))
                x = np.subtract(x,product)
                err = np.linalg.norm(x -xold)
                rows.append([i,x[0][0],x[1][0],err])
                if err < error:
                    break
            else:
                print("Max error could not be reached\n")
            print(tabulate(rows, headers= headers1)) 
            print("\n\n")

        elif opc == 2:
            iteration, error = menu2()
            x = values1()
            rows=[]
            for i in range(iteration):
                xold = x
                J_inv = np.linalg.inv(J_func2(x))
                product = np.dot(J_inv, func2(x))
                x = np.subtract(x,product)
                err = np.linalg.norm(x -xold)
                rows.append([i,x[0][0],x[1][0],err])

                if err < error:
                    break
            else:
                print("Max error could not be reached\n")
            print(tabulate(rows, headers= headers1))
            print("\n\n")

        elif opc == 3:
            iteration, error = menu2()
            x = values2()
            rows=[]
            for i in range(iteration):
                xold = x
                J_inv = np.linalg.inv(J_func3(x))
                product = np.dot(J_inv, func3(x))
                x = np.subtract(x,product)
                err = np.linalg.norm(x -xold)
                rows.append([i,x[0][0],x[1][0], x[2][0],err])

                if err < error:
                    break
            else:
                print("max error could not be reached\n")        
            print(tabulate(rows, headers= headers2))
            print("\n\n")
        elif opc ==4:
            iteration, error = menu2()
            x = values2()
            rows=[]
            for i in range(iteration):
                xold = x
                J_inv = np.linalg.inv(J_func4(x))
                product = np.dot(J_inv, func4(x))
                x = np.subtract(x,product)
                err = np.linalg.norm(x -xold)
                rows.append([i,x[0][0],x[1][0], x[2][0],err])

                if err < error:
                    break        
            else:
                print("max error could not be reached")
            print(tabulate(rows, headers= headers2))
            print("\n\n")

        elif opc ==5:
            break

########### interpolation_functions
def menu_interpolation():
    print('''\t\t\tPolynomial Interpolation\n
        \t\tNewton Interpolation Method\n''')
    while True:
        print('Enter the number of points in the table (Order does not matter, points must be equidistant)')
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
            line = line -1
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
    return x, y

def sort(x, y):
    matrix = np.array([x,y]).T

    for i in range(len(matrix)-1):
        for j in range(len(matrix)-1):
            if matrix[j][0] > matrix[j+1][0]:
                temp1 = matrix[j][0]
                temp2 = matrix[j][1]
                matrix[j][0] = matrix[j+1][0]
                matrix[j][1] = matrix[j+1][1]
                matrix[j+1][0] = temp1
                matrix[j+1][1] = temp2
    return matrix

def check_equidistant(x):
    is_equidistant = True
    difference = x[0][0]-x[1][0]
    for i in range(len(x)-1):
        diff = x[i][0]-x[i+1][0]
        if difference != diff:
            is_equidistant = False
    if is_equidistant == False:
        print("Points are not equidistants. Newton Interpolation Method can not be used.")
        quit()
    
def menu2_interpolation(matrix):
    lower_bound = matrix[0][0]
    upper_bound = matrix[len(matrix)-1][0]
    
    while True:
        print("Point to interpolate")
        try:
            a = float(input())
        except ValueError:
            print("Not a number. Try again")
        
        else:
            if a > lower_bound and a < upper_bound:
                break
            else:
                print("Point out of range.")
                while True:
                    print("Try again(Y/n)")
                    opc = input()
                    if(opc == 'n' or opc =='N'):
                        quit()
                    elif(opc !='y' and opc != 'Y'):
                        print("Not an option. Try again")
                    else:
                        break
    return a

def select_backward_forward(point, x):
    for i in range(len(x)-1):
        if point > x[i][0]:
            start = i
            continue
        else:
            break

    forward_degree = len(x)-start-1
    backward_degree = start+1
    print("\n\tEnter an option:")
    print("\n1. Newton forward, max degree of polynomial {}".format(forward_degree))
    print("\n2. Newton backward, max degree of polynomial {}".format(backward_degree))
    while True:
        try: 
            opc = int(input())
        except ValueError:
            print("Not a number. Try again")
        else:
            if(opc < 1 or opc > 2):
                print('Out of range. Try again')
            else:
                break
    return opc, start, forward_degree, backward_degree
    
def backward_forward(start, opc, x, point, forward_degree, backward_degree):
    if opc == 1:
        forward(start, x, point, forward_degree)
    else:
        backward(x, point, backward_degree, start) 

def forward(start, x, point, forward_degree):
    if(forward_degree == 0):
        print("This can not be done")
        return
    print('Enter the grade of the polynomial')
    while True:
        try: 
            degree = int(input())
        except ValueError:
            print("Not a number. Try again")
        else:
            if(degree > forward_degree or degree < 0):
                print("Out of range")
            else:
                break

    matrix = np.full((degree+1,degree+2),0)
    i=0
    for j in range(start,len(matrix)+start):
        matrix[i][0] = x[j][0]
        matrix[i][1] = x[j][1]
        i+=1 
    for i in range(2 ,len(matrix)+1):
        for j in range(len(matrix)-i+1):
            matrix[j][i] = matrix[j+1][i-1]-matrix[j][i-1]
    print('\nDifference table:')
    print(matrix) 
    h = np.absolute(matrix[0][0]-matrix[1][0])
    s = (point-matrix[0][0])/h
    y = matrix[0][1]
    s1 = s
    fact = 1
    for k in range(2,len(matrix[1])):
        yold = y
        y = y+(s1*matrix[0][k]/fact)
        fact = fact*k
        s1 = s1*(s-(k-1))
        
    print("Value of the interpolation: {:.8f}".format(y))     
    if degree !=1:
        print("error:{:.8f}".format(y-yold))     
    
def backward(x, point, backward_degree, start):

    if(backward_degree == 0):
        print("This can not be done")
        return
    print('Enter the grade of the polynomial')
    while True:
        try: 
            degree = int(input())
        except ValueError:
            print("Not a number. Try again")
        else:
            if(degree > backward_degree or degree <= 0):
              
              print("Out of range")
            else:
                break

    matrix = np.full((degree+1,degree+2),0)
    i=0
    for j in range(start+1-degree,start+2):
        matrix[i][0] = x[j][0]
        matrix[i][1] = x[j][1]
        i+=1

    for i in range(2 ,len(matrix)+1):
        for j in range(len(matrix)-i+1):
            matrix[j][i] = matrix[j+1][i-1]-matrix[j][i-1]
    print('\nDifference table')
    print(matrix) 

    h = np.absolute(matrix[0][0]-matrix[1][0])
    s = (point-matrix[len(matrix)-1][0])/h
    y = matrix[len(matrix)-1][1]
    s1 = s
    fact = 1
    for k in range(2,len(matrix[0])): 
        yold = y
        y = y+(s1*matrix[len(matrix)-k][k]/fact)
        fact = fact*k
        s1 = s1*(s-(k-1))
    print("Value of the interpolation: {:.8f}\t ".format(y))
    if degree !=1:
        print("error:{:.8f}".format(y-yold))         

def newton_interpolation_method():
    option2=0
    while option2 != 2:
        x, y = menu_interpolation()

        while True:
            matrix = sort(x, y)
            check_equidistant(matrix)
            point = menu2_interpolation(matrix)
            opc, start, forward_degree, backward_degree= select_backward_forward(point, matrix)
            backward_forward(start, opc, matrix, point, forward_degree, backward_degree)
            print("Do you want to intepolate another point with the same table?(Y/n)")
            try:
                option = input()
            except ValueError:
                print("Not an option. Try again")
            else:
                if(option == 'n' or option =='N'):
                    break
                else:
                    continue
        print('\n\tEnter an option')
        print('1. Go back to method')
        print('2. Main menu')

        while True:
            try:
                option2 = int(input())
            except ValueError:
                print('Not a number. Try again')
            else:
                if(option2<1 or option2 > 2):
                    print('Out of range. Try again')
                else: 
                    break
        if(option2 == 2):
            break

################## ordinary least squares functions
def ordinary_least_squares():
    while True:
        opc = menu_ols()
        if opc == 1:
            x, y= read_xlsx()
            
        elif opc == 2:
            x, y = read_keyboard()
        else: 
            break
        ols(x, y)

def menu_ols():
    print('\n\t\t Ordinary least squares regression')
    print('\n\t Enter a option: ') 
    print('''\n 1. Read data from excel file, must be in same directory, must be named "ols.xlsx", do not include index, 
    include headers, independent variable must be first and data must be vertically''')
    print('\n 2. Enter data from keyboard.') 
    print('\n 3. Main menu')
    while True:
        try:
            opc = int(input())
        except ValueError:
            print('Not a number try again.\n')
        else:
            if( opc == 1 or opc ==2 or opc ==3):
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

########## integration functions
def function_1(x):
    return (x**4*((np.sqrt(3+2*x**2))/3))

def function_2(x):
    return (x**5/ np.power((x**2+4),(1/5)))

def menu_integration():
    print('\t\tCalculate the integration of the following equations:')
    print('''\n
Using Simpson 1/3 for even intervals, 
Simpson 1/3 + 3/8 por odd intervals
Romberg method''')
    x, a, b = smp.symbols('x a b') 
    smp.init_printing()  
    f1 = x**4*((smp.sqrt(3+2*x**2))/3)
    f2 = x**5/ smp.Pow((x**2+4),(1/5))
    print('\n1.\n')
    smp.pprint(smp.Integral(f1, (x, a, b)))
    print('\n2.\n')
    smp.pprint(smp.Integral(f2, (x, a, b)))
    print('\n3. Main menu')
    print('Enter an option:')
    while True:
        try: 
            opc = int(input())
        except ValueError:
            print("Not a number. Try again\n")
        else:
            if 1 <= opc <= 3:
                break
            else:
                print("Out of range. Try again\n")

    return opc

def menu_2_integration(opc):
    print('Enter the lower limit of the integral:')
    while True:
        try: 
            a = float(input())
        except ValueError:
            print("Not a number. Try again\n")
        else:
            break
    print('Enter the upper limit of the integral:')
    while True:
        try: 
            b = float(input())
        except ValueError:
            print("Not a number. Try again\n")
        else:
            break
    print('Enter the number of integration intervals:')
    while True:
        try: 
            n = int(input())
        except ValueError:
            print("Not a number. Try again\n")
        else:
            if n<1:
                print("Intervals must be positive. Try again")
            else:
                break
    return a, b, n   

def check_integrability(a, b):
        if b < a:
            print("This can't be done. Upper limit must be bigger than low limit")
            return -1
        elif b == a:
            print('The limits are equals. The value is 0 ')    
            return -1
        else: 
            return 0

def ploting(a, b, opc):
    x = np.linspace(a, b)
    if opc == 1:
        plt.plot(x, function_1(x))
    else:
        plt.plot(x, function_2(x))
    plt.show()

def integral_1_3(a, b, opc):
    m = (a + b)/2
    if opc == 1:
        integral = (b - a) / 6 * (function_1(a) + 4 * function_1(m) + function_1(b))
    else:
        integral = (b - a) / 6 * (function_2(a) + 4 * function_2(m) + function_2(b))
    return integral

def integral_3_8(a, b, opc):
    m1 = (2*a + b)/ 3
    m2 = (a + 2*b)/ 3
    if opc == 1:
        integral = (b - a) / 8 * (function_1(a) + 3 * function_1(m1) + 3 * function_1(m2) + function_1(b))
    else:
        integral = (b - a) / 8 * (function_2(a) + 3 * function_2(m1) + 3 * function_2(m2) + function_2(b))
    return integral

def simpson(a, b, n, opc):
    if n % 2 == 0:
        h = (b - a)/n
        result = simpson_1_3(a, b, n, opc, h) 
        print('\n\n\nThe integration value using Simpson 1/3 for {} intervals is {}\n'.format(n, result))
    else:
        n = n-1
        h = (b - a)/n
        b1 = b - h
        result = simpson_1_3(a, b1, n, opc, h)
        
        result_2 = simpson_3_8(b1, b, 3, opc)
        print('\n\n\nThe integration value using Simpson 1/3 = {} + Simpson 3/8 = {}. Total = {} \n'.format(result, result_2, result+result_2))
        
def simpson_1_3(a, b, n, opc, h):
   
    suma = 0
    for i in range(n):
        b = a + h
        area = integral_1_3(a, b, opc)
        suma = suma + area
        a = b 
    return suma

def simpson_3_8(a, b, n, opc):
    h = (b - a)/3
    suma = 0
    for i in range(n):
        b = a + h
        area = integral_3_8(a, b, opc)
        suma = suma + area
        a = b 
    return suma

def romberg(a, b, opc):
    if opc == 1:
        integral = scipy.integrate.romberg(function_1, a, b, show=True)
    else:
        integral = scipy.integrate.romberg(function_2, a, b, show=True)

def integration():
    while True:
        opc = menu_integration()
        if opc == 3:
            break
        a, b, n = menu_2_integration(opc)
        integrability = check_integrability(a, b)
        if integrability == -1:
            break
        simpson(a, b, n, opc)
        romberg(a, b, opc)
        print('Wait for plot')
        ploting(a, b, opc)

######## main functions
def main_menu():
    print('''\n


                                                       /$$                     /$$                                 /$$     /$$                       /$$          
                                                      |__/                    | $$                                | $$    | $$                      | $$          
 /$$$$$$$  /$$   /$$ /$$$$$$/$$$$   /$$$$$$   /$$$$$$  /$$  /$$$$$$$  /$$$$$$ | $$       /$$$$$$/$$$$   /$$$$$$  /$$$$$$  | $$$$$$$   /$$$$$$   /$$$$$$$  /$$$$$$$
| $$__  $$| $$  | $$| $$_  $$_  $$ /$$__  $$ /$$__  $$| $$ /$$_____/ |____  $$| $$      | $$_  $$_  $$ /$$__  $$|_  $$_/  | $$__  $$ /$$__  $$ /$$__  $$ /$$_____/
| $$  \ $$| $$  | $$| $$ \ $$ \ $$| $$$$$$$$| $$  \__/| $$| $$        /$$$$$$$| $$      | $$ \ $$ \ $$| $$$$$$$$  | $$    | $$  \ $$| $$  \ $$| $$  | $$|  $$$$$$ 
| $$  | $$| $$  | $$| $$ | $$ | $$| $$_____/| $$      | $$| $$       /$$__  $$| $$      | $$ | $$ | $$| $$_____/  | $$ /$$| $$  | $$| $$  | $$| $$  | $$ \____  $$
| $$  | $$|  $$$$$$/| $$ | $$ | $$|  $$$$$$$| $$      | $$|  $$$$$$$|  $$$$$$$| $$      | $$ | $$ | $$|  $$$$$$$  |  $$$$/| $$  | $$|  $$$$$$/|  $$$$$$$ /$$$$$$$/
|__/  |__/ \______/ |__/ |__/ |__/ \_______/|__/      |__/ \_______/ \_______/|__/      |__/ |__/ |__/ \_______/   \___/  |__/  |__/ \______/  \_______/|_______/ 
                                                                                                                                                                  
                                                                                                                                                                  
                                                                                                                                                                  
                                                                                                                                                                  
                                                                                                                                                                  
                                                  /$$$$$$   /$$$$$$   /$$$$$$   /$$$$$$   /$$$$$$  /$$$$$$  /$$$$$$/$$$$                                          
                                                 /$$__  $$ /$$__  $$ /$$__  $$ /$$__  $$ /$$__  $$|____  $$| $$_  $$_  $$                                         
                                                | $$  \ $$| $$  \__/| $$  \ $$| $$  \ $$| $$  \__/ /$$$$$$$| $$ \ $$ \ $$                                         
                                                | $$  | $$| $$      | $$  | $$| $$  | $$| $$      /$$__  $$| $$ | $$ | $$                                         
                                                | $$$$$$$/| $$      |  $$$$$$/|  $$$$$$$| $$     |  $$$$$$$| $$ | $$ | $$ /$$                                     
                                                | $$____/ |__/       \______/  \____  $$|__/      \_______/|__/ |__/ |__/|__/                                     
                                                | $$                           /$$  \ $$                                                                          
                                                | $$                          |  $$$$$$/                                                                          
                                                |__/                           \______/                                                                           
                                                                                                                                                                  
Press enter to continue...                                                                                                                                                                 
                                                                                                                                                                
                                                                                                                                                                
                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                                                                                                                     
                                                                                                                                                                                                      
MIT licesce.
Copyright(c) 2022
        ''')
    a = input()

def main_menu_2():

    while True:
        print('\n\t\tThis program implements the following numerical methods.')

        print('\n1. Newton-Raphson Method.')
        print('\n2. Newton Interpolation Method.')
        print('\n3. Ordinary Least Squares Method.')
        print('\n4. Integration Methods.') 
        print('\n5. Quit.')
        print('\nEnter a option:')
        try: 
            opc = int(input())
        except ValueError:
            print("Not a number. Try again\n")
        else:
            if 1 <= opc <= 5:
                break
            else:
                print("Out of range. Try again\n")
    return opc

def main_switch(opc):
    if opc == 1:
        newton_rapshon_method()
    elif opc == 2:
        newton_interpolation_method()
    elif opc ==3:
        ordinary_least_squares()
    elif opc == 4:
        integration()
    else:
        os._exit(0)



if __name__=="__main__":
    main_menu()
    while True:
        main_opc = main_menu_2()
        main_switch(main_opc)

    
        
    

