from re import ASCII
import numpy as np
import math 
import sympy as smp
from tabulate import *

def func1(x):
    f1 = x[0][0]**2 + x[0][0]*x[1][0] -10
    f2 = x[1][0]**2 + 3*x[0][0]*x[1][0]**2 -50
    return np.array([[f1],[f2]])

def J_func1(x):
    return np.array([[2*x[0][0] + x[1][0], x[0][0]],
                    [3*x[1][0]**2, 6*x[0][0]*x[1][0] + 1]])

def func2(X):
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

        f1 = 1.- x**2 + x*y- 10
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
        g3 = x + y**2 -2*y +2*z -5
        t3 = 2*x**2 -12*x + y**2 -3*z + 8
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
        print('''\n5. Exit
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


if __name__=="__main__":    
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
            quit()

