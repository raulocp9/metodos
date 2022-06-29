

import numpy as np
import math 
import sympy as smp
from tabulate import *
import statsmodels.api as sm
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import scipy 

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
    smp.pprint(smp.Integral(f1, (x, a, b)),use_unicode=False)
    print('\n2.\n')
    smp.pprint(smp.Integral(f2, (x, a, b)))
    print('\n3. Quit')
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

def check_integrability():
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
        print('\nThe integration value using Simpson 1/3 for {} intervals is {}\n'.format(n, result))
    else:
        n = n-1
        h = (b - a)/n
        b1 = b - h
        result = simpson_1_3(a, b1, n, opc, h)
        
        result_2 = simpson_3_8(b1, b, 3, opc)
        print('\nThe integration value using Simpson 1/3 = {} + Simpson 3/8 = {}. Total = {} \n'.format(result, result_2, result+result_2))
        
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

if __name__ == "__main__":

    while True:
        opc = menu_integration()
        if opc == 3:
            break
        a, b, n = menu_2_integration(opc)
        integrability = check_integrability()
        if integrability == -1:
            break
        simpson(a, b, n, opc)
        romberg(a, b, opc)
        print('\n\nWait for plot')
        ploting(a, b, opc)
