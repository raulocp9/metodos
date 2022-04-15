import numpy as np
import math
import sympy as smp
from tabulate import *


def menu():
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
    
def menu2(matrix):
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

def select_backward_forward(poit, x):
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
        backward(x, point, backward_degree) 

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
    print(matrix)
    for i in range(2 ,len(matrix)+1):
        for j in range(len(matrix)-i+1):
            matrix[j][i] = matrix[j+1][i-1]-matrix[j][i-1]
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
    print("Value of the interpolation {:.8f}\t {:.8f}".format(y, y-yold))     
     

def backward(x, point, backward_degree):
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

    for j in range(backward_degree+1):
        matrix[j][0] = x[j][0]
        matrix[j][1] = x[j][1]
    print(matrix)
    for i in range(2 ,len(matrix)+1):
        for j in range(len(matrix)-i+1):
            matrix[j][i] = matrix[j+1][i-1]-matrix[j][i-1]
    print(matrix) 
    print(degree)
    print(len(matrix))
    print(len(matrix[1]))
    print(matrix.shape)
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
    print("Value of the interpolation {:.8f}\t {:.8f}".format(y, y-yold))     

if __name__=="__main__":
    while True:
        x, y = menu()
        while True:
            matrix = sort(x, y)
            check_equidistant(matrix)
            point = menu2(matrix)
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
        print('1. Go back to main menu')
        print('2. Exit')
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
        if(option == 2):
            exit()
        else: 
            continue    
        
    