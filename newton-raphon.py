import numpy as np
import math 

def func1(x):
    f1 = x[0][0]**2 + x[0][0]*x[1][0] -10
    f2 = x[1][0] + 3*x[0][0]*x[1][0]**2 -50
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
        print('''\t\t\t Menu:

        1. x^2 + xy -10 = 0
           y + 3xy^2 -50 = 0
        
        2. x^2 + y^2 -9 = 0
           -e^x -2y -3 = 0
        
        3. 2x^2 -4x + y^2 + 3z^2 + 6z +2 = 0
           x + y^2 -2y + 2z -5 = 0
           2x^2 -12x + y^2 -3z +8 = 0
        
        4. x^2 -4x +y^2 = 0
           x^2 -x -12y +1 = 0
           3x^2 -12x + y^2 -3z^2 +8 = 0
        
        5. Exit
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
    print("Enter the maximum number of iterations")
    iteration = int(input())
    print("Enter the maximum Ea(Absolute Error)")
    error = float(input())
    return iteration, error

def values1():

    print("Enter the initial value of x")
    x = float(input())
    print("Enter the initial value of y")
    y = float(input())
    print("\n")
    return np.array([[x],[y]])

def values2():
    print("Enter the initial value of x")
    a  = float(input())
    print("Enter the initial value of y")
    b = float(input())
    print("Enter the initial values of z")
    c = float(input())
    print("\n")
    return np.array([[a],[b],[c]])


if __name__=="__main__":

    while True: 
        opc = menu() 
        if opc == 1:#newton-raphon method
            iteration, error = menu2() # max number of iterations and max error given by user
            x = values1()# initial point given by user
    
            for i in range(iteration):
                xold = x
                J_inv = np.linalg.inv(J_func1(x))
                product = np.dot(J_inv, func1(x))
                x = np.subtract(x,product)
                err = np.linalg.norm(x -xold)
                print(''' Iteration\t\tx\t\t\ty\t\t\terror
    {}\t \t{}\t {}\t {:.10f}
                
                
                        '''.format(i, x[0][0],x[1][0], err) )

                if err < error:
                    break
            else:
                print("max error could not be reached")

        elif opc == 2:
            iteration, error = menu2()
            x = values1()

            for i in range(iteration):
                xold = x
                J_inv = np.linalg.inv(J_func2(x))
                product = np.dot(J_inv, func2(x))
                x = np.subtract(x,product)
                err = np.linalg.norm(x -xold)
                print(''' Iteration\t\tx\t\t\ty\t\t\terror
    {}\t \t{}\t {}\t {:.10f}
                
                
                        '''.format(i, x[0][0],x[1][0], err) )

                if err < error:
                    break
            else:
                print("max error could not be reached")
            
        elif opc == 3:
            iteration, error = menu2()
            x = values2()# initial point given by user
    
            for i in range(iteration):
                xold = x
                J_inv = np.linalg.inv(J_func3(x))
                product = np.dot(J_inv, func3(x))
                x = np.subtract(x,product)
                err = np.linalg.norm(x -xold)
                print(''' Iteration\t\tx\t\t\ty\t\t\tz\t\terror
    {}\t \t{}\t {}\t {}\t{:.10f}
                
                
                        '''.format(i, x[0][0], x[1][0], x[2][0], err) )

                if err < error:
                    break
            else:
                print("max error could not be reached")        

        elif opc ==4:
            iteration, error = menu2()
            x = values2()# initial point given by user
    
            for i in range(iteration):
                xold = x
                J_inv = np.linalg.inv(J_func4(x))
                product = np.dot(J_inv, func4(x))
                x = np.subtract(x,product)
                err = np.linalg.norm(x -xold)
                print(''' Iteration\t\tx\t\t\ty\t\t\tz\t\terror
    {}\t \t{}\t {}\t{}\t {:.10f}
                
                
                        '''.format(i, x[0][0], x[1][0], x[2][0], err) )

                if err < error:
                    break        
            else:
                print("max error could not be reached")

        elif opc ==5:
            quit()

