import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def warmUpExercise():
    '''
    WARMUPEXERCISE Example function in octave
    A = WARMUPEXERCISE() is an example function that returns the 5x5 identity matrix
    :return: 5x5 identity matrix
    '''
    # ============= YOUR CODE HERE ==============
    #Instructions: Return the 5x5 identity matrix
    #   in octave, we return values by defining which variables
    #   represent the return values (at the top of the file)
    #   and then set them accordingly.
    A=np.eye(5, dtype=np.int32)
    return A


def readFile(filePath):
    '''
    this function wasn't inculuded in the project but it's little different and diffcult to load data in python, so
    i put it in seperate function.
    :param filePath:
    :return: x values to be put on x-axis
            y values to be put on y-axis
    '''
    file = open(filePath)
    x = []
    y = []
    for line in file:
        stripped_line = line.strip().split(",")
        x.append(float(stripped_line[0]))
        y.append(float(stripped_line[1]))
    xn=np.array(x)
    yn=np.array(y)
    del x
    del y
    return xn,yn

def plotData(x,y,label="Date points", marker="x"):
    """

     PLOTDATA Plots the data points x and y into a new figure
     PLOTDATA(x,y) plots the data points and gives the figure axes labels of
     population and profit
     """
    #plt.ion()
    #plt.show()

    #====================== YOUR CODE HERE ======================

    #Instructions: Plot the training data into a figure using the
    #       "figure" and "plot" commands. Set the axes labels using
    #       the "xlabel" and "ylabel" commands. Assume the
    #       population and revenue data have been passed in
    #       as the x and y arguments of this function.
    plt.figure(1)
    plt.plot(x, y, marker,label="Date points")
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.title("Population vs. Profits")
    plt.style.use('fivethirtyeight')
    plt.legend()


    #plt.draw()
    #plt.pause(0.01)


def computeCost(x,y,theta ):
    '''
    COMPUTECOST Compute cost for linear regression
    J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y
    :param x: x values
    :param y: y values
    :param theta: theta to be tested
    :return: the cost of theta
    '''
    #Initialize some useful values
    m=len(y)
    hx=theta[0] + x*theta[1]
    squaredDiff = (np.sum((hx-y)**2))
    return squaredDiff/(2*m)

def gradientDescent(x,y, theta, alpha, num_iters):
    '''
    GRADIENTDESCENT Performs gradient descent to learn theta
    theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
    taking num_iters gradient steps with learning rate alpha

    '''
    # Initialize some useful values
    m = len(y)

    for iter in range(num_iters):
        hx =theta[0] + x * theta[1]
        sumDiff =np.sum(hx -y)
        sumDiffMultx=np.sum((hx-y)*x)
        theta0New=theta[0]-(alpha/m)* sumDiff
        theta1New =theta[1]-(alpha/m)* sumDiffMultx
        theta[0]=theta0New
        theta[1]=theta1New



## ==================== Part 1: Basic Function ====================
print('Running warmUpExercise ...')
print('5x5 Identity Matrix:')
print(warmUpExercise())
onHold=input("Program paused. Press enter to continue.")

## ======================= Part 2: Plotting =======================
print("Plotting Data ...")
x,y=readFile("ex1data1.txt")
X =np.ones((len(x),2))
X[:,1]=x
m=len(y) # number of training examples

#Plot Data
plotData(x,y)

input("Program paused. Press enter to continue.")

## =================== Part 3: Cost and Gradient descent ===================

theta=np.zeros((2,1),dtype=np.float64) # initialize fitting parameters
#Some gradient descent settings
iterations=1500
alpha=0.01
print("Testing the cost function ...")

#compute and display initial cost
J = computeCost(x, y, theta)
print('With theta = [0 ; 0]\nCost computed = %f\n', J)
print('Expected cost value (approx) 32.07\n')

# further testing of the cost function
J = computeCost(x, y, np.array([-1,2]))
print('With theta = [-1 ; 2]\nCost computed = %f\n', J)
print('Expected cost value (approx) 54.24\n')

input("Program paused. Press enter to continue.")

print("Running Gradient Descent ...")
# run gradient descent
gradientDescent(x, y, theta, alpha, iterations)
print('Theta found by gradient descent:')
print(theta)
print('Expected theta values (approx)')
print(' -3.6303\n  1.1664')


# Plot the linear fit
plotData(x,np.dot(X,theta), "Prediction line", marker="-")

#Predict values for population sizes of 35,000 and 70,000
predict1=np.dot(np.array([1,3.5]),theta)
print('For population = 35,000, we predict a profit of %f\n',predict1*10000)
predict2=np.dot(np.array([1,7]),theta)
print('For population = 70,000, we predict a profit of %f\n',predict2*10000)

# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...')
theta0_vals=np.linspace(-10,10,100,dtype=float)
theta1_vals=np.linspace(-1,4,100,dtype=float)

j_vals=np.zeros((len(theta0_vals),len(theta1_vals)),dtype=float)

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t=np.array([[theta0_vals[i]],[theta1_vals[j]]])
        j_vals[i,j]=computeCost(x,y,t)

#theta0_vals_expanded,theta1_vals_expanded,j_vals_expanded=expandForGraph(theta0_vals,theta1_vals,j_vals)
plt.figure(2)
ax = plt.axes(projection='3d')
ax.plot_surface(theta0_vals,theta1_vals, j_vals,rstride=1, cstride=1,
                cmap=cm.coolwarm, edgecolor='none')
ax.set_title("Cost function")
ax.set_xlabel('\u03F4 0')
ax.set_ylabel('\u03F4 1')
ax.set_zlabel('J');
plt.figure(1)
plt.show()



