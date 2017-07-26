#The optimal values of m and b can be actually calculated with way less effort than doing a linear regression.
#this is just to demonstrate gradient descent

from numpy import *

# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    new_MSE = compute_error_for_line_given_points(new_b, new_m, points)
    #print("b, m, E: {}, {}, {}".format(new_b, new_m, new_MSE))
    return [new_b, new_m, new_MSE]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    mse_gain = 100000000
    mse_old = 10000000
    i = 0 # number of iterations
    # stops when gain in error is less than 0.1%
    while i < num_iterations and mse_gain >= 0.001:
        b, m, mse_new = step_gradient(b, m, array(points), learning_rate)
        mse_gain = (mse_old - mse_new) / mse_old
        #print("gain: {}".format(mse_gain))
        mse_old = mse_new
        i += 1
    return [b, m, i]

# runs gradient descent on data
def run(learning_rate):
    points = genfromtxt("data.csv", delimiter=",")
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}, learning rate = {3}".format(
        initial_b,
        initial_m,
        compute_error_for_line_given_points(initial_b, initial_m, points),
        learning_rate))
    print("Running...")
    [b, m, iterations] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(
        iterations, b, m, compute_error_for_line_given_points(b, m, points)))
    return [log(learning_rate), iterations]

# runs gradient descent several times, and store learning rate used and
# number of iterations needed
def metarun():
    total_tries = 40
    metadata = zeros(shape=(total_tries,2))
    for j in range(total_tries):
        learning_rate = exp(-(j*2.5/total_tries + 8))
        metadata[j] = run(learning_rate)
        print("log_learning_rate: {}, iterations needed: {} \n".format(
            metadata[j][0], metadata[j][1]))

if __name__ == '__main__':
    metarun()
