import tensorflow as tf
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.optimize as opt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import time


def frank_function(x):

    x1 = x[0]
    x2 = x[1]

    f0 = 0.75*np.exp(-(9*x1 - 2)**2 / 4 - (9*x2 -2)**2 /4 )

    f1 = 0.75*np.exp(-(9*x1 + 1)**2 / 49 - (9*x2 + 1)/10)

    f2 = 0.5*np.exp(-(9*x1 - 7)**2 / 4 - (9*x2 -3)**2 /4)

    f3 = - 0.2*np.exp(-(9*x1-4)**2 - (9*x2 - 7)**2)

    return  f0 + f1 + f2 +f3


def create_dataset(num_points, num_features):

    #Generate the data points

    X = np.random.rand(num_points, num_features)

    Y = np.zeros((num_points,1))

    gen = np.random

    gen.seed(100)

    for idx,point in enumerate(X) :

        # add noise to the output of the frank function

        epsilon = 0.2*gen.rand() - 0.1

        Y[idx] = frank_function(point) + epsilon

    # Split the dataset

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

    return X_train, Y_train, X_test, Y_test

def grid_MLP(num_units, lamdas):

    # Grid search function that given the lists of hyperparameters returns
    # the best values

    MSE = np.zeros((len(num_units),len(lamdas)))

    for idx,N in enumerate(num_units):
        for idx2,lamda in enumerate(lamdas):

            X_train, Y_train, X_test, Y_test = create_dataset(100,2)

            MSE_test,_, _ = run_MLP(X_train, Y_train, X_test, Y_test, N, lamda)

            MSE[idx,idx2] = MSE_test

    i,j = np.unravel_index(MSE.argmin(), MSE.shape)

    return num_units[i],lamdas[j]

def grid_RBF(num_units, lamdas, sigmas):

    # Grid search function that given the lists of hyperparameters returns
    # the best values

    MSE = np.zeros((len(num_units),len(lamdas),len(sigmas)))

    for idx,N in enumerate(num_units):
        for idx2,lamda in enumerate(lamdas):
            for idx3,gamma in enumerate(sigmas):

                X_train, Y_train, X_test, Y_test = create_dataset(100,2)

                num_features = X_train.shape[1]

                MSE_test,_,_,_,_ = run_RBF(X_train, Y_train, X_test, Y_test, N, lamda, num_features, gamma)

                MSE[idx,idx2,idx3] = MSE_test

    i,j,k = np.unravel_index(MSE.argmin(), MSE.shape)

    return num_units[i],lamdas[j],sigmas[k]


def plotter_MLP(W,V):

    # Plot function for the MLP

    X = np.arange(0,1,0.01)

    Y = np.arange(0,1,0.01)

    X, Y = np.meshgrid(X,Y)

    def MLP(X1,X2):

        dumpFeature = np.ones(1)

        X = np.hstack((X1,X2))

        X = np.hstack((X,dumpFeature))

        return np.tanh(X.dot(W)).dot(V)

    Z = np.zeros(X.shape)

    F = np.zeros(X.shape)

    for row_x,row_y,r in zip(X,Y,range(X.shape[0])):
        for i,j,c in zip(row_x,row_y,range(X.shape[1])):

            Z[r,c] = MLP(i,j)

            F[r,c] = frank_function([i,j])

    fig = plt.figure(1)

    fig2 = plt.figure(2)

    # Generating the surfaces

    ax = fig.gca(projection='3d',title="MLP Approximation")

    ax2 = fig2.gca(projection='3d',title="Frank Function")

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    surf2 = ax2.plot_surface(X, Y, F, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    fig2.colorbar(surf2, shrink=0.5, aspect=5)

    plt.show()

def plotter_RBF(V,C,sigma):

    # Plotting function for RBF

    X = np.arange(0,1,0.01)

    Y = np.arange(0,1,0.01)

    X, Y = np.meshgrid(X,Y)

    def gaussian_function(x):

        return np.exp(-(x / sigma)**2)

    def RBF(X1, X2):

        X = np.hstack((X1,X2))

        Y = V.dot(gaussian_function(np.linalg.norm(C - X,axis=1)))

        return Y

    Z = np.zeros(X.shape)

    F = np.zeros(X.shape)

    for row_x,row_y,r in zip(X,Y,range(X.shape[0])):
        for i,j,c in zip(row_x,row_y,range(X.shape[1])):

            Z[r,c] = RBF(i,j)

            F[r,c] = frank_function([i,j])

    fig = plt.figure(1)

    fig2 = plt.figure(2)

    # Generating the surfaces

    ax = fig.gca(projection='3d',title="RBF Approximation")

    ax2 = fig2.gca(projection='3d',title="Frank Function")

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    surf2 = ax2.plot_surface(X, Y, F, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    fig2.colorbar(surf2, shrink=0.5, aspect=5)

    plt.show()

class MultiLayerPerceptron():

    def __init__(self, num_features, N, lamda):

        self.N = N

        self.lamda = lamda # Regularizzation factor

        self.gradient = None

        self.num_features = num_features

        self.X = tf.placeholder(tf.float32, shape=[None,self.num_features])

        self.Y = tf.placeholder(tf.float32, shape=[None,1])

        self.W = None # Hidden weights

        self.V = None #Output weigths

        self.predict = None

        self.loss = None

        self.opt = None

        self.opt_dec = None

        # Extra set of attribute that we use to modify the stopping criteria
        # of the optimization routine for both block at each iteration

        self.gtol = 1e-5

        self.theta = 0.5

        self.iter = 0


    def build_net(self):

        #Divide the variables in two main group, this is usefull only in the
        # block methods

        with tf.variable_scope("hidden"):

            W = tf.truncated_normal(shape=[self.num_features, self.N],seed=20)

            self.W = tf.Variable(initial_value= W)

        with tf.variable_scope("output"):

            V = tf.truncated_normal(shape=[self.N,1],seed=20)

            self.V = tf.Variable(initial_value=V)

        #Compute the output of the network

        hidden = tf.matmul(self.X, self.W)

        g = tf.tanh(hidden)

        output = tf.matmul(g ,self.V)

        self.predict = output

    def MSE_(self):

        #Mean Square Error

        self.MSE = tf.reduce_mean(tf.square(self.predict - self.Y))

    def loss_(self):

        #Regularizzed loss

        self.MSE_()

        self.loss = self.MSE + self.lamda * (tf.reduce_sum(tf.square(self.W)) + tf.reduce_sum(tf.square(self.V)))


    def optimize_(self):

        # Compute gradient norm and select the optimization routines for each
        # block of variables

        gradient = tf.gradients(self.loss, [self.W,self.V])

        self.gradient = tf.sqrt(tf.square(tf.norm(gradient[0])) + tf.square(tf.norm(gradient[1])))


        hidden_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="hidden")

        output_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="output")

        # Modify the gradient tollerance of each method at every iteration
        # For the L-BFGS the criteria used is refer to the projected gradient
        # but we find that the result is good with the same value of the conjugate
        # gradient

        self.opt = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
            var_list = hidden_weights, method="L-BFGS-B",options={ "gtol":self.gtol*self.theta**self.iter})

        self.opt_dec = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
            var_list = output_weights, method="CG",options={ "gtol":self.gtol*self.theta**self.iter})

    def initializer(self):

        #Build the network, loss and optimizer graph in tensorflow

        self.build_net()
        self.loss_()
        self.optimize_()

    def optimize(self, sess, x, y):

        self.opt.minimize(sess, feed_dict={self.X : x, self.Y : y})

    def block_optimize(self, sess, x, y):

        # For every call of the method modify the number of iterations
        # and redefine the optimization routines using the new tollerance

        self.iter += 1

        self.optimize_()

        self.opt_dec.minimize(sess, feed_dict={self.X : x, self.Y : y})

    def regularized_error(self,sess,x,y):

        return sess.run(self.loss, feed_dict={self.X : x, self.Y : y})

    def gradient_norm(self,sess, x, y):

        return sess.run(self.gradient, feed_dict={self.X : x, self.Y : y})

    def MeanSquareError(self,sess, x, y):

        return sess.run(self.MSE, feed_dict={self.X : x, self.Y : y})

class RBF:

    def __init__(self,X, Y, sigma, ro, N,centers=None, output_weights=None):

        self.X = X
        self.Y = Y

        self.P = X.shape[0]

        self.features = X.shape[1]

        self.sigma = sigma

        self.centers = centers

        self.V = output_weights

        self.ro = ro

        self.N = N

        # Extra set of attribute that we use to modify the stopping criteria
        # of the optimization routine for both block at each iteration

        self.gtol = 1e-5

        self.theta = 0.5

        self.iter = 0

    def gaussian_function(self,x):

        return np.exp(-(x / self.sigma)**2)

    def RBF_net(self, omega):

        # Determine which variable set as to be optimized looking at the value
        # of the attribute in the class

        if self.centers is None:

            c = omega

            C = c.reshape(self.N,self.features)

            V = self.V

        else:

            C = self.centers

            V = omega

        Y = np.zeros(self.P)

        # Compute the output of the network

        for i in range(self.P):

            Y[i] = V.dot(self.gaussian_function(np.linalg.norm(C - self.X[i],axis=1)))

        return Y

    def MSE(self, omega):

        # Mean square error

        return np.mean(np.square(self.RBF_net(omega) - self.Y.ravel()))

    def grad_centers(self,omega):

        # Compute the gradient with respect to the centers

        diff = (self.RBF_net(omega) - self.Y.ravel())

        grad = np.zeros((self.N,self.features))

        C = omega.reshape(self.N,self.features)

        for center_idx, center in enumerate(C):

            grad_center_j = 0

            for i, element in enumerate(diff):

                grad_center_j += element * self.V[center_idx]*self.gaussian_function(np.linalg.norm(self.X[i] - center))*2*(self.X[i] - center)/self.sigma**2

            grad[center_idx] = (2/self.P)*grad_center_j + 2*self.ro*center

        return grad.ravel()

    def grad_output(self,omega):

        # Compute the gradient with respect to the output weights

        diff = (self.RBF_net(omega) - self.Y.ravel())

        grad = np.zeros(omega.shape)

        for idx,center in enumerate(self.centers):

            for point_idx,point in enumerate(self.X):

                grad[idx] += diff[point_idx] * self.gaussian_function(np.linalg.norm(point - center))

            grad[idx] *= -(1/self.P)

            grad[idx] += 2*self.ro*omega[idx]

        return grad.ravel()

    def reg_err(self, omega):

        #Regularizzed square loss

        empirical_risk = self.MSE(omega)

        reg_term = self.ro * np.sum(np.square(omega))

        return empirical_risk + reg_term

    def optimize(self, method, omega0, non_linear):

        # For every call of the function (in the non linear optimization setting)
        #increase the number of iter for modifying the gradient tollerance in
        #the optimization routines

        if non_linear:

            self.iter += 1

            # Use the gradient only in the non-linear part

            theta = opt.minimize(self.reg_err, x0 = omega0, method  = method, jac=self.grad_centers, options={'disp' : False,"gtol":self.gtol*self.theta**self.iter})

        else:

            theta = opt.minimize(self.reg_err, x0 = omega0, method  = method, jac=None, options={'disp' : False,"gtol":self.gtol*self.theta**self.iter})

        return theta

    def test (self,X_test, Y_test, omega):

        self.P = X_test.shape[0]

        self.X = X_test

        self.Y = Y_test

        return self.MSE(omega)

def train_RBF(model, method, omega0, non_linear):

    # Optimize the parameters and compute training time

    start = time.time()

    omega = model.optimize(method=method,omega0=omega0,non_linear=non_linear)

    comp_time = time.time() - start

    if method == 'CG':

        grad_eval = omega['njev']

    else:

        grad_eval = omega['nit']

    return omega['x'],omega['nfev'],grad_eval,comp_time

def run_MLP_block_wise(X_train, Y_train, X_test, Y_test, N, ro, iterations):

    # Add dump feature to include biases in the model

    X_train = np.hstack((X_train,np.ones((X_train.shape[0],1))))

    X_test = np.hstack((X_test,np.ones((X_test.shape[0],1))))

    # Construct the model and initialize graph and variables

    mlp = MultiLayerPerceptron(3, N, ro)

    mlp.initializer()

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        start = time.time()

        for i in range(iterations):

            # Perform the two block optimizzation

            mlp.block_optimize(sess, X_train, Y_train)

            mlp.optimize(sess, X_train, Y_train)

            obj_value = mlp.regularized_error(sess, X_train, Y_train)

            MSE_test = mlp.MeanSquareError(sess, X_test, Y_test)

            gradient_norm = mlp.gradient_norm(sess, X_train, Y_train)

            W = sess.run(mlp.W)

            V = sess.run(mlp.V)

            print("MSE train : {} -- MSE test : {} -- Gradient norm : {}".format(obj_value, MSE_test, gradient_norm))

            # Apply stopping criteria

            if gradient_norm < 4e-4:

                break

        comp_time = time.time() - start

        #plotter_MLP(W,V)

        sess.close()

    return MSE_test, obj_value, comp_time, i

def run_RBF_block_wise(X_train, Y_train, X_test, Y_test, N, ro, sigma, iterations):

    num_features = X_train.shape[1]

    # Compute initial guess of the center using K-means

    kmeans=KMeans(n_clusters=N).fit(X_train)

    centers = kmeans.cluster_centers_

    start = time.time()

    gen = np.random

    gen.seed(100)

    V = gen.rand(N)

    # Accumulators for the number of function and gradient evaluation

    n_fev = 0

    n_gev = 0

    for i in range(iterations):

        # Build the class for the output weights optimization, and compute the
        # required quantities

        rbf_V = RBF(X_train, Y_train, sigma, ro, N, centers=centers, output_weights=None)

        omega, fev,gradev, comp_time = train_RBF(rbf_V,'CG',V,non_linear=False)

        n_fev += fev

        n_gev += gradev

        V = omega

        # Build the class for the centers optimization, and compute the
        # required quantities

        rbf_C = RBF(X_train, Y_train, sigma, ro, N, centers=None, output_weights=V)

        omega, fev , gradev, comp_time = train_RBF(rbf_C,'L-BFGS-B',centers,non_linear=True)

        n_fev += fev

        n_gev += gradev

        centers = omega.reshape(N,num_features)

        # We need to rebuild this classes for compute the MSE on test and the
        # full objective value on the train

        rbf_test = RBF(X_test, Y_test, sigma, ro, N, centers=centers, output_weights=None)

        MSE_test = rbf_test.MSE(V)

        rbf_train = RBF(X_train, Y_train, sigma, ro, N, centers=centers, output_weights=None)

        obj_value = rbf_train.reg_err(V)

        print("RBF ---- > MSE train : {} -- MSE test : {} -- time : {} s".format(obj_value,MSE_test,comp_time))

        # Compute the gradient norm over all the variables

        grad_c = rbf_C.grad_centers(centers)

        grad_v = rbf_train.grad_output(V)

        grad = np.sqrt(np.linalg.norm(grad_c)**2 + np.linalg.norm(grad_v)**2)

        if grad < 8e-4 :

            break

    comp_time = time.time() - start

    #plotter_RBF(V, centers, sigma)

    return MSE_test,obj_value, n_fev,n_gev, comp_time, i
