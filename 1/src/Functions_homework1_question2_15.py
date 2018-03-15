import tensorflow as tf
import numpy as np
import scipy.optimize as opt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
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

        # initialize all variables belong to the MLP

        self.N = N

        self.lamda = lamda #Regularizzation factor

        self.gradient = None

        self.num_features = num_features

        self.X = tf.placeholder(tf.float32, shape=[None,self.num_features])

        self.Y = tf.placeholder(tf.float32, shape=[None,1])

        self.W = None #weights

        self.V = None #output_weights

        self.predict = None

        self.MSE = None

        self.loss = None

        self.opt = None

        self.opt_dec = None

    def build_net(self):

        #Divide the variables in two main group, this is usefull only in the
        # block methods

        with tf.variable_scope("hidden"):

            W = tf.truncated_normal(shape=[self.num_features, self.N],seed=20) #initialize randomly

            self.W = tf.Variable(initial_value= W)

        with tf.variable_scope("output"):

            V = tf.truncated_normal(shape=[self.N,1],seed=20) #initialize randomly

            self.V = tf.Variable(initial_value=V)

        #Compute the output of the network

        hidden = tf.matmul(self.X, self.W)

        g = tf.tanh(hidden)

        output = tf.matmul(g ,self.V)

        self.predict = output

    def MSE_(self):

        #Mean square error

        self.MSE = tf.reduce_mean(tf.square(self.predict - self.Y))

    def loss_(self):

        #Regularizzed square loss

        self.MSE_()

        self.loss = self.MSE + self.lamda * (tf.reduce_sum(tf.square(self.W)) + tf.reduce_sum(tf.square(self.V)))

    def optimize_(self, flag):

        # Compute the gradient explicitly of the loss, this is usefull only in the question 3

        gradient = tf.gradients(self.loss, [self.W, self.V])

        self.gradient = tf.sqrt(tf.reduce_sum(tf.square(gradient[0])) + tf.reduce_sum(tf.square(gradient[1])))

        if flag == "block":

            # Extract the variables belonging to a specific group from the graph

            hidden_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="hidden")

            output_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="output")

            # Choose a different optimizer for each group of variables

            self.opt = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                var_list = hidden_weights, method="BFGS",
                options={"ftol": 1e-15,"gtol" : 1e-15})

            self.opt_dec = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                var_list = output_weights, method="CG",
                options={'disp' : True})

        if flag == "global":

            #Here we optimize over all the trainable variable in the graph

            self.opt = tf.contrib.opt.ScipyOptimizerInterface(self.loss, method="L-BFGS-B")

    def initializer(self, flag="global"):

        #Build the network, loss and optimizer graph in tensorflow

        self.build_net()
        self.loss_()
        self.optimize_(flag)

    def optimize(self, sess, x, y):

        self.opt.minimize(sess, feed_dict={self.X : x, self.Y : y})

    def block_optimize(self, sess, x, y):

        self.opt_dec.minimize(sess, feed_dict={self.X : x, self.Y : y})

    def regularized_error(self,sess,x,y):

        return sess.run(self.loss, feed_dict={self.X : x, self.Y : y})

    def gradient_norm(self,sess, x, y):

        return sess.run(self.gradient, feed_dict={self.X : x, self.Y : y})

    def MeanSquareError(self,sess, x, y):

        return sess.run(self.MSE, feed_dict={self.X : x, self.Y : y})

class RBF:

    def __init__(self,X, Y, sigma, ro, N,centers=None):

        self.X = X
        self.Y = Y

        self.P = X.shape[0]

        self.features = X.shape[1]

        self.sigma = sigma

        self.centers = centers

        self.ro = ro #Regularizzed factor

        self.N = N

    def gaussian_function(self,x):

        return np.exp(-(x / self.sigma)**2)

    def RBF_net(self, omega):

        # Given the vector of parameters, build a matrix C that contain, in each
        # row, a specific center and the vector of output weights

        # Determine which variable set as to be optimized looking at the value
        # of the attribute in the class

        if self.centers is None:

            c = omega[:-self.N]

            C = c.reshape(self.N,self.features)

            V = omega[-self.N:]

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

    def reg_err(self, omega):

        #Regularizzed square loss

        empirical_risk = self.MSE(omega)

        reg_term = self.ro * np.sum(np.square(omega.ravel()))

        return empirical_risk + reg_term

    def optimize(self, method, omega0):

        #Run the optimization routine

        theta = opt.minimize(self.reg_err, x0 = omega0, method  = method, options={'disp' : True})

        return theta

def train_RBF(model, omega0):

    # Optimize the parameters and compute training time

    start = time.time()

    omega = model.optimize(method='CG',omega0=omega0)

    comp_time = time.time() - start

    return omega['x'], omega['nfev'],omega['njev'],omega['nit'],comp_time

def run_MLP_Extreme_Learning(X_train, Y_train, X_test, Y_test, N, ro):

    # Add dump feature to include biases in the model

    X_train_MLP = np.hstack((X_train,np.ones((X_train.shape[0],1))))

    X_test_MLP = np.hstack((X_test,np.ones((X_test.shape[0],1))))

    # Construct the model and initialize graph and variables

    mlp = MultiLayerPerceptron(3, N, ro)

    mlp.initializer(flag='block')

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        start = time.time()

        # Run the optimization only for the output weights

        mlp.block_optimize(sess, X_train_MLP, Y_train)

        comp_time = time.time() - start

        obj_value= mlp.regularized_error(sess, X_train_MLP, Y_train)

        MSE_test = mlp.MeanSquareError(sess, X_test_MLP, Y_test)

        gradient_norm = mlp.gradient_norm(sess, X_train_MLP, Y_train)

        W = sess.run(mlp.W)

        V = sess.run(mlp.V)

        print("MLP ---- > MSE train : {} -- MSE test : {} -- time : {} s".format(obj_value, MSE_test, comp_time))

        #plotter_MLP(W,V)

        sess.close()

    return MSE_test, obj_value, comp_time

def run_RBF_Unsupervised_Centers(X_train, Y_train, X_test, Y_test, N, ro, sigma):

    #Compute the RBF centers using K-means

    kmeans=KMeans(n_clusters=N).fit(X_train)

    centers = kmeans.cluster_centers_

    #Initialize the model and the initial guess of the output weights

    rbf = RBF(X=X_train, Y=Y_train, sigma=sigma, ro=ro, N=N, centers=centers)

    gen = np.random

    gen.seed(100)

    omega0=gen.rand(N)

    # Run the optimizer

    omega, f_eval, grad_eval,iter_, comp_time = train_RBF(rbf, omega0)

    # Create another model for evaluate the MSE on test

    rbf_test = RBF(X=X_test, Y=Y_test, sigma=sigma, ro=ro, N=N, centers=centers)

    MSE_test = rbf_test.MSE(omega)

    obj_value = rbf.reg_err(omega)

    #plotter_RBF(omega, centers,np.sqrt(0.3))

    print("RBF ---- > MSE train : {} -- MSE test : {} -- time : {} s".format(obj_value,MSE_test,comp_time))

    return MSE_test,obj_value, f_eval, grad_eval,iter_, comp_time
