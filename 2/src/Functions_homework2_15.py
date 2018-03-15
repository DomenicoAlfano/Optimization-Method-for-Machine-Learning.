import numpy as np
import scipy.optimize as opt
from cvxopt import matrix, solvers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time


def import_dataset():

    path='2017.12.11 Dataset Project 2.csv'

    my_data = pd.read_csv(path, header=1)

    D = my_data.values

    # Extract the labels

    Y = D.T[-1]

    # Replace 0 with -1 in the labels

    Y[ Y==0 ] = -1

    X = D.T[:-1].T

    # Initializzation of the scaler
    # The transformation applied is (x - mu)/sigma

    scaler = StandardScaler()

    # Split the dataset (70% train , 30% test)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Computation of mu and sigma on the training set

    scaler.fit(X_train)

    # Here we apply the corresponding transformation to both
    # the training and the test set

    X_train_scaled = scaler.transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, Y_train, X_test_scaled, Y_test

########################################### Question 1 #############################################################

class SVM_solver(object):

    def __init__(self, X, Y, C, gamma):

        # Set the dataset as internal attributes of the class

        self.X = X

        self.Y = Y

        # Initilizze the hyperparameter

        self.C = C

        self.gamma = gamma

        # Initializze the Hessian matrix and the e vector

        self.Q = np.zeros((Y.shape[0],Y.shape[0]))

        self.e = -1.0*np.ones((self.Y.shape[0],1))

        # Attribute for the optimal lamdas and the bias

        self.lamda_star = None

        self.b = 0

    def RBF_Kernel(self,x_i, x_j):

        # Gaussian Kernel

        return np.exp(-self.gamma*np.linalg.norm(x_i - x_j)**2)

    def hessian_matrix_init(self):

        # Initializze the kernel matrix

        K = np.zeros((self.Y.shape[0],self.Y.shape[0]))

        for i,x_i in enumerate(self.X):

            for j,x_j in enumerate(self.X):

                K[i][j] = self.RBF_Kernel(x_i,x_j)

        # Build the hessian

        y = np.diag(self.Y)

        self.Q = y.dot(K).dot(y)

    def predict(self,x):

        # Compute the predicted label for a specific data point x

        t = 0

        for idx,point in enumerate(self.X):

            t += self.lamda_star[idx]*self.Y[idx]*self.RBF_Kernel(point, x)

        return np.sign(t + self.b)

    def accuracy(self,X_test, Y_test):

        # Compute the accuracy

        cc = 0

        num_points = Y_test.shape[0]

        for idx, label in enumerate(Y_test):

            t = self.predict(X_test[idx])

            if t*label > 0:

                cc += 1

        return cc/num_points

    def optimize(self, X_test, Y_test):

        self.hessian_matrix_init()

        # Build the required matricies for the cvxopt optimizer

        P = matrix(self.Q,tc='d') # Hessian

        c = matrix(self.e,tc='d') # e vector

        A = matrix(self.Y,(1, self.Y.shape[0]),tc='d') # Equality constrain

        b = matrix(0.0)

        G = matrix(np.vstack((-np.eye(self.Y.shape[0]),np.eye(self.Y.shape[0]))),tc='d') # Inequality constrains

        h = matrix(np.hstack((np.zeros(self.Y.shape[0]),self.C*np.ones(self.Y.shape[0]))))

        start_time = time.time()

        # Start the solver

        x = solvers.qp(P,c,G,h,A,b)

        comp_time = time.time() - start_time

        # Compute the bias for the classifier

        self.lamda_star = np.asarray(x['x'])

        support_vector_idx = self.lamda_star.argmax()

        b_x = self.X[support_vector_idx]

        b_y = self.Y[support_vector_idx]

        self.b = (1 - b_y*self.predict(b_x))/b_y

        # Compute the accuracy on the test set

        test_accuracy = self.accuracy(X_test, Y_test)

        return x['primal objective'], x['iterations'], test_accuracy, comp_time


def grid_search(Cs, Gammas, X_train, Y_train, X_test, Y_test):

    # Initializze the grid

    accuracy = np.zeros((len(Cs), len(Gammas)))

    for idx_gamma, gamma in enumerate(Gammas):
        for idx_C, C in enumerate(Cs):

            solve = SVM_solver(X=X_train, Y=Y_train, C=C, gamma=gamma)

            solve.hessian_matrix_init()

            _, _, acc, _ = solve.optimize(X_test, Y_test)

            print("gamma : {} -- C : {} -- test accuracy : {} -- train accuracy : {}".format(solve.gamma, solve.C, acc, solve.accuracy(X_train, Y_train)))

            accuracy[idx_C][idx_gamma] = acc

    # Take the indecies that correspond to the higher accuracy on the test set

    i, j = np.unravel_index(accuracy.argmax(), accuracy.shape)

    return Cs[i], Gammas[j]

#################################################### Question 2 ###################################################################


class SVM_decMethod(object):

    def __init__(self, X, Y, C, gamma):

        # Set the dataset as internal attributes of the class

        self.X = X

        self.Y = Y

        # Initilizze the hyperparameter

        self.C = C

        self.gamma = gamma

        # Initializze the Hessian matrix and the e vector

        self.Q = np.zeros((Y.shape[0],Y.shape[0]))

        self.e = -1.0*np.ones((self.Y.shape[0],1))

        # Attribute for the optimal lamdas and the bias

        self.lamda_star = None

        self.b = 0

    def RBF_Kernel(self,x_i, x_j):

        # Gaussian Kernel

        return np.exp(-self.gamma*np.linalg.norm(x_i - x_j)**2)

    def hessian_matrix_init(self):

        # Initilizze the kernel matrix

        K = np.zeros((self.Y.shape[0],self.Y.shape[0]))

        for i,x_i in enumerate(self.X):

            for j,x_j in enumerate(self.X):

                K[i][j] = self.RBF_Kernel(x_i,x_j)

        # Build the hessian

        y = np.diag(self.Y)

        self.Q = y.dot(K).dot(y)

    def objective_function(self):

        # Compute the value of the objective given lamda star

        F = 1/2*self.lamda_star.T.dot(self.Q).dot(self.lamda_star) + self.e.T.dot(self.lamda_star)

        return F[0][0]

    def update_gradient(self,lamda_k_1, lamda_k, working_set):

        # Update the gradient of the function based on the working set of the current iterations

        # Extract the colomns of the hessian that correspond to the working set

        Q_w_colomns = self.Q.T[np.ix_(working_set)].T

        # Compute the difference between the lamdas component in the working set

        delta_lamda = lamda_k_1[working_set] - lamda_k[working_set]

        # Compute the increment of the gradient

        delta_gradient = Q_w_colomns.dot(delta_lamda)

        return delta_gradient

    def predict(self,x):

        # Compute the predicted label for a specific data point x

        t = 0

        for idx,point in enumerate(self.X):

            t += self.lamda_star[idx]*self.Y[idx]*self.RBF_Kernel(point, x)

        return np.sign(t + self.b)

    def accuracy(self,X_test, Y_test):

        # Compute the accuracy on a given test set

        cc = 0

        num_points = Y_test.shape[0]

        for idx, label in enumerate(Y_test):

            t = self.predict(X_test[idx])

            if t*label > 0:

                cc += 1

        return cc/num_points

    def build_matrix(self, working_set, not_working_set, lamda):

        # Compute the required quantities for the cvxopt solver

        Y = self.Y.reshape(len(self.Y),1)

        # Hessian matrix and e vector of the subproblem

        Q_ww = self.Q[np.ix_(working_set,working_set)]

        e_w = self.e[working_set]

        lamda_not_w = lamda[not_working_set]

        Q_not_w_w = self.Q[np.ix_(not_working_set,working_set)]

        # Slit the label

        Y_w = Y[working_set]

        Y_not_w = Y[not_working_set]

        # Build the matricies for the optimizer

        P = matrix(Q_ww,tc='d')

        c = matrix((lamda_not_w.T.dot(Q_not_w_w) + e_w.T).T,tc='d')

        A = matrix(Y_w,(1, Y_w.shape[0]),tc='d') # Equality constrain

        b = matrix(-Y_not_w.T.dot(lamda_not_w), tc='d')

        G = matrix(np.vstack((-np.eye(Y_w.shape[0]),np.eye(Y_w.shape[0]))),tc='d') # Inequality Constrains

        h = matrix(np.hstack((np.zeros(Y_w.shape[0]),self.C*np.ones(Y_w.shape[0]))),tc='d')

        return P,c,G,h,A,b

    def qp_solver(self, working_set, not_working_set, lamda):

        P,c,G,h,A,b = self.build_matrix(working_set, not_working_set, lamda)

        x = solvers.qp(P,c,G,h,A,b)

        return np.array(x['x']), x['iterations']

    def optimize(self, lamda, q, X_test, Y_test):

        Y = self.Y.reshape(len(self.Y),1)

        # Initilizze hessian and gradient

        iterations = 0

        self.hessian_matrix_init()

        grad = np.copy(self.e)

        start_time = time.time()

        while True:

            lamda_k = np.copy(lamda)

            grad_y = -grad/Y # Point-wise division

            # Index set

            idx = np.arange(0,len(Y)).reshape(len(Y),1)

            # Build the condition for the required set as logical arrays

            # In this case we need to introduce some tollerance in the conditions
            # due to the numerical approximation of lamda in the algorithm

            tollerance = 1e-5

            lamda_L = lamda <= tollerance

            lamda_U = np.logical_and(lamda >= (self.C - tollerance), lamda <= self.C)

            # Build the logical arrays

            L_plus_condition = np.logical_and(lamda_L,Y==1)

            L_minus_condition = np.logical_and(lamda_L,Y==-1)

            U_plus_condition = np.logical_and(lamda_U,Y==1)

            U_minus_condition = np.logical_and(lamda_U,Y==-1)

            F_condition = np.logical_and(lamda > tollerance,lamda < (self.C - tollerance))

            # Extract the corresponding indicies from the index set

            L_plus = list(idx[L_plus_condition])

            L_minus = list(idx[L_minus_condition])

            U_plus = list(idx[U_plus_condition])

            U_minus = list(idx[U_minus_condition])

            F = list(idx[F_condition])

            # Build R(lamda_k) and S(lamda_k)

            R = sorted(L_plus + U_minus + F)

            S = sorted(L_minus + U_plus + F)

            # Compute m_lamda_k and M_lamda_k

            m_lamda = grad_y[R].max()

            M_lamda = grad_y[S].min()

            # Terminal condition equivalent to m_lamda - M_lamda < epsillon
            # the condition must be satisfied at least at the 3 decimal
            # with a round approximation.

            if round(m_lamda,3) <= round(M_lamda,3):

                print("Optimization finished : K.K.T. point reached")

                break

            else:

                # Take the indicies that correspond to the higher/lower value of
                # -grad_i/y_i in the index set R and S respectivly

                # Sort the indicies of the two array by the values contained

                max_grad_y_R = (grad_y[R].ravel()).argsort()[:][::-1]

                min_grad_y_S = (grad_y[S].ravel()).argsort()[:]

                # Take q/2 indicies for each set

                max_grad_y = max_grad_y_R[0:int(q/2)]

                min_grad_y = min_grad_y_S[0:int(q/2)]

                # Build the working set

                I = [R[i] for i in max_grad_y]

                J = [S[j] for j in min_grad_y]

                working_set = I + J

                # Take the rest of the indicies

                idx_set = list(idx.ravel())

                not_working_set = [x for x in idx_set if x not in working_set]

                # Solve the subproblem

                lamda_w, num_iter = self.qp_solver(working_set, not_working_set, lamda_k)

                # Update lamda and the gradient

                lamda[working_set] = np.copy(lamda_w)

                delta_grad = self.update_gradient(lamda, lamda_k, working_set)

                grad += delta_grad

                # Update the number of iterations considering the total number
                # of iterations that the solver perform in each subproblem

                iterations += num_iter

        comp_time = time.time() - start_time

        # Compute the bias for the classifier

        self.lamda_star = lamda

        support_vector_idx = lamda.argmax()

        b_x = self.X[support_vector_idx]

        b_y = self.Y[support_vector_idx]

        self.b = (1 - b_y*self.predict(b_x))/b_y

        # Compute the accuracy on the test set and the final value of the objective

        acc = self.accuracy(X_test, Y_test)

        obj_value = self.objective_function()

        return lamda, acc, obj_value, iterations, comp_time

####################################################### Question 3 #######################################################à


class SVM_MVP_solver(object):

    def __init__(self, X, Y, C, gamma):

        # Set the dataset as internal attributes of the class

        self.X = X

        self.Y = Y

        # Initilizze the hyperparameter

        self.C = C

        self.gamma = gamma

        # Initializze the Hessian matrix and the e vector

        self.Q = np.zeros((Y.shape[0],Y.shape[0]))

        self.e = -1.0*np.ones((self.Y.shape[0],1))

        # Attribute for the optimal lamdas and the bias

        self.lamda_star = None

        self.b = 0

    def RBF_Kernel(self,x_i, x_j):

        # Gaussian Kernel

        return np.exp(-self.gamma*np.linalg.norm(x_i - x_j)**2)

    def hessian_matrix_init(self):

        # Initilizze the kernel matrix

        K = np.zeros((self.Y.shape[0],self.Y.shape[0]))

        for i,x_i in enumerate(self.X):

            for j,x_j in enumerate(self.X):

                K[i][j] = self.RBF_Kernel(x_i,x_j)

        # Build the hessian

        y = np.diag(self.Y)

        self.Q = y.dot(K).dot(y)

    def objective_function(self):

        # Compute the value of the objective given lamda star

        F = 1/2*self.lamda_star.T.dot(self.Q).dot(self.lamda_star) + self.e.T.dot(self.lamda_star)

        return F[0][0]

    def predict(self,x):

        # Compute the predicted label for a specific data point x

        t = 0

        for idx,point in enumerate(self.X):

            t += self.lamda_star[idx]*self.Y[idx]*self.RBF_Kernel(point, x)

        return np.sign(t + self.b)

    def accuracy(self,X_test, Y_test):

        # Compute the accuracy on a given test set

        cc = 0

        num_points = Y_test.shape[0]

        for idx, label in enumerate(Y_test):

            t = self.predict(X_test[idx])

            if t*label > 0:

                cc += 1

        return cc/num_points

    def optimize(self,X_test, Y_test,lamda):

        # Initilizze hessian and gradient

        self.hessian_matrix_init()

        Y_train = self.Y.reshape(len(self.Y),1)

        iterations = 0

        grad = np.copy(self.e)

        start_time = time.time()

        while True:

            lamda_k = np.copy(lamda)

            grad_y = -grad/Y_train # Point-wise division

            # Index set

            idx = np.arange(0,len(Y_train)).reshape(len(Y_train),1)

            # Build the condition for the required set as logical arrays

            L_plus_condition = np.logical_and(lamda==0,Y_train==1)

            L_minus_condition = np.logical_and(lamda==0,Y_train==-1)

            U_plus_condition = np.logical_and(lamda==self.C,Y_train==1)

            U_minus_condition = np.logical_and(lamda==self.C,Y_train==-1)

            F_condition = np.logical_and(lamda > 0,lamda < self.C)

            # Extract the corresponding indicies from the index set

            L_plus = list(idx[L_plus_condition])

            L_minus = list(idx[L_minus_condition])

            U_plus = list(idx[U_plus_condition])

            U_minus = list(idx[U_minus_condition])

            F = list(idx[F_condition])

            # Build R(lamda_k) and S(lamda_k)

            R = sorted(L_plus + U_minus + F)

            S = sorted(L_minus + U_plus + F)

            # Compute m_lamda_k and M_lamda_k

            m_lamda = grad_y[R].max()

            M_lamda = grad_y[S].min()

            # Terminal condition equivalent to m_lamda - M_lamda < epsillon
            # the condition must be satisfied at least at the 3 decimal
            # with a round approximation.


            # We notice that even if we move the "accuracy" to the 15th decimal
            # the results doesn't change much.

            if round(m_lamda,3) <= round(M_lamda,3) :

                print("Optimization finished : K.K.T. point reached")

                break

            else:

                # Extract the index that correspond to the most violeting pair

                i = R[grad_y[R].argmax()]

                j = S[grad_y[S].argmin()]

                # Initializze the direction

                d_ij = np.zeros((len(lamda),1))

                d_ij[i] = 1/Y_train[i]

                d_ij[j] = -1/Y_train[j]

                # Build the working set

                w_set = [i,j]

                # Extract the colomns of Q that correspond to the working set

                Q_w_colomns = self.Q.T[np.ix_(w_set)].T

                # Build the submatrix Q_ww ( rows and colomns corresponding to the working set)

                Q_ww = Q_w_colomns[np.ix_(w_set)]

                d_ij_non_zero = d_ij[w_set]

                # Compute (d_ij)^T * Q * d_ij

                a = d_ij_non_zero.T.dot(Q_ww).dot(d_ij_non_zero)

                # Linesearch

                step_size = np.zeros((2,1))

                # Determine the maximun step size

                for key,value in enumerate([i,j]):

                    if d_ij[value] == 1:

                        step_size[key] = self.C - lamda[value]

                    elif d_ij[value] == -1:

                        step_size[key] = lamda[value]

                    else:

                        return

                t_max = step_size.min()

                # Determine the maximun step size that preserve feasibility
                # and modify the t_max accordly

                if a > 0:

                    t_star = -grad[w_set].T.dot(d_ij_non_zero)/a

                    if t_star < t_max:

                        t_max = t_star

            # Update lamda and the gradient

            lamda += t_max*d_ij

            delta_lamda = lamda[w_set] - lamda_k[w_set]

            grad += Q_w_colomns.dot(delta_lamda)

            iterations += 1

        comp_time = time.time() - start_time

        # Store lamda star and compute the bias for the classifier

        self.lamda_star = lamda

        support_vector_idx = lamda.argmax()

        b_x = self.X[support_vector_idx]

        b_y = self.Y[support_vector_idx]

        self.b = (1 - b_y*self.predict(b_x))/b_y

        # Compute the accuracy and the final value of the objective

        acc = self.accuracy(X_test, Y_test)

        obj_value = self.objective_function()

        return lamda, acc, obj_value, iterations, comp_time
