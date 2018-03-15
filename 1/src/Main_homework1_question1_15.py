import numpy as np
from Functions_homework1_question1_15 import *
from sklearn.cluster import KMeans

################################# MLP #########################################

#num_units = [5,7,10,15,20,25,30]

#rho = [1e-5,5e-5,1e-4,1e-3]

X_train, Y_train, X_test, Y_test = create_dataset(100,2)

N_star = 20

rho_star = 1e-5

#N_star, rho_star = grid_MLP(num_units,rho)

MSE_test, obj_value,comp_time = run_MLP(X_train, Y_train, X_test, Y_test, N_star, rho_star)

file_ = open("output.txt",'w')

file_.write("Homework1, question 1, point 1\n")
file_.write("Training objective function,{}\n".format(obj_value))
file_.write("Test MSE,{}\n".format(MSE_test))
file_.write("Training computing time,{} \n".format(comp_time))
file_.write("Num. function evaluation, \n")
file_.write("Num. gradient evaluation,n.a.\n")

########################################## RBF ##################################

num_features = X_train.shape[1]

#N_star,rho_star,sigma_star = grid_RBF(num_units,lamdas,sigmas)

N_star = 5

rho_star = 1e-5

sigma_star = 0.3

MSE_test, obj_value, f_eval, grad_eval, comp_time = run_RBF(X_train, Y_train, X_test, Y_test, N_star, rho_star, num_features, sigma_star)

file_.write("Homework1, question 1, point 2\n")
file_.write("Training objective function,{}\n".format(obj_value))
file_.write("Test MSE,{}\n".format(MSE_test))
file_.write("Training computing time,{} \n".format(comp_time))
file_.write("Num. function evaluation,{}\n".format(f_eval))
file_.write("Num. gradient evaluation,n.a\n")
file_.close()
