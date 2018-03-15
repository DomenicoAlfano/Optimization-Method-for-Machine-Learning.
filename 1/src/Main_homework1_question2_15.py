from Functions_homework1_question2_15 import *


################################# MLP #########################################

#Optimal parameters
num_units = 20

ro = 1e-5

X_train, Y_train, X_test, Y_test = create_dataset(100,2)

MSE_test,obj_value, comp_time = run_MLP_Extreme_Learning(X_train, Y_train, X_test, Y_test, num_units, ro)

file_ = open("output.txt",'a')

file_.write("Homework1, question 2, point 1\n")
file_.write("Training objective function,{}\n".format(obj_value))
file_.write("Test MSE,{}\n".format(MSE_test))
file_.write("Training computing time,{} \n".format(comp_time))
file_.write("Num. function evaluation, \n")
file_.write("Num. gradient evaluation, \n")
########################################## RBF ##################################

#Optimal parameters
num_units = 5

ro = 1e-5

sigma = 0.3

MSE_test, obj_value,f_eval, grad_eval,_, comp_time = run_RBF_Unsupervised_Centers(X_train, Y_train, X_test, Y_test, num_units, ro, sigma)


file_.write("Homework1, question 2, point 2\n")
file_.write("Training objective function,{}\n".format(obj_value))
file_.write("Test MSE,{}\n".format(MSE_test))
file_.write("Training computing time,{} \n".format(comp_time))
file_.write("Num. function evaluation,{}\n".format(f_eval))
file_.write("Num. gradient evaluation,{}\n".format(grad_eval))
file_.close()
