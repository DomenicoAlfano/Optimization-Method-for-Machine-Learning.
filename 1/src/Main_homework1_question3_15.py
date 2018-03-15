from Functions_homework1_question3_15 import *

iterations = 100

num_units = 20
ro = 1e-5

X_train, Y_train, X_test, Y_test = create_dataset(100,2)

MSE_test, obj_value,comp_time, iterations_MLP =run_MLP_block_wise(X_train, Y_train, X_test, Y_test, num_units, ro, iterations)



file_ = open("output.txt",'a')

file_.write("Homework1, question 3, MLP\n")
file_.write("Training objective function,{}\n".format(obj_value))
file_.write("Test MSE,{}\n".format(MSE_test))
file_.write("Training computing time,{} \n".format(comp_time))
file_.write("Num. function evaluation, \n")
file_.write("Num. gradient evaluation,n.a.\n")

########################### RBF ########################àà

num_units = 5

iterations = 100

ro = 1e-5

sigma = 0.3

MSE_test,obj_value, n_fev,n_grad, comp_time, iterations_RBF = run_RBF_block_wise(X_train, Y_train, X_test, Y_test, num_units  , ro, sigma, iterations)

file_.write("Homework1, question 3, RBF\n")
file_.write("Training objective function,{}\n".format(obj_value))
file_.write("Test MSE,{}\n".format(MSE_test))
file_.write("Training computing time,{} \n".format(comp_time))
file_.write("Num. function evaluation,{}\n".format(n_fev))
file_.write("Num. gradient evaluation,n.a.\n")
file_.close()

print("Num. outer iterations MLP : {} -- Num. outer iterations RBF : {}".format(iterations_MLP+1,iterations_RBF+1))
