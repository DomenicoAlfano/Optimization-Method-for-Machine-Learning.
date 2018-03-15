from Functions_homework2_15 import *


X_train, Y_train, X_test, Y_test = import_dataset()

###################### Grid Search ############################

Gammas = [0.001,0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 0.7]

Cs = [0.001, 0.01, 0.1, 1., 10., 100.]

#C, gamma = grid_search(Cs, Gammas, X_train, Y_train, X_test, Y_test)

gamma = 0.3
C = 1.0

###################### Question 1 #############################


solver_full = SVM_solver(X=X_train, Y=Y_train, C=C, gamma=gamma)

obj_value, iterations, acc, comp_time = solver_full.optimize(X_test, Y_test)

print("C : {} -- gamma : {} --Obj Value : {} -- Accuracy : {} -- Iter : {} -- Time : {}".format(C, gamma, obj_value, acc, iterations, comp_time))

file_ = open("output.txt",'w')
file_.write("Homework 2, question 1\n")
file_.write("Training objective function,{}\n".format(obj_value))
file_.write("Test accuracy (decimal number between 0 and 1),{}\n".format(acc))
file_.write("Training computing time,{}\n".format(comp_time))
file_.write("Function evaluations,{}\n".format(iterations))
file_.write("Gradient evaluations,{}\n".format(iterations))


###################### Question 2 #############################


q = 10

num_points = len(Y_train)

lamda = np.zeros((num_points,1))

solver_dec = SVM_decMethod(X=X_train, Y=Y_train, C=C, gamma=gamma)

lamda, acc, obj_value, iterations, comp_time = solver_dec.optimize(lamda, q, X_test, Y_test)

print("Obj Value : {} -- Accuracy : {} -- Iter : {} -- Time : {}".format(obj_value, acc, iterations, comp_time))

file_ = open("output.txt",'a')
file_.write("Homework 2, question 2\n")
file_.write("Training objective function,{}\n".format(obj_value))
file_.write("Test accuracy (decimal number between 0 and 1),{}\n".format(acc))
file_.write("Training computing time,{}\n".format(comp_time))
file_.write("Function evaluations,{}\n".format(iterations))
file_.write("Gradient evaluations,{}\n".format(iterations))


###################### Question 3 #############################


lamda = np.zeros((num_points,1))

solver_mvp = SVM_MVP_solver(X=X_train, Y=Y_train, C=C, gamma=gamma)

lamda, acc, obj_value, iterations, comp_time = solver_mvp.optimize(X_test, Y_test, lamda)

print("Obj Value : {} -- Accuracy : {} -- Iter : {} -- Time : {}".format(obj_value, acc, iterations, comp_time))

file_ = open("output.txt",'a')
file_.write("Homework 2, question 3\n")
file_.write("Training objective function,{}\n".format(obj_value))
file_.write("Test accuracy (decimal number between 0 and 1),{}\n".format(acc))
file_.write("Training computing time,{}\n".format(comp_time))
file_.write("Function evaluations,{}\n".format(iterations))
file_.write("Gradient evaluations,{}\n".format(iterations))


###############################################################
