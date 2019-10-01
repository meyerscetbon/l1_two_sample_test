import numpy as np
import l1_two_sample_test

dim = 2
num_sample_test = 2000
X = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), 2 * num_sample_test)

mean_shift = np.zeros(dim)
mean_shift[0] = 1
Y = np.random.multivariate_normal(mean_shift, np.eye(dim), 2 * num_sample_test)

# Split X, Y into train and test sets
Itr = np.zeros(2 * num_sample_test, dtype=bool)
tr_ind = np.random.choice(2 * num_sample_test, int(num_sample_test), replace=False)
Itr[tr_ind] = True
Ite = np.logical_not(Itr)

X_tr, Y_tr = X[Itr, :], Y[Itr, :]
X_te, Y_te = X[Ite, :], Y[Ite, :]

# L1_opt_J_ME
test_locs, gwidth2 = l1_two_sample_test.solver_ME(X_tr, Y_tr)
test = l1_two_sample_test.test_asymptotic_ME(X_te, Y_te, test_locs, gwidth2)
print(test)

# L1_opt_J_SCF
test_locs, gwidth2 = l1_two_sample_test.solver_SCF(X_tr, Y_tr)
test = l1_two_sample_test.test_asymptotic_SCF(X_te, Y_te, test_locs, gwidth2)
print(test)
