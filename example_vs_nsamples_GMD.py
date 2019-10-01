"""
"""

import numpy as np

import l1_two_sample_test


alpha = 0.01
J = 5

method = "GMD"
dim = 100
num_samples_test = np.arange(500, 6000, 500)

labels = ["L1_opt_J_ME", "L1_grid_J_ME", "L1_opt_J_SCF", "L1_grid_J_SCF"]
num_of_tests = len(labels)


def proba_above_tresh_GMD(seed, num_sample_test, dim, num_of_tests, my=1):
    """
    Compute the type-II error in the GMD problem:
    P = N(0, I) and Q = N( (my,0,0, 000), I).
    Only the first dimension of the means  differ.
    ----------
    num_sample_test : integer
        Test sample size
    dim : integer
        Dimension of the problem
    num_of_tests : integer
        Number of l1-based tests
    my : float
        The difference between the first coordinates of the two distributions.
    Return
    -------
    tests_error : array-like, shape = [num_of_tests]
        Indicate if the tests rejected correctly the null hypothesis
    """
    tests_error = np.zeros(num_of_tests)

    np.random.seed(seed)

    X = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), 2 * num_sample_test)

    mean_shift = np.zeros(dim)
    mean_shift[0] = my
    Y = np.random.multivariate_normal(mean_shift, np.eye(dim), 2 * num_sample_test)

    Itr = np.zeros(2 * num_sample_test, dtype=bool)
    tr_ind = np.random.choice(2 * num_sample_test, int(num_sample_test), replace=False)
    Itr[tr_ind] = True
    Ite = np.logical_not(Itr)

    X_tr, Y_tr = X[Itr, :], Y[Itr, :]
    X_te, Y_te = X[Ite, :], Y[Ite, :]

    # L1_opt_J_ME
    test_locs, gwidth2 = l1_two_sample_test.solver_ME(X_tr, Y_tr, J)
    test = l1_two_sample_test.test_asymptotic_ME(X_te, Y_te, test_locs, gwidth2, alpha)
    if test["h0_rejected"] == False:
        tests_error[0] = 1

    # L_1_grid_J_ME
    test_locs, gwidth2 = l1_two_sample_test.initial2_T_gwidth2(X_tr, Y_tr, J)
    test = l1_two_sample_test.test_asymptotic_ME(X_te, Y_te, test_locs, gwidth2, alpha)
    if test["h0_rejected"] == False:
        tests_error[1] = 1

    # L1_opt_J_SCF
    test_locs, gwidth2 = l1_two_sample_test.solver_SCF(X_tr, Y_tr, J)
    test = l1_two_sample_test.test_asymptotic_SCF(X_te, Y_te, test_locs, gwidth2, alpha)
    if test["h0_rejected"] == False:
        tests_error[2] = 1


    # L1_grid_J_SCF
    test_locs, gwidth2 = l1_two_sample_test.initial4_T_gwidth2(X_tr, Y_tr, J)
    test = l1_two_sample_test.test_asymptotic_SCF(X_te, Y_te, test_locs, gwidth2, alpha)
    if test["h0_rejected"] == True:
        tests_error[3] = 1

    return tests_error


with open("l1_test_vs_nsample.csv", "w") as file:
    for num_sample_test in num_samples_test:

        result = \
            proba_above_tresh_GMD(seed=1,
                                  num_sample_test=num_sample_test,
                                  dim=dim, num_of_tests=num_of_tests)

        s1 = ",".join(str(e) for e in result)
        s = method + "," + str(num_sample_test) + "," + s1 + "\n"

        file.write(s)
        file.flush()
