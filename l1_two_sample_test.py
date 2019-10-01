"""
Implementation of the l1-based 2-sample tests, presented in

Comparing distributions: l1 geometry improves kernel two-sample testing
M Scetbon, G Varoquaux, NeurIPS 2019
"""

import numpy as np
import torch
from scipy.linalg import sqrtm


def gauss_kernel(X, test_locs, gwidth2):
    """ Compute a X.shape[0] x test_locs.shape[0] Gaussian kernel matrix
    ----------
    X : array-like, shape = [n_samples, n_features]
        Samples from distribution P
    test_locs : array-like, shape = [n_locations, n_features]
        Finite set of test locations
    gwidth2 : float
        The square Gaussian width of the Radial basis function kernel
    Return
    -------
    K : array-like, shape = [n_samples, n_locations]
    """
    n, d = X.shape
    D2 = (
        np.sum(X ** 2, 1)[:, np.newaxis]
        - 2 * np.dot(X, test_locs.T)
        + np.sum(test_locs ** 2, 1)
    )
    K = np.exp(-D2 / (2.0 * gwidth2))
    return K


def sf_naka(J, x):
    """ Survival function (also defined as 1 - cdf) of Naka(0.5,1,J)
    ----------
    x : float
    J : integer
    Return
    -------
    res : float
    """
    mean = np.zeros(J)
    cov = np.eye(J)
    X = np.random.multivariate_normal(mean, cov, 1000000)
    S = np.sum(np.abs(X), 1)
    n = np.shape(S)[0]
    m = np.shape(S[S > x])[0]
    res = m / n
    return res


def test_asymptotic_ME(X, Y, test_locs, gwidth2, alpha):
    """ L1-based two-sample test using the mean embeddings
    functions, evaluated at a finite set of test locations.
    Use Gaussian kernel.
    ----------
    X : array-like, shape = [n_samples_1, n_features]
        Samples from distribution P
    Y : array-like, shape = [n_samples_2, n_features]
        Samples from distribution Q
    test_locs : array-like, shape = [n_locations, n_features]
        Finite set of test locations
    gwidth2 : float
        The square Gaussian width of the Radial basis function kernel
    alpha : float
        The level of the test
    Return
    -------
    results : dictionary
    {alpha: float, pvalue: float, h0_rejected: Boolean, test: float}
    """

    n, d = X.shape
    m, d = Y.shape
    J, d = test_locs.shape

    t = n + m
    ro = n / t

    z_1 = gauss_kernel(X, test_locs, gwidth2)
    cov_1 = np.cov(z_1.T)

    z_2 = gauss_kernel(Y, test_locs, gwidth2)
    cov_2 = np.cov(z_2.T)

    cov = (1 / ro) * cov_1 + (1 / (1 - ro)) * cov_2

    reg = 1e-5
    S = np.mean(z_1, axis=0) - np.mean(z_2, axis=0)
    if J > 1:
        S = np.sqrt(t) * np.linalg.solve(sqrtm(cov + reg * np.eye(J)), S)
    else:
        S = np.sqrt(t) * np.sqrt(1 / cov) * S

    S = np.sum(np.abs(S))
    p_value = sf_naka(J, S)

    results = {
        "alpha": alpha,
        "pvalue": p_value,
        "h0_rejected": p_value < alpha,
        "test": S,
    }

    return results


def dist_matrix(X, Y):
    """
    Construct a pairwise Euclidean distance matrix of
    size X.shape[0] x Y.shape[0].
    ----------
    X : array-like, shape = [n_samples_1, n_features]
        Samples from distribution P
    Y : array-like, shape = [n_samples_2, n_features]
        Samples from distribution Q
    Return
    -------
    D : array-like, shape = [n_samples_1, n_samples_2]
    """
    sx = np.sum(X ** 2, 1)
    sy = np.sum(Y ** 2, 1)
    D2 = sx[:, np.newaxis] - 2.0 * np.dot(X, Y.T) + sy[np.newaxis, :]
    # to prevent numerical errors from taking sqrt of negative numbers
    D2[D2 < 0] = 0
    D = np.sqrt(D2)
    return D


def meddistance(X, subsample=None, mean_on_fail=True):
    """
    Compute the median of pairwise distances of points
    in the matrix.
    Useful as a heuristic for setting Gaussian kernel's width.
    ----------
    X : array-like, shape = [n_samples, n_features]
        Samples from distribution P
    subsample :  float
        Subsample the samples from X
    mean_on_fail : boolean
    Return
    ------
    med: float
    """

    if subsample is None:
        D = dist_matrix(X, X)
        Itri = np.tril_indices(D.shape[0], -1)
        Tri = D[Itri]
        med = np.median(Tri)
        if med <= 0:
            # use the mean
            return np.mean(Tri)
        return med

    else:
        assert subsample > 0
        rand_state = np.random.get_state()
        np.random.seed(9827)
        n = X.shape[0]
        ind = np.random.choice(n, min(subsample, n), replace=False)
        np.random.set_state(rand_state)
        # recursion just one
        return meddistance(X[ind, :], None, mean_on_fail)


def init_locs_randn(X, Y, seed=1):
    """
    Fit a Gaussian to the merged data of the two samples and draw
    1 points from the Gaussian
    ----------
    X : array-like, shape = [n_samples_1, n_features]
        Samples from distribution P
    Y : array-like, shape = [n_samples_2, n_features]
        Samples from distribution Q
    Return
    -------
    T0 : array-like, shape = [1, n_features]
    """

    # set the seed
    rand_state = np.random.get_state()
    np.random.seed(seed)

    d = X.shape[1]
    # fit a Gaussian in the middle of X, Y and draw sample to initialize T
    xy = np.vstack((X, Y))
    mean_xy = np.mean(xy, 0)
    cov_xy = np.cov(xy.T)
    [Dxy, Vxy] = np.linalg.eig(cov_xy + 1e-3 * np.eye(d))
    Dxy = np.real(Dxy)
    Vxy = np.real(Vxy)
    Dxy[Dxy <= 0] = 1e-3
    eig_pow = 0.9  # 1.0 = not shrink
    reduced_cov_xy = Vxy.dot(np.diag(Dxy ** eig_pow)).dot(Vxy.T) + 1e-3 * np.eye(d)

    T0 = np.random.multivariate_normal(mean_xy, reduced_cov_xy, 1)
    # reset the seed back to the original
    np.random.set_state(rand_state)
    return T0


def initial2_T_gwidth2(X, Y, n_test_locs):
    """
    Fit a Gaussian to each dataset and draw half of n_test_locs from
    each. Linear search for the best Gaussian width in the
    list that maximizes the test power.
    ----------
    X : array-like, shape = [n_samples_1, n_features]
        Samples from distribution P
    Y : array-like, shape = [n_samples_2, n_features]
        Samples from distribution Q
    n_test_locs : float
        Number of test locations.
    Return
    -------
    (T0, gwidth2) : tuple
    T0 : array-like, shape = [n_test_locs, n_features]
    gwidth2: float
        The square Gaussian width of the Radial basis function kernel
    """

    n, d = X.shape
    m, d = Y.shape
    J = n_test_locs

    if n_test_locs == 1:
        T0 = init_locs_randn(X, Y)

    else:
        # fit a Gaussian to each of X, Y
        mean_x = np.mean(X, 0)
        mean_y = np.mean(Y, 0)
        cov_x = np.cov(X.T)
        [Dx, Vx] = np.linalg.eig(cov_x + 1e-3 * np.eye(d))
        Dx = np.real(Dx)
        Vx = np.real(Vx)
        # a hack in case the data are high-dimensional and the covariance matrix
        # is low rank.
        Dx[Dx <= 0] = 1e-3

        # shrink the covariance so that the drawn samples will not be so
        # far away from the data
        eig_pow = 0.9  # 1.0 = not shrink
        reduced_cov_x = Vx.dot(np.diag(Dx ** eig_pow)).dot(Vx.T) + 1e-3 * np.eye(d)
        cov_y = np.cov(Y.T)
        [Dy, Vy] = np.linalg.eig(cov_y + 1e-3 * np.eye(d))
        Vy = np.real(Vy)
        Dy = np.real(Dy)
        Dy[Dy <= 0] = 1e-3
        reduced_cov_y = Vy.dot(np.diag(Dy ** eig_pow).dot(Vy.T)) + 1e-3 * np.eye(d)
        # integer division
        Jx = int(n_test_locs // 2)
        Jy = n_test_locs - Jx

        assert Jx + Jy == n_test_locs, "total test locations is not n_test_locs"
        Tx = np.random.multivariate_normal(mean_x, reduced_cov_x, Jx)
        Ty = np.random.multivariate_normal(mean_y, reduced_cov_y, Jy)
        T0 = np.vstack((Tx, Ty))

    def gain(param):

        t = n + m
        ro = n / t

        z_1 = gauss_kernel(X, T0, param)
        cov_1 = np.cov(z_1.T)

        z_2 = gauss_kernel(Y, T0, param)
        cov_2 = np.cov(z_2.T)

        cov = (1 / ro) * cov_1 + (1 / (1 - ro)) * cov_2

        reg = 1e-5
        S = np.mean(z_1, axis=0) - np.mean(z_2, axis=0)
        if J > 1:
            S = np.sqrt(t) * np.linalg.solve(sqrtm(cov + reg * np.eye(J)), S)
        else:
            S = np.sqrt(t) * np.sqrt(1 / cov) * S

        S = np.sum(np.abs(S))

        return S

    Z = np.concatenate((X, Y))
    med = meddistance(Z, 1000)
    list_gwidth2 = np.hstack(((med ** 2) * (2.0 ** np.linspace(-5, 5, 50))))
    list_gwidth2.sort()
    powers = np.zeros(len(list_gwidth2))

    for wi, gwidth2 in enumerate(list_gwidth2):
        powers[wi] = gain(gwidth2)

    best = np.argmax(powers)
    gwidth2 = list_gwidth2[best]

    return (T0, gwidth2)


def solver_ME(
    X, Y, n_test_locs, max_iter=300, T_step_size=1, gwidth_step_size=0.1, tol_fun=1e-4
):
    """
    Optimize the test locations and the Gaussian kernel width by
    maximizing the test power.

    Parameters
    ----------
    X : array-like, shape = [n_samples_1, n_features]
        Samples from distribution P
    Y : array-like, shape = [n_samples_2, n_features]
        Samples from distribution Q
    n_test_locs : integer
        Number of test locations
    max_iter : integer
        gradient ascent iterations
    T_step_size : float
        gradient step size for the test locations
    gwidth_step_size : float
        gradient step size for the gaussian width
    tol_fun : float
        termination tolerance of the objective value

    Return
    -------
    (T, gwidth2): tuple
    T : array-like, shape = [n_test_locs, n_features]
    gwidth2: float
        The square Gaussian width of the Radial basis function kernel
    """

    n, d = X.shape
    m, d = Y.shape
    J = n_test_locs

    t = n + m

    T0, gwidth20 = initial2_T_gwidth2(X, Y, n_test_locs)

    param = np.ones((J, d))
    param[:, :] = T0
    param = np.ravel(param, order="C")
    param = np.hstack((param, np.array([np.sqrt(gwidth20)])))

    def gain_torch(param):

        """ Compute f using pytorch expressions """
        if not (isinstance(param, torch.Tensor)):
            param = torch.tensor(param, dtype=torch.float64, requires_grad=True)

        t = n + m
        ro = n / t
        d = np.shape(X)[1]

        X_new = torch.from_numpy(X)
        Y_new = torch.from_numpy(Y)

        D2 = torch.sum(X_new ** 2, 1)
        D2 = torch.reshape(D2, (D2.shape[0], 1))
        D2 = D2 - 2 * torch.matmul(
            X_new, torch.t(torch.reshape(param[: J * d], (J, d)))
        )
        D2 = D2 + torch.sum(torch.reshape(param[: J * d], (J, d)) ** 2, 1)

        z_1 = torch.exp(-D2 / (2 * param[J * d] ** 2))
        cov_1 = z_1 - torch.mean(z_1, 0)
        cov_1 = (1 / (cov_1.shape[0] - 1)) * torch.matmul(torch.t(cov_1), cov_1)

        D2 = torch.sum(Y_new ** 2, 1)
        D2 = torch.reshape(D2, (D2.shape[0], 1))
        D2 = D2 - 2 * torch.matmul(
            Y_new, torch.t(torch.reshape(param[: J * d], (J, d)))
        )
        D2 = D2 + torch.sum(torch.reshape(param[: J * d], (J, d)) ** 2, 1)

        z_2 = torch.exp(-D2 / (2 * param[J * d] ** 2))
        cov_2 = z_2 - torch.mean(z_2, 0)
        cov_2 = (1 / (cov_2.shape[0] - 1)) * torch.matmul(torch.t(cov_2), cov_2)

        cov = (1 / ro) * cov_1 + (1 / (1 - ro)) * cov_2

        reg = 1e-5
        S = torch.mean(z_1, 0) - torch.mean(z_2, 0)

        if J > 1:
            cov = cov + reg * torch.eye(J).double()
            u, d, v = torch.svd(cov, some=True)
            d = torch.diag(d)
            square_root = torch.matmul(torch.matmul(u, torch.sqrt(d)), v.t())

            S = (torch.sqrt(torch.Tensor([t]).double())) * (
                torch.inverse(square_root).matmul(S)
            )

        else:
            S = torch.sqrt(t / cov) * S

        S = torch.sum(torch.abs(S))

        return S

    def grad_gain_torch(param):
        """ Compute the gradient of f using pytorch's autograd """
        param = torch.tensor(param, dtype=torch.float64, requires_grad=True)
        stat = gain_torch(param)
        stat.backward()
        return param.grad.numpy()

    step_pow = 0.5
    max_gam_sq_step = 1.0

    # gradient_ascent
    for t in range(max_iter):

        gain_old = gain_torch(param)
        gradient = grad_gain_torch(param)

        grad_T = gradient[: J * d]
        grad_gwidth = gradient[J * d]

        param[: J * d] = (
            param[: J * d]
            + T_step_size
            * grad_T
            / (t + 1) ** step_pow
            / np.sum(grad_T ** 2) ** step_pow
        )

        update_gwidth = (
            gwidth_step_size
            * np.sign(grad_gwidth)
            * min(np.abs(grad_gwidth), max_gam_sq_step)
            / (t + 1) ** step_pow
        )
        param[J * d] = param[J * d] + update_gwidth

        if param[J * d] < 0:
            param[J * d] = np.abs(param[J * d])

        if t >= 2 and abs(gain_torch(param) - gain_old) <= tol_fun:
            return (param[: J * d].reshape(J, d), param[J * d] ** 2)

    T = param[: J * d].reshape(J, d)
    gwidth2 = param[J * d] ** 2

    return (T, gwidth2)


def test_asymptotic_SCF(X, Y, test_locs, gwith2, alpha):
    """ L1-based two-sample test using the smooth characteristic
    functions, evaluated at a finite set of test locations.
    Use Gaussian kernel.
    ----------
    X : array-like, shape = [n_samples_1, n_features]
        Samples from distribution P
    Y : array-like, shape = [n_samples_2, n_features]
        Samples from distribution Q
    test_locs : array-like, shape = [n_locations, n_features]
        Finite set of test locations
    gwidth2 : float
        The square Gaussian width of the Radial basis function kernel
    alpha : float
        The level of the test
    Return
    -------
    results : dictionary
    {alpha: float, pvalue: float, h0_rejected: Boolean, test: float}
    """

    n, d = X.shape
    m, d = Y.shape
    J, d = test_locs.shape

    t = n + m
    ro = n / t

    X_scale = X / np.sqrt(gwith2)
    Y_scale = Y / np.sqrt(gwith2)

    fx = np.exp(-np.sum(X_scale ** 2, 1) / 2).reshape(n, 1)
    fy = np.exp(-np.sum(Y_scale ** 2, 1) / 2).reshape(m, 1)

    x_freq = np.dot(X_scale, test_locs.T).reshape(n, J)
    y_freq = np.dot(Y_scale, test_locs.T).reshape(m, J)

    X_1 = np.cos(x_freq) * fx
    X_2 = np.sin(x_freq) * fx

    Y_1 = np.cos(y_freq) * fy
    Y_2 = np.sin(y_freq) * fy

    z_1 = np.hstack((X_1, X_2))
    z_2 = np.hstack((Y_1, Y_2))

    cov_1 = (1 / ro) * np.cov(z_1.T)
    cov_2 = (1 / (1 - ro)) * np.cov(z_2.T)

    cov = cov_1 + cov_2

    reg = 1e-5
    S = np.mean(z_1, axis=0) - np.mean(z_2, axis=0)
    S = np.sqrt(t) * np.linalg.solve(sqrtm(cov + reg * np.eye(2 * J)), S)

    S = np.sum(np.abs(S))
    p_value = sf_naka(2 * J, S)

    results = {
        "alpha": alpha,
        "pvalue": p_value,
        "h0_rejected": p_value < alpha,
        "test": S,
    }

    return results


####Initialization of SCF ####


def initial3_T_gwidth2(X, Y, n_test_locs):
    """
    Fit a Gaussian to each dataset and draw half of n_test_locs from
    each. Linear search for the best Gaussian width in the
    list that maximizes the test power.
    ----------
    X : array-like, shape = [n_samples_1, n_features]
        Samples from distribution P
    Y : array-like, shape = [n_samples_2, n_features]
        Samples from distribution Q
    n_test_locs : float
        Number of test locations.
    Return
    -------
    (T0, gwidth2) : tuple
    T0 : array-like, shape = [n_test_locs, n_features]
    gwidth2: float
        The square Gaussian width of the Radial basis function kernel
    """

    n, d = X.shape
    m, d = Y.shape
    J = n_test_locs

    if n_test_locs == 1:
        T0 = init_locs_randn(X, Y)

    else:
        # fit a Gaussian to each of X, Y
        mean_x = np.mean(X, 0)
        mean_y = np.mean(Y, 0)
        cov_x = np.cov(X.T)
        [Dx, Vx] = np.linalg.eig(cov_x + 1e-3 * np.eye(d))
        Dx = np.real(Dx)
        Vx = np.real(Vx)
        # a hack in case the data are high-dimensional and the covariance matrix
        # is low rank.
        Dx[Dx <= 0] = 1e-3

        # shrink the covariance so that the drawn samples will not be so
        # far away from the data
        eig_pow = 0.9  # 1.0 = not shrink
        reduced_cov_x = Vx.dot(np.diag(Dx ** eig_pow)).dot(Vx.T) + 1e-3 * np.eye(d)
        cov_y = np.cov(Y.T)
        [Dy, Vy] = np.linalg.eig(cov_y + 1e-3 * np.eye(d))
        Vy = np.real(Vy)
        Dy = np.real(Dy)
        Dy[Dy <= 0] = 1e-3
        reduced_cov_y = Vy.dot(np.diag(Dy ** eig_pow).dot(Vy.T)) + 1e-3 * np.eye(d)

        Jx = int(n_test_locs // 2)
        Jy = n_test_locs - Jx

        assert Jx + Jy == n_test_locs, "total test locations is not n_test_locs"
        Tx = np.random.multivariate_normal(mean_x, reduced_cov_x, Jx)
        Ty = np.random.multivariate_normal(mean_y, reduced_cov_y, Jy)
        T0 = np.vstack((Tx, Ty))

    def gain(param):

        t = n + m
        ro = n / t

        X_scale = X / np.sqrt(param)
        Y_scale = Y / np.sqrt(param)

        fx = np.exp(-np.sum(X_scale ** 2, 1) / 2).reshape(n, 1)
        fy = np.exp(-np.sum(Y_scale ** 2, 1) / 2).reshape(m, 1)

        x_freq = np.dot(X_scale, T0.T).reshape(n, J)
        y_freq = np.dot(Y_scale, T0.T).reshape(m, J)

        X_1 = np.cos(x_freq) * fx
        X_2 = np.sin(x_freq) * fx

        Y_1 = np.cos(y_freq) * fy
        Y_2 = np.sin(y_freq) * fy

        z_1 = np.hstack((X_1, X_2))
        z_2 = np.hstack((Y_1, Y_2))

        cov_1 = (1 / ro) * np.cov(z_1.T)
        cov_2 = (1 / (1 - ro)) * np.cov(z_2.T)

        cov = cov_1 + cov_2

        reg = 1e-5
        S = np.mean(z_1, axis=0) - np.mean(z_2, axis=0)
        S = np.sqrt(t) * np.linalg.solve(sqrtm(cov + reg * np.eye(2 * J)), S)

        S = np.sum(np.abs(S))

        return S

    Z = np.concatenate((X, Y))
    med = meddistance(Z, 1000)
    list_gwidth2 = np.hstack(((med ** 2) * (2.0 ** np.linspace(-5, 5, 50))))
    list_gwidth2.sort()
    powers = np.zeros(len(list_gwidth2))

    for wi, gwidth2 in enumerate(list_gwidth2):
        powers[wi] = gain(gwidth2)

    best = np.argmax(powers)
    gwidth2 = list_gwidth2[best]

    return (T0, gwidth2)


def initial4_T_gwidth2(X, Y, n_test_locs):
    """Test frequencies are drawn from the standard Gaussian.
    Linear search for the best Gaussian width in the
    list that maximizes the test power.
    ----------
    X : array-like, shape = [n_samples_1, n_features]
        Samples from distribution P
    Y : array-like, shape = [n_samples_2, n_features]
        Samples from distribution Q
    n_test_locs : float
        Number of test locations.
    Return
    -------
    (T0, gwidth2) : tuple
    T0 : array-like, shape = [n_test_locs, n_features]
    gwidth2: float
        The square Gaussian width of the Radial basis function kernel
    """

    n, d = X.shape
    m, d = Y.shape
    J = n_test_locs

    T0 = np.random.randn(J, d)

    def gain(param):

        t = n + m
        ro = n / t

        X_scale = X / np.sqrt(param)
        Y_scale = Y / np.sqrt(param)

        fx = np.exp(-np.sum(X_scale ** 2, 1) / 2).reshape(n, 1)
        fy = np.exp(-np.sum(Y_scale ** 2, 1) / 2).reshape(m, 1)

        x_freq = np.dot(X_scale, T0.T).reshape(n, J)
        y_freq = np.dot(Y_scale, T0.T).reshape(m, J)

        X_1 = np.cos(x_freq) * fx
        X_2 = np.sin(x_freq) * fx

        Y_1 = np.cos(y_freq) * fy
        Y_2 = np.sin(y_freq) * fy

        z_1 = np.hstack((X_1, X_2))
        z_2 = np.hstack((Y_1, Y_2))

        cov_1 = (1 / ro) * np.cov(z_1.T)
        cov_2 = (1 / (1 - ro)) * np.cov(z_2.T)

        cov = cov_1 + cov_2

        reg = 1e-5
        S = np.mean(z_1, axis=0) - np.mean(z_2, axis=0)
        S = np.sqrt(t) * np.linalg.solve(sqrtm(cov + reg * np.eye(2 * J)), S)

        S = np.sum(np.abs(S))

        return S

    stdx = np.mean(np.std(X, 0))
    stdy = np.mean(np.std(Y, 0))
    mstd = (stdx + stdy) / 2.0
    mean_sd = mstd
    scales = 2.0 ** np.linspace(-4, 4, 20)
    list_gwidth = np.hstack(
        (mean_sd * scales * (d ** 0.5), 2 ** np.linspace(-20, 10, 20))
    )
    list_gwidth2 = list_gwidth ** 2
    list_gwidth2.sort()
    powers = np.zeros(len(list_gwidth2))

    for wi, gwidth2 in enumerate(list_gwidth2):
        powers[wi] = gain(gwidth2)

    best = np.argmax(powers)
    gwidth2 = list_gwidth2[best]

    return (T0, gwidth2)


def solver_SCF(
    X, Y, n_test_locs, max_iter=300, T_step_size=1, gwidth_step_size=0.1, tol_fun=1e-4
):

    """
    Optimize the test frequencies and the Gaussian kernel width by
    maximizing the test power.
    ----------
    X : array-like, shape = [n_samples_1, n_features]
        Samples from distribution P
    Y : array-like, shape = [n_samples_2, n_features]
        Samples from distribution Q
    n_test_locs : array-like, shape = [n_locations, n_features]
        Number of test frequencies
    max_iter : integer
        gradient ascent iterations
    T_step_size : float
        gradient step size for the test locations
    gwidth_step_size : float
        gradient step size for the gaussian width
    tol_fun : float
        termination tolerance of the objective value

    Return
    -------
    (T, gwidth2): tuple
    T : array-like, shape = [n_test_locs, n_features]
    gwidth2: float
        The square Gaussian width of the Radial basis function kernel
    """

    n, d = X.shape
    m, d = Y.shape

    J = n_test_locs

    T0, gwidth20 = initial3_T_gwidth2(X, Y, n_test_locs)
    # T0,gwidth20 = initial4_T_gwidth2(X,Y,n_test_locs)

    param = np.ones((J, d))
    param[:, :] = T0
    param = np.ravel(param, order="C")
    param = np.hstack((param, np.array([np.sqrt(gwidth20)])))

    def gain_torch(param):

        """ Compute f using pytorch expressions """
        if not (isinstance(param, torch.Tensor)):
            param = torch.tensor(param, dtype=torch.float64, requires_grad=True)

        t = n + m
        ro = n / t
        d = np.shape(X)[1]

        X_new = torch.from_numpy(X)
        Y_new = torch.from_numpy(Y)

        X_scale = X_new / param[J * d]
        Y_scale = Y_new / param[J * d]

        fx = torch.exp((-1 / 2) * torch.sum(X_scale ** 2, 1)).reshape(n, 1)
        fy = torch.exp((-1 / 2) * torch.sum(Y_scale ** 2, 1)).reshape(m, 1)

        x_freq = torch.matmul(
            X_scale, torch.reshape(param[: J * d], (J, d)).t()
        ).reshape(n, J)
        y_freq = torch.matmul(
            Y_scale, torch.reshape(param[: J * d], (J, d)).t()
        ).reshape(m, J)

        X_1 = torch.cos(x_freq) * fx
        X_2 = torch.sin(x_freq) * fx

        Y_1 = torch.cos(y_freq) * fy
        Y_2 = torch.sin(y_freq) * fy

        z_1 = torch.cat((X_1, X_2), 1)
        cov_1 = z_1 - torch.mean(z_1, 0)
        cov_1 = (1 / (cov_1.shape[0] - 1)) * torch.matmul(torch.t(cov_1), cov_1)

        z_2 = torch.cat((Y_1, Y_2), 1)
        cov_2 = z_2 - torch.mean(z_2, 0)
        cov_2 = (1 / (cov_2.shape[0] - 1)) * torch.matmul(torch.t(cov_2), cov_2)

        cov = (1 / ro) * cov_1 + (1 / (1 - ro)) * cov_2

        reg = 1e-5
        S = torch.mean(z_1, 0) - torch.mean(z_2, 0)

        cov = cov + reg * torch.eye(2 * J).double()

        u, d, v = torch.svd(cov, some=True)
        d = torch.diag(d)
        square_root = torch.matmul(torch.matmul(u, torch.sqrt(d)), v.t())

        S = (torch.sqrt(torch.Tensor([t]).double())) * (
            torch.inverse(square_root).matmul(S)
        )

        S = torch.sum(torch.abs(S))

        return S

    def grad_gain_torch(param):
        """ Compute the gradient of f using pytorch's autograd """
        param = torch.tensor(param, dtype=torch.float64, requires_grad=True)
        stat = gain_torch(param)
        stat.backward()
        return param.grad.numpy()

    step_pow = 0.5
    max_gam_sq_step = 1.0

    for t in range(max_iter):

        try:
            param_old = param.copy()
            gain_old = gain_torch(param_old)

            gradient = grad_gain_torch(param)
            grad_T = gradient[: J * d]
            grad_gwidth = gradient[J * d]

            param[: J * d] = (
                param[: J * d]
                + T_step_size
                * grad_T
                / (t + 1) ** step_pow
                / np.sum(grad_T ** 2) ** 0.5
            )

            update_gwidth = (
                gwidth_step_size
                * np.sign(grad_gwidth)
                * min(np.abs(grad_gwidth), max_gam_sq_step)
                / (t + 1) ** step_pow
            )
            param[J * d] = param[J * d] + update_gwidth

            if param[J * d] < 0:
                param[J * d] = np.abs(param[J * d])

            if t >= 2 and abs(gain_torch(param) - gain_old) <= tol_fun:
                return (param[: J * d].reshape(J, d), param[J * d] ** 2)

        except RuntimeError:
            print("Exception occurred during gradient descent. Stop optimization.")
            return (param_old[: J * d].reshape(J, d), param_old[J * d] ** 2)

    T = param[: J * d].reshape(J, d)
    gwidth2 = param[J * d] ** 2

    return (T, gwidth2)
