import numpy as np
import scipy.sparse.linalg as SSL
import scipy.linalg as SL
from kernel import kernel


def JDA(Xs, Xt, Ys, Yt0, options):
    k = options["k"]
    lmbda = options["lmbda"]
    ker = options["ker"]
    gamma = options["gamma"]
    data = options["data"]

    print('JDA:  data{}s  k={}  lambda={}\n'.format(data, k, lmbda))

    # Set predefined variables
    # X = [Xs,Xt];
    X = np.hstack((Xs, Xt))
    # X = X*diag(sparse(1./sqrt(sum(X.^2))));
    X = np.matmul(X, np.diag(1 / np.sqrt(np.sum(np.square(X), 0))))
    m, n = X.shape
    ns = Xs.shape[-1]
    nt = Xt.shape[-1]
    C = len(np.unique(Ys))

    # Construct MMD matrix
    # e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
    a = 1 / (ns * np.ones([ns, 1]))
    b = -1 / (nt * np.ones([nt, 1]))
    e = np.vstack((a, b))
    M = np.matmul(e, e.T) * C

    if len(Yt0) != 0 and len(Yt0) == nt:
        for c in np.unique(Yt0):
            e = np.zeros([n, 1])
            # e(Ys==c) = 1/length(find(Ys==c));
            e[Ys == c] = 1 / len(e[Ys == c])
            # e(ns+find(Yt0==c)) = -1/length(find(Yt0==c));
            e[ns + np.where(Yt0 == c)[0]] = -1 / np.where(Yt0 == c)[0].shape[0]
            # e(isinf(e)) = 0;
            e[np.where(np.isinf(e))[0]] = 0
            M = M + np.matmul(e, e.T)

    # ‘fro’  A和A‘的积的对角线和的平方根，即sqrt(sum(diag(A'*A)))
    # M = M/norm(M,'fro');
    divider = np.sqrt(np.sum(np.diag(np.matmul(M.T, M))))
    M = M / divider

    # Construct centering matrix
    a = np.eye(n)
    b = 1 / (n * np.ones([n, n]))
    H = a - b

    # Joint Distribution Adaptation: JDA
    if "primal" == ker:
        pass
    else:
        # K is the same, M is also the same
        K = kernel(ker, X, None, gamma)
        # but a and b are not the same
        # [A,~] = eigs(K*M*K'+lambda*eye(n),K*H*K',k,'SM');
        a = np.matmul(np.matmul(K, M), K.T) + options["lmbda"] * np.eye(n)
        b = np.matmul(np.matmul(K, H), K.T)
        print("calculate eigen value and eigen vector")
        eigenvalue, eigenvector = SL.eig(a, b)
        print("eigen value and eigen vector calculated!")
        av = np.array(list(map(lambda item: np.abs(item), eigenvalue)))
        idx = np.argsort(av)[:k]
        _ = eigenvalue[idx]
        A = eigenvector[:, idx]
        Z = np.matmul(A.T, K)

    print('Algorithm JDA terminated!!!\n\n')

    return Z, A
