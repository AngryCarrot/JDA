import numpy as np
import scipy.stats as stats
import scipy.io as sio
from sklearn.neighbors import KNeighborsClassifier

from JDA import JDA


# Set algorithm parameters
options = {"k": 100, "lmbda": 1.0, "ker": 'linear', "gamma": 1.0}
# options.k = 100
# options.lambdaP = 1.0
# options.ker = 'linear'     # 'primal' | 'linear' | 'rbf'
# options.gamma = 1.0        # kernel bandwidth: rbf only
T = 10

srcStr = ['Caltech10','Caltech10','Caltech10','amazon','amazon','amazon','webcam','webcam','webcam','dslr','dslr','dslr']
tgtStr = ['amazon','webcam','dslr','Caltech10','webcam','dslr','Caltech10','amazon','dslr','Caltech10','amazon','webcam']

result = []

for source, target in zip(srcStr, tgtStr):
    source = srcStr[0]
    target = tgtStr[0]
    print("{}->{}".format(source, target))
    options["data"] = "{}_vs_{}".format(source, target)

    # Source
    data = sio.loadmat(
        r"data/{}_SURF_L10.mat".format(source))
    fts = data["fts"]
    # fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); %fts先求每行的和，然后将每行的元素都除以相应行的和
    fts = list(map(lambda item: item / sum(item), fts))
    # Xs = zscore(fts,1);
    Xs = stats.zscore(fts)
    Xs = Xs.T
    Ys = data["labels"]
    Ys = np.array(list(map(lambda item: item[0], Ys)))

    # Target
    data = sio.loadmat(
        r"data/{}_SURF_L10.mat".format(target))
    fts = data["fts"]
    # fts = fts. / repmat(sum(fts, 2), 1, size(fts, 2));
    fts = list(map(lambda item: item / sum(item), fts))
    Xt = stats.zscore(fts)
    Xt = Xt.T
    Yt = data["labels"]
    Yt = np.array(list(map(lambda item: item[0], Yt)))

    print("data prepared!")

    ns = Xs.shape[-1]
    nt = Xt.shape[-1]

    # soft label
    # Cls = knnclassify(Xt',Xs',Ys,1);
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(Xs.T, Ys)

    Cls = clf.predict(Xt.T)
    acc = len(Cls[Cls == Yt]) / len(Yt)
    print("first cls: {}".format(acc))

    Cls = []
    Acc = []
    for t in range(T):
        print('==============================Iteration {}=============================='.format(t))
        Z, A = JDA(Xs, Xt, Ys, Cls, options)
        # Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
        Z = np.matmul(Z, np.diag(1 / np.sqrt(np.sum(np.square(Z), 0))))

        Zs = Z[:, :ns]
        Zt = Z[:, ns:]

        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(Zs.T, Ys)

        Cls = clf.predict(Zt.T)
        # acc = length(find(Cls==Yt))/length(Yt);
        acc = len(Cls[Cls == Yt]) / len(Yt)
        print("JDA + NN = {}".format(acc))
        Acc.append(acc)


    result.append(Acc[-1])
    print("\n\n\n")


