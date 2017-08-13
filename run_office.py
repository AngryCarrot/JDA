import numpy as np
import scipy.stats as stats
import scipy.io as sio
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from JDA import JDA


# Set algorithm parameters
options = {"k": 200, "lmbda": 1.0, "ker": 'linear', "gamma": 1.0}
# options.k = 100
# options.lambdaP = 1.0
# options.ker = 'linear'     # 'primal' | 'linear' | 'rbf'
# options.gamma = 1.0        # kernel bandwidth: rbf only
T = 20

# srcStr = ['caltech10','caltech10','caltech10','amazon','amazon','amazon','webcam','webcam','webcam','dslr','dslr','dslr']
# tgtStr = ['amazon','webcam','dslr','caltech10','webcam','dslr','caltech10','amazon','dslr','caltech10','amazon','webcam']
srcStr = ['caltech10','caltech10','caltech10', 'amazon', 'webcam','dslr']
tgtStr = ['amazon','webcam','dslr','caltech10','caltech10','caltech10']
feature_prefix = r"../FFS/{}_{}_vgg_fc6_JDA_{}_feat.npy"
result = []

for source, target in zip(srcStr, tgtStr):
    print("{}->{}".format(source, target))
    options["data"] = "{}_vs_{}".format(source, target)

    # Source
    data = np.load(r"../features/{}_vgg_fc6_feat.npy".format(source))
    fts = np.array([item["feature"] for item in data])
    # fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); %fts先求每行的和，然后将每行的元素都除以相应行的和
    fts = list(map(lambda item: item / sum(item), fts))
    # Xs = zscore(fts,1);
    # Xs = stats.zscore(fts)
    mean = np.mean(fts)
    std = np.std(fts)
    Xs = (fts - mean) / std
    Xs = Xs.T
    Ys = np.array([item["label"] for item in data])

    # Target
    data = np.load(r"../features/{}_vgg_fc6_feat.npy".format(target))
    fts = np.array([item["feature"] for item in data])
    # fts = fts. / repmat(sum(fts, 2), 1, size(fts, 2));
    fts = list(map(lambda item: item / sum(item), fts))
    # Xt = stats.zscore(fts)
    mean = np.mean(fts)
    std = np.std(fts)
    Xt = (fts - mean) / std
    Xt = Xt.T
    Yt = np.array([item["label"] for item in data])

    print("data prepared!")

    ns = Xs.shape[-1]
    nt = Xt.shape[-1]

    # soft label
    clf = LogisticRegression()
    clf.fit(Xs.T, Ys)

    Cls = clf.predict(Xt.T)
    acc = len(Cls[Cls == Yt]) / len(Yt)
    print("first cls: {}".format(acc))

    Cls = []
    Acc = []
    for t in range(T):
        print('==============================Iteration {} =============================='.format(t))
        Z, A = JDA(Xs, Xt, Ys, Cls, options)
        # Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
        Z = np.matmul(Z, np.diag(1 / np.sqrt(np.sum(np.square(Z), 0))))

        Zs = Z[:, :ns]
        Zt = Z[:, ns:]


        clf = LogisticRegression()
        clf.fit(Zs.T, Ys)

        Cls = clf.predict(Zt.T)
        # acc = length(find(Cls==Yt))/length(Yt);
        acc = len(Cls[Cls == Yt]) / len(Yt)
        print("JDA + NN = {}".format(acc))
        Acc.append(acc)
    np.save(feature_prefix.format(source, target, "source"), {"feature": Zs, "label": Ys})
    np.save(feature_prefix.format(target, target, "target"), {"feature": Zt, "label": Yt})

    result.append(Acc[-1])
    print("\n\n\n")


