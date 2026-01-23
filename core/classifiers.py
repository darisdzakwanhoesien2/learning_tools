import numpy as np
from sklearn.neighbors import NearestCentroid
from scipy.spatial.distance import mahalanobis

def centroid_classifier(X_train, y_train, X_test):
    clf = NearestCentroid()
    clf.fit(X_train, y_train)
    return clf.predict(X_test), clf

def mahalanobis_classifier(X_train, y_train, X_test):
    classes = np.unique(y_train)
    stats = {}

    for c in classes:
        Xc = X_train[y_train == c]
        stats[c] = {
            "mean": Xc.mean(axis=0),
            "cov": np.cov(Xc.T)
        }

    preds = []
    for x in X_test:
        dists = {
            c: mahalanobis(x, stats[c]["mean"],
                           np.linalg.inv(stats[c]["cov"]))
            for c in classes
        }
        preds.append(min(dists, key=dists.get))

    return np.array(preds)