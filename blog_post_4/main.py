
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as pl

n = 10000 # number of unit records
n_select = 100
frac_green = 0.5

def generate_data(n):
    a = 0.5
    b = 10
    # X1 = np.random.gamma(a, b, size=n)
    X1 = np.random.uniform(low=0., high=50., size=n)
    X2 = np.random.choice([0,1], size=n, p=[1 - frac_green, frac_green])
    return np.vstack((X1, X2)).T

def logistic(x, L, k, x_0):
    r = L / (1 + np.exp( -1 * k * (x - x_0) ))
    return r

def true_prob(X):
    L = 0.75
    k = 1
    x_0 = 10.0
    base_rate = 0.8
    X_noise = X[:, 0] + np.random.normal(scale=1.0, size=X.shape[0])
    p1 = base_rate - logistic(X_noise, L, k, x_0)
    results = np.random.uniform(size=p1.shape) < p1
    return p1, results


X = generate_data(n)
p, Y = true_prob(X)

model = LogisticRegression(solver='lbfgs')
# model = LogisticRegression(class_weight='balanced', solver='lbfgs')
# model = SVC(probability=True, kernel='poly')
model.fit(X, Y)
p_est = model.predict_proba(X)[:, 1]


# rank and select the top
rank = np.argsort(p_est)[::-1]
p_rank = p_est[rank][0:n_select]
p_true = p[rank][0:n_select]
y_true = Y[rank][0:n_select]
acc = np.sum(y_true) / y_true.shape[0]

# small change re. gender skews rankings
# draws of noise totally change rangkings
# discontinuity implies invidivual fairness not satisfied
# accuracy/cost slightly better with rankings
# covariate shift on the relative propensity of the genders
# ranking means you never observe most of the space
# plot error rate vs size of samples -- curve has different behaviour
# conceptually, difference between auditing and university admission -- could
# terry tao miss out at uni?
# is there some explict work here on the probability our model is wrong
# doing utility function question on the utils
# define costs for false positives and false negatives then see what happens
# over the whole cohort


pl.figure()
pl.scatter(X[:, 0], p)
pl.scatter(X[:, 0], Y)
pl.scatter(X[:, 0], p_est)
pl.show()

pl.figure()
pl.scatter(p, p_est)
pl.show()

