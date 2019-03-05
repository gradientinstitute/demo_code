# Import libraries:


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as pl


n = 10000 # number of unit records
frac_green = 0.5  # fraction of population that are green
def generate_data(n, delta_loc_green=0.1, delta_scale_green=0,
                  seed=1, frac_green=0.5):
    loc = 0.0
    scale = 1.0
    n_green = int(frac_green * n)
    n_nogreen = n - n_green
    rnd = np.random.RandomState(seed=1)
    X_isgreen = np.zeros(n, dtype=bool)
    X_isgreen[0:n_green] = True
    rnd.shuffle(X_isgreen)
    X1_green = rnd.normal(loc=(loc + delta_loc_green),
                          scale=(scale + delta_scale_green), size=n_green)
    X1_nogreen = rnd.normal(loc=loc, scale=scale, size=n_nogreen)
    X1 = np.zeros(n)
    X1[X_isgreen] = X1_green
    X1[~X_isgreen] = X1_nogreen
    return np.vstack((X1, X_isgreen)).T

X_latent = generate_data(n, seed=1)
isgreen = X_latent[:, 1].astype(bool)
# pl.figure(figsize=(5,2.5), dpi=150)
# pl.hist(X_latent[:, 0][isgreen], bins=20)
# pl.hist(X_latent[:, 0][~isgreen], bins=20)


def logistic(x, L, k, x_0):
    r = L / (1 + np.exp( -1 * k * (x - x_0) ))
    return r

def true_prob(X):
    L = 0.75
    k = 2
    x0 = 1.0
    base_rate = 0.05
    p = base_rate + logistic(X[:,0], L, k, x0)
    return p

P = true_prob(X_latent)


def sample_results(p, seed=0):
    rnd = np.random.RandomState(seed)
    results = rnd.uniform(size=p.shape) < p
    return results

# Note I'm getting the true prob here
P = true_prob(X_latent)
Y = sample_results(P)





def draw_observations(X_latent, seed=1, loc=0.0, scale=0.3):
    rnd = np.random.RandomState(seed)
    X_noise = X_latent[:, 0] + rnd.normal(loc=loc, scale=scale, size=X_latent.shape[0])
    return np.vstack((X_noise, X_latent[:,1])).T

X = draw_observations(X_latent)

# pl.figure()
# pl.scatter(X[:, 0], P)
# pl.scatter(X[:, 0], Y)
# pl.show()




n_select = 100

def train_test(X, Y, P):
    X_train = X[:n//2]
    X_test = X[n//2:]
    Y_train = Y[:n//2]
    Y_test = Y[n//2:]
    P_train = P[n//2:]
    P_test = P[:n//2]
    return X_train, X_test, Y_train, Y_test, P_train, P_test

def estimate(X_train, Y_train, X_test):
    model = LogisticRegression(solver='lbfgs')
    # model = LogisticRegression(class_weight='balanced', solver='lbfgs')
    # model = SVC(probability=True, kernel='poly')
    model.fit(X_train, Y_train)
    p_est = model.predict_proba(X_test)[:, 1]
    return p_est


def rank_eval(p, X_test, Y_test, P_test):
    rank = np.argsort(p)[::-1]
    p_rank = p[rank][0:n_select]
    X_rank = X_test[rank][0:n_select]
    Y_rank = Y_test[rank][0:n_select]
    P_true_rank = P_test[rank][0:n_select]
    acc = np.sum(Y_rank) / Y_rank.shape[0]
    frac_green = np.sum(X_rank[:, 1]) / n_select
    return acc, frac_green

def sample_eval(p, X_test, Y_test, P_test):
    selected = []
    cumulative = np.cumsum(p)
    norm = cumulative / np.amax(cumulative)
    for i in range(n_select):
        while True:
            draw = np.random.rand()
            idx = np.searchsorted(norm, draw)
            if idx not in selected:
                selected.append(idx)
                break
    Y_selected = Y_test[selected]
    X_selected = X_test[selected]
    acc = np.sum(Y_selected) / n_select
    frac_green = np.sum(X_selected[:, 1]) / n_select
    return acc, frac_green

def baserate_selection_plot():
    baserates = np.linspace(0, 2, 20)
    out = [baserate_rank(b) for b in baserates]
    acc_vector, bias_vector, sacc_vector, sbias_vector, tot_vec = zip(*out)
    return baserates, acc_vector, bias_vector, sacc_vector, sbias_vector, tot_vec

def baserate_rank(b):
    Xl = generate_data(n, delta_loc_green=b)
    P = true_prob(Xl)
    Y = sample_results(P)

    print("bias: {}".format(b))
    print("total positive rate: {}".format(np.sum(Y) / Y.shape[0]))
    i_green = Xl[:,1] ==1.0
    Y_green = Y[i_green]
    Y_nogreen = Y[~i_green]
    n_green = Y_green.shape[0]
    n_nogreen = Y_nogreen.shape[0]
    print("green positive rate: {}".format(np.sum(Y_green) / n_green))
    print("nongreen positive rate: {}".format(np.sum(Y_nogreen) / n_nogreen))

    X = draw_observations(Xl)
    X, Xs, Y, Ys, P, Ps = train_test(X, Y, P)
    Pe = estimate(X, Y, Xs)
    acc, frac_green = rank_eval(Pe, Xs, Ys, Ps)
    sample_acc, sample_frac_green = sample_eval(Pe, Xs, Ys, Ps)

    print("rank acc: {} sample acc: {}".format(acc, sample_acc))
    print("rank frac: {} sample frac: {}".format(frac_green, sample_frac_green))

    n_green = np.sum(Xs[:, 1])
    n_green_guilty = np.sum(Ys[(Xs[:, 1]).astype(bool)])
    n_guilty = np.sum(Ys)
    total_frac_green = n_green_guilty / n_guilty
    return acc, frac_green, sample_acc, sample_frac_green, total_frac_green


baserates, acc_vector, bias_vector, sacc_vector, sbias_vector, tot_vec = baserate_selection_plot()
pl.figure()
pl.plot(tot_vec, bias_vector, 'r-', label="Rank Bias")
pl.plot(tot_vec, sbias_vector, 'g-', label="Sample Bias")
pl.plot(tot_vec, tot_vec, 'k-', label="Total Proportion")
pl.legend()
pl.figure()
pl.plot(tot_vec, acc_vector, 'r-', label="Rank Acc")
pl.plot(tot_vec, sacc_vector, 'g-', label="Sampling Acc")
pl.legend()
pl.show()
