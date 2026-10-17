"""Controlled simulation for the estimand audit (Section 4.4 / 5.1).

Hierarchical generator: unit u has latent z_u ~ N(0,1), label prevalence
pi_u = expit(mu + tau z_u), frame labels y ~ Bernoulli(pi_u), and scores
s = delta*y + b_u + eps with eps ~ N(0,1) and unit offset
b_u = sigma_b (rho z_u + sqrt(1-rho^2) e_u). Within every unit the score
distributions differ only through delta, so the within-unit AUROC target is
Phi(delta/sqrt(2)) regardless of sigma_b and rho.
"""
import sys, json
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.stats import norm
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import oea_audit as o

U, NMIN, NMAX, MU, TAU, DELTA = 40, 20, 60, -2.0, 3.5, 0.8


def draw(rng, sigma_b, rho, U=U):
    z = rng.normal(size=U); e = rng.normal(size=U)
    pi = 1 / (1 + np.exp(-(MU + TAU * z)))
    b = sigma_b * (rho * z + np.sqrt(1 - rho ** 2) * e)
    n = rng.integers(NMIN, NMAX + 1, U)
    ys, ss = [], []
    for u in range(U):
        y = (rng.random(n[u]) < pi[u]).astype(int)
        ys.append(y); ss.append(DELTA * y + b[u] + rng.normal(size=n[u]))
    return ys, ss


def exp_a_one(args):
    sigma_b, rho, rep = args
    rng = np.random.default_rng([1, int(sigma_b * 10), int((rho + 1) * 10), rep])
    ys, ss = draw(rng, sigma_b, rho)
    r = o.estimands(ys, ss)
    return dict(sigma_b=sigma_b, rho=rho, rep=rep, **{k: r[k] for k in ['E1', 'X', 'W', 'E2', 'alpha', 'eligible']})


def true_e1(sigma_b, rho, n_units=40000, seed=99):
    rng = np.random.default_rng(seed)
    ys, ss = draw(rng, sigma_b, rho, U=n_units)
    y = np.concatenate(ys); s = np.concatenate(ss)
    return o.auc(y, s)


def exp_b_one(args):
    sigma_b, rho, rep, nb = args
    rng = np.random.default_rng([2, int(sigma_b * 10), int((rho + 1) * 10), rep])
    ys, ss = draw(rng, sigma_b, rho)
    cache = o.unit_summaries(ys, ss)
    y = np.concatenate(ys); s = np.concatenate(ss)
    pt = o.estimands(ys, ss, cache=cache)
    e1_frame, e1_clu, w_clu = [], [], []
    for _ in range(nb):
        i = rng.integers(0, y.size, y.size)
        e1_frame.append(o.auc(y[i], s[i]))
        idx = rng.integers(0, len(ys), len(ys))
        r = o.estimands(ys, ss, idx=idx, cache=cache)
        e1_clu.append(r['E1']); w_clu.append(r['W'])
    q = lambda v: np.nanquantile(v, [0.025, 0.975])
    return dict(sigma_b=sigma_b, rho=rho, rep=rep, E1=pt['E1'], W=pt['W'],
                E1_frame=q(e1_frame).tolist(), E1_clu=q(e1_clu).tolist(), W_clu=q(w_clu).tolist())


if __name__ == '__main__':
    out = sys.argv[1]
    with Pool() as pool:
        grid = [(sb, rho, r) for sb in (0.0, 0.5, 1.0, 2.0)
                for rho in np.round(np.arange(-0.8, 0.81, 0.2), 1) for r in range(500)]
        a = pd.DataFrame(pool.map(exp_a_one, grid, chunksize=50))
        a.to_csv(f'{out}/sim_A_raw.csv', index=False)
        scen = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.6), (1.0, -0.6)]
        truth = {f'{sb}_{rho}': true_e1(sb, rho) for sb, rho in scen}
        truth['W'] = float(norm.cdf(DELTA / np.sqrt(2)))
        json.dump(truth, open(f'{out}/sim_truth.json', 'w'), indent=1)
        b = pd.DataFrame(pool.map(exp_b_one, [(sb, rho, r, 400) for sb, rho in scen for r in range(400)], chunksize=10))
        b.to_json(f'{out}/sim_B_raw.json', orient='records')
    print('done')
