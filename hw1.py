import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(999)

# q1
def normal_qq_plot(n, normal_f):
    rand_nums = normal_f(n)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    res = stats.probplot(rand_nums, plot=ax)
    ax.set_title("QQ Plot for normal with sample size {}".format(n))


def poor_man_normal(n):
    unif_nums = np.random.rand(n, 12)
    return unif_nums.sum(1) - 6.0


normal_qq_plot(100, np.random.randn)
normal_qq_plot(100, poor_man_normal)

# q2
def sim_straddle(n, S0, K, sigma, r, T):
    sim_S_T = S0 * np.exp((r - sigma**2/2) * T + sigma * sigma**(1/2) * np.random.randn(n))
    sim_disc_pay_off = np.abs(sim_S_T - K) * np.exp(-r * T)
    return sim_disc_pay_off.mean(), sim_disc_pay_off.std() / n**.5

sim_straddle(10000, 100, 100, .1, .05, 1)


# q3

def sim_corr_asian_option(num_experiment, n, N, S0, sigma, r, T):
    corr_output = np.zeros(num_experiment)
    delta_t = T / N
    for i in range(num_experiment):
        rand_nums = np.random.randn(n, N)
        exp_increment = (r - sigma**2/2) * delta_t * np.ones((n, N)) + rand_nums * sigma * delta_t**.5
        exp_cum_sum = np.cumsum(exp_increment, axis=1)
        sim_S_path = S0 * np.exp(exp_cum_sum)
        A = sim_S_path.mean(axis=1)
        corr_output[i] = np.corrcoef(A, sim_S_path[:, -1])[0, 1]
    return corr_output.mean(), corr_output.std() / num_experiment**.5

sim_corr_asian_option(300, 10000, 52, 100, .1, .05, 1)

# q4


