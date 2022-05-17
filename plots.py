import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({'font.size': 12})

#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                   bias   R-squared:                       0.944
# Model:                            OLS   Adj. R-squared:                  0.944
# Method:                 Least Squares   F-statistic:                     1949.
# Date:                Thu, 04 Nov 2021   Prob (F-statistic):               0.00
# Time:                        16:13:38   Log-Likelihood:                 3347.1
# No. Observations:                2428   AIC:                            -6650.
# Df Residuals:                    2406   BIC:                            -6523.
# Df Model:                          21                                         
# Covariance Type:            nonrobust                                         
# ==================================================================================================
#                                      coef    std err          t      P>|t|      [0.025      0.975]
# --------------------------------------------------------------------------------------------------
# Intercept                         -0.0206      0.034     -0.606      0.545      -0.087       0.046
# hspace[T.lin]                     -0.0239      0.006     -3.817      0.000      -0.036      -0.012
# group[T.pdtb]                      0.0536      0.016      3.340      0.001       0.022       0.085
# group[T.rst]                       0.0600      0.018      3.256      0.001       0.024       0.096
# bert[T.pooled]                     0.0034      0.006      0.601      0.548      -0.008       0.015
# bert[T.sentence]                   0.0250      0.009      2.872      0.004       0.008       0.042
# news[T.notnews]                   -0.0029      0.010     -0.289      0.773      -0.022       0.017
# train_error                        0.3262      0.080      4.054      0.000       0.168       0.484
# lamb                              -0.0150      0.048     -0.312      0.755      -0.109       0.079
# hdisc                              0.1545      0.081      1.906      0.057      -0.004       0.313
# bert[T.pooled]:hdisc              -0.0313      0.009     -3.622      0.000      -0.048      -0.014
# bert[T.sentence]:hdisc            -0.1370      0.013    -10.600      0.000      -0.162      -0.112
# hspace[T.lin]:hdisc                0.0194      0.009      2.159      0.031       0.002       0.037
# group[T.pdtb]:hdisc               -0.0210      0.021     -1.002      0.316      -0.062       0.020
# group[T.rst]:hdisc                 0.0671      0.028      2.410      0.016       0.013       0.122
# news[T.notnews]:hdisc              0.0320      0.013      2.529      0.012       0.007       0.057
# hdisc:train_error                  1.9665      0.196     10.052      0.000       1.583       2.350
# np.power(hdisc, 2)                 0.4831      0.052      9.323      0.000       0.381       0.585
# train_error:np.power(hdisc, 2)    -1.6867      0.152    -11.074      0.000      -1.985      -1.388
# lamb:train_error                  -0.5861      0.122     -4.803      0.000      -0.825      -0.347
# np.power(lamb, 2)                 -0.1346      0.071     -1.892      0.059      -0.274       0.005
# train_error:np.power(lamb, 2)      0.4043      0.100      4.029      0.000       0.208       0.601
# ==============================================================================
# Omnibus:                        2.707   Durbin-Watson:                   1.548
# Prob(Omnibus):                  0.258   Jarque-Bera (JB):                2.718
# Skew:                          -0.046   Prob(JB):                        0.257
# Kurtosis:                       3.136   Cond. No.                         463.
# ==============================================================================

pooled_v_average = lambda hdisc: 0.0034 + hdisc * (-0.0313)
sentence_v_average = lambda hdisc: 0.0250 + hdisc * (-0.1370)
lin_v_fc = lambda hdisc: -0.0239 +  hdisc * 0.0194
notnews_v_news = lambda hdisc: -0.0029 + hdisc * 0.0320
pdtb_v_gum = lambda hdisc: 0.0536 + hdisc * (-0.0210)
rst_v_gum = lambda hdisc: 0.0600 + hdisc * 0.0671

DELTA = 0.01
START = 0
L = 3

lamb_increase = lambda train_error, delta: (-0.0150) * delta \
    + (-0.5861) * (train_error * delta) \
    + (-0.1346) * (delta ** 2) + (-0.1346) * (delta * START * 2) \
    + 0.4043 * ((delta ** 2) * train_error) + 0.4043 * (delta * START * train_error * 2)

# using reference group for all categories
hdisc_inc = lambda train_error, delta: 0.1545 * delta \
    + 1.9665 * (train_error * delta) \
    + 0.4831 * (delta ** 2) + 0.4831 * (delta * START * 2) \
    + (-1.6867) * ((delta ** 2) * train_error) + (-1.6867) * (delta * START * train_error * 2) \
    # + (-0.1370) * delta # sentence bert decreases bias the most and still positive trend

def apply(fn, x, negative=False, z=None):
    m = -1 if negative else 1
    if z is None:
        return [m * fn(xi) for xi in x]
    else:
        return [m * fn(z, xi) for xi in x]

if __name__ == '__main__':
    fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    x = [i / 100 for i in range(101)]
    # BERT
    y = apply(pooled_v_average, x, negative=True)
    ax.flat[0].plot(x, y, c='r', lw=L, label='P-BERT')
    y = apply(sentence_v_average, x, negative=True)
    ax.flat[0].plot(x, y, c='b', lw=L, ls='--', label='S-BERT')
    ax.flat[0].set_xlabel('Discrepancy')
    ax.flat[0].set_ylabel('Change in error-gap')
    ax.flat[0].set_title('Compared to A-BERT')
    ax.flat[0].legend()
    # HSPACE
    y = apply(lin_v_fc, x, negative=True)
    ax.flat[1].plot(x, y, c='b', lw=L, label='Linear model')
    ax.flat[1].set_xlabel('Discrepancy')
    ax.flat[1].set_title('Compared to FCN')
    ax.flat[1].legend()
    # NEWS
    y = apply(notnews_v_news, x, negative=True)
    ax.flat[2].plot(x, y, c='b', lw=L, label='Non-news target')
    ax.flat[2].set_xlabel('Discrepancy')
    ax.flat[2].set_title('Compared to news')
    ax.flat[2].legend()
    # DATASET
    y = apply(pdtb_v_gum, x, negative=True)
    ax.flat[3].plot(x, y, c='b', lw=L, label='PDTB style')
    y = apply(rst_v_gum, x, negative=True)
    ax.flat[3].plot(x, y, c='r', ls='--', lw=L, label='RST')
    ax.flat[3].set_xlabel('Discrepancy')
    ax.flat[3].set_title('Compared to GUM')
    ax.flat[3].legend()
    plt.tight_layout()
    plt.savefig('error-gap-analysis')

    fig, ax = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
    y = apply(lamb_increase, x, z=.1)
    ax.flat[0].plot(x, y, c='b', lw=L, label='train error = 0.1')
    y = apply(lamb_increase, x, z=.2)
    ax.flat[0].plot(x, y, c='r', lw=L, ls='--', label='train error = 0.3')
    y = apply(lamb_increase, x, z=.5)
    ax.flat[0].plot(x, y, c='g', lw=L, ls=':', label='train error = 0.5')
    ax.flat[0].set_ylabel('Change in estim. error')
    ax.flat[0].set_xlabel('Lambda')
    ax.flat[0].set_title('Compared to Lambda = 0')
    ax.flat[0].legend()
    y = apply(hdisc_inc, x, z=.1)
    ax.flat[1].plot(x, y, c='b', lw=L, label='train error = 0.1')
    y = apply(hdisc_inc, x, z=.2)
    ax.flat[1].plot(x, y, c='r', lw=L, ls='--', label='train error = 0.3')
    y = apply(hdisc_inc, x, z=.5)
    ax.flat[1].plot(x, y, c='g', lw=L, ls=':', label='train error = 0.5')
    ax.flat[1].set_xlabel('Discrepancy')
    ax.flat[1].set_title('Compared to Discrepancy = 0')
    # ax.flat[1].legend()
    plt.tight_layout()
    plt.savefig('estim-error-analysis')



