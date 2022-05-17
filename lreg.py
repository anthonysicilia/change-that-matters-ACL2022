import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import statsmodels.formula.api as smf
import statsmodels.api as sm

import seaborn as sns; sns.set(style='whitegrid')

from pathlib import Path

PATHS = [
    'results-gum-10-5-2021',
    'results-rst-11-2-21',
    'results-pdtb-11-4-21',
]

def interactions(x, y, quad_both=False, quadx=False):
    if quad_both:
        return f'+ {x} * {y} + {x} * np.power({y}, 2) + {y} * np.power({x}, 2)'
    elif quadx:
        return f'+ {x} * {y} + {y} * np.power({x}, 2)'
    else:
        return f'+ {x} * {y}'

FORM = 'bias ~ train_error + lamb + hdisc' \
    '+ hspace + group + bert + news' \
    '+ bert * hdisc + hspace * hdisc + group * hdisc + news * hdisc' \
    + interactions('hdisc', 'train_error', quadx=True) \
    + interactions('lamb', 'train_error', quadx=True)
    
def _plot_residuals(axis, df, model, col):
    if ' * ' in col:
        c1, c2 = col.split(' * ')
        # axis.scatter(df[c1] * df[c2], model.resid)
        sns.regplot(df[c1] * df[c2], model.resid, ax=axis, scatter_kws={'s' : 1})
    else:
        sns.regplot(df[col], model.resid, ax=axis, scatter_kws={'s' : 1})
    axis.set_xlabel(col)
    axis.set_ylabel('resid')

def plot_residuals(axes, df, model, cols):
    for i, c in enumerate(cols):
        _plot_residuals(axes.flat[i], df, model, c)

def lm_diag(directory, df):
    Path(f'{directory}').mkdir(parents=True, exist_ok=True)
    model = lm(FORM, directory, df)
    cols = ['train_error', 'lamb', 'hdisc']
    cols += ['lamb * hdisc', 'train_error * hdisc', 'train_error * lamb']
    _, ax = plt.subplots(2, 3, figsize=(12,6))
    plot_residuals(ax, df, model, cols)
    plt.tight_layout()
    plt.savefig(f'{directory}/resid')
    return model

def lm(formula, directory, df):
    Path(f'{directory}').mkdir(parents=True, exist_ok=True)
    res = smf.ols(formula, data=df).fit()
    with open(f'{directory}/summary.txt', 'w') as out:
        out.write(str(res.summary()))
        out.write('\n\n\n')
        out.write(str(res.summary().as_latex()))
    sm.qqplot(res.resid, scale=np.sqrt(np.var(res.resid)), line="45")
    plt.savefig(f'{directory}/qqplot'); plt.clf()
    plt.hist(res.resid, bins=20)
    plt.savefig(f'{directory}/resids'); plt.clf()
    return res

def make_A_mat(cols, zero_constrs):
    A = np.zeros((len(zero_constrs), cols))
    for i, constr in enumerate(zero_constrs):
        A[i, constr - 1] = 1
    # H0: Ab = 0, HA: Ab != 0
    return A

def isnews(s):
    if any([kw in s for kw in ['news', 'pdtb', 'rst']]):
        return 'news'
    else:
        return 'notnews'

def common_label_rename(s):
    if 'rst_gum_pdtb' in s:
        tail = s.split('_')[-1]
        return f'com_{tail}'
    else:
        return s

if __name__ == '__main__':
    df = pd.concat([pd.read_csv(f'{path}/{csv}') 
        for path in PATHS
        for csv in os.listdir(path)])
    print(len(df))
    df['lamb'] = df['ben_david_lambda']
    df['group_num'] = df['group_num'].map(common_label_rename)
    df['group'] = df['group_num'].map(lambda s: s.split('_')[0])
    df['bert'] = df['group_num'].map(lambda s: s.split('_')[1])
    df['hdisc'] = df['our_h_divergence']
    df['train_error'] = df['train_error']
    df['transfer_error'] = df['transfer_error']
    df['error_gap'] = (df['train_error'] - df['transfer_error']).abs()
    df['bias'] = df['hdisc'] - df['error_gap']
    df['news'] = df['target'].map(isnews)
    full = lm_diag('diagnosis', df)
    print('Avg resid:', np.mean(full.resid))