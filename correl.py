import pandas as pd
import os

from scipy.stats import spearmanr, pearsonr

from lreg import isnews

PATHS = [
    'results-gum-10-5-2021',
    'results-rst-11-2-21',
    'results-pdtb-11-4-21'
]

COLUMNS = ['frs', 'knn', 'energy', 'mmd', 'mmd_bbsd', 'hdisc']


def write_correls(rname, df, idx=None):
    # Aggregate & .4846 & .4861 & .3849 & .2826 & \textbf{.7334} & .4855 & .4847 & .5364 & .4069 &

    row = f'{rname}'

    df = df[idx].copy() if idx is not None else df.copy()
    
    for c in COLUMNS:
        if c == 'knn':
            continue
        s = spearmanr(df[c], df['error_gap']).correlation
        row = row + f' & {s:.4f}'

    for c in COLUMNS:
        if c == 'knn':
            continue
        p = pearsonr(df[c], df['error_gap'])[0]
        row = row + f' & {p:.4f}'
    
    row = row + ' \\\\'
    print(row)

    # print(df[COLUMNS].corr())
    # print(len(df))

if __name__ == '__main__':

    df = pd.concat([pd.read_csv(f'{path}/{csv}') 
        for path in PATHS
        for csv in os.listdir(path)])
    
    df['group'] = df['group_num'].map(lambda s: s.split('_')[0])
    df['hdisc'] = df['our_h_divergence']
    df['error_gap'] = (df['train_error'] - df['transfer_error']).abs()
    df['news'] = df['target'].map(isnews)
    
    write_correls('All', df)
    write_correls('PDTB', df, df['group'] == 'pdtb')
    write_correls('RST', df, (df['group'] == 'rst') | (df['group'] == 'gum'))
    write_correls('News', df, (df['news'] == 'news'))
    write_correls('Non-News', df, ~(df['news'] == 'news'))
    fn = lambda s: s.split('-')[0]
    within = (df['source'].map(fn) == df['target'].map(fn))
    write_correls('WD', df, within)
    write_correls('OOD', df, ~within)



