import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
file_name = 'all_analyze/select_features/test_sklearn_ExtraTreesClassifier.csv'


def load_data(name, name1='all_analyze/data/all_in_one_data.csv'):
    df = pd.read_csv(name)
    del df['Unnamed: 0']
    train_cols = df.columns.values.tolist()

    X = pd.read_csv(name1)
    x_cols = X.columns.values.tolist()
    for x in x_cols:
        if x not in train_cols and x != 'type':
            del X[x]

    return X,train_cols


def handle_kmeans():
    df,train_cols = load_data(file_name)
    print(df)
    print()
    print('anova result:')
    formula = 'type~ '
    for x in train_cols:
        formula=formula+x+' + '
    formula=formula[:-2]
    anova_results = anova_lm(ols(formula, df).fit())
    print(anova_results)

handle_kmeans()