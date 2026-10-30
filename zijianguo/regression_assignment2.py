"""Fit and evaluate logistic regression for lexical BA title patterns in the US cohort.

Run analyse_assignment2.py first, then:
python regression_assignment2.py --postings-file /path/to/linkedin_job_postings.csv

Dependencies: numpy, pandas, matplotlib, scikit-learn.
Target: the advertised title contains 'business analyst', NOT validated role relevance.
This script produces analysis evidence and charts only; it creates no documents.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
    confusion_matrix, roc_curve, precision_recall_curve)

FEATURES = ['Communication', 'Project management', 'Problem solving', 'Data analysis',
    'Leadership', 'SQL', 'Requirements gathering', 'Excel', 'Power BI', 'Tableau',
    'Python', 'Agile', 'Collaboration/teamwork', 'Analytical skills', 'Process improvement']

def normalise(series):
    return series.fillna('').str.casefold().str.split().str.join(' ')

def binary(series):
    """Read CSV booleans without silently interpreting the string False as True."""
    values = series.astype(str).str.casefold()
    assert values.isin(['true', 'false', '1', '0']).all()
    return values.isin(['true', '1']).astype('int8')

def evaluate(y, probability):
    predicted = (probability >= .5).astype(int)
    return dict(accuracy=float(accuracy_score(y, predicted)),
        precision=float(precision_score(y, predicted, zero_division=0)),
        recall=float(recall_score(y, predicted, zero_division=0)),
        f1=float(f1_score(y, predicted, zero_division=0)),
        roc_auc=float(roc_auc_score(y, probability)),
        average_precision=float(average_precision_score(y, probability)),
        brier=float(brier_score_loss(y, probability)),
        log_loss=float(log_loss(y, probability)))

def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--postings-file', type=Path, required=True)
    parser.add_argument('--analysis-dir', type=Path, default=here/'assignment2_outputs')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--figure-dir', type=Path, default=here/'assignment2_figures')
    args = parser.parse_args()
    out = args.output_dir or args.analysis_dir
    out.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.analysis_dir/'cohort_indicators.csv')
    postings = pd.read_csv(args.postings_file,
        usecols=['job_link', 'job_title', 'company', 'job_location', 'search_country'],
        dtype=str, keep_default_na=False)
    data = data.merge(postings, on='job_link', how='left', validate='one_to_one', indicator=True)
    assert data['_merge'].eq('both').all()
    assert data.search_country.eq('United States').all()
    x = pd.DataFrame({name: binary(data[name]) for name in FEATURES})
    y = binary(data.title_mentions_business_analyst)
    assert np.array_equal(y.to_numpy(), normalise(data.job_title).str.contains('business analyst', regex=False).astype(int))
    assert x.notna().all().all() and set(np.unique(x)).issubset({0,1})
    # Group identical signatures so these repetitions never cross train and test.
    signatures = pd.MultiIndex.from_arrays([normalise(data[name])
        for name in ['job_title', 'company', 'job_location']])
    groups = pd.factorize(signatures)[0]
    split = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
    train, test = next(split.split(x, y, groups))
    assert set(groups[train]).isdisjoint(groups[test])
    assert len(set(y.iloc[train])) == len(set(y.iloc[test])) == 2
    # Fixed settings chosen before inspecting test results. No tuning on test data.
    model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000)
    model.fit(x.iloc[train], y.iloc[train])
    assert model.n_iter_[0] < model.max_iter, 'Model did not converge'
    probability = model.predict_proba(x.iloc[test])[:,1]
    prevalence = float(y.iloc[train].mean())
    baseline = np.repeat(prevalence, len(test))
    train_probability = model.predict_proba(x.iloc[train])[:,1]
    test_metrics = evaluate(y.iloc[test], probability)
    baseline_metrics = evaluate(y.iloc[test], baseline)
    train_metrics = evaluate(y.iloc[train], train_probability)
    pd.DataFrame([{'model':'Logistic regression (test)', **test_metrics},
        {'model':'Training-prevalence baseline (test)', **baseline_metrics},
        {'model':'Logistic regression (training; diagnostic only)', **train_metrics}]).to_csv(out/'regression_metrics.csv',index=False)
    pd.DataFrame({'feature':FEATURES,'coefficient':model.coef_[0],
        'odds_ratio':np.exp(model.coef_[0])}).sort_values('odds_ratio', ascending=False).to_csv(out/'regression_coefficients.csv',index=False)
    matrix = confusion_matrix(y.iloc[test], probability>=.5, labels=[0,1])
    pd.DataFrame(matrix,index=['Actual no title phrase','Actual title phrase'],
        columns=['Predicted no title phrase','Predicted title phrase']).to_csv(out/'regression_confusion_matrix.csv')
    membership = data[['job_link']].copy()
    membership['signature_group'] = groups
    membership['split'] = 'train'
    membership.loc[test,'split'] = 'test'
    membership['actual_title_phrase'] = y
    membership['predicted_probability'] = model.predict_proba(x)[:,1]
    membership.to_csv(out/'regression_predictions.csv',index=False)
    summary = dict(target='Title contains literal phrase business analyst (case/whitespace normalised)',
        positive_class=1, features=FEATURES, seed=42, split='GroupShuffleSplit; test_size=0.2 of signature groups',
        group_definition='Normalised title + advertiser + location', n=len(data),
        train_n=len(train),test_n=len(test),train_positive=int(y.iloc[train].sum()),
        test_positive=int(y.iloc[test].sum()),train_prevalence=prevalence,
        test_prevalence=float(y.iloc[test].mean()),group_overlap=0,
        unique_groups=int(len(set(groups))),C=1.0,solver='lbfgs',max_iter=2000,
        iterations=int(model.n_iter_[0]),intercept=float(model.intercept_[0]),
        penalty='L2 (default)',class_weight=None,threshold=.5,
        sklearn_version=sklearn.__version__,test_metrics=test_metrics,baseline_metrics=baseline_metrics,
        train_metrics=train_metrics,confusion_matrix=matrix.tolist())
    (out/'regression_summary.json').write_text(json.dumps(summary,indent=2))
    fpr,tpr,_ = roc_curve(y.iloc[test],probability)
    precision,recall,_ = precision_recall_curve(y.iloc[test],probability)
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    axes[0].plot(fpr,tpr,color='#207D8D',label=f'Logistic AUC = {test_metrics["roc_auc"]:.3f}')
    axes[0].plot([0,1],[0,1],'--',color='gray',label='Constant-score baseline')
    axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='Held-out ROC curve')
    axes[1].plot(recall,precision,color='#207D8D',label=f'Logistic AP = {test_metrics["average_precision"]:.3f}')
    axes[1].axhline(y.iloc[test].mean(),ls='--',color='gray',label=f'Baseline AP = {y.iloc[test].mean():.3f}')
    axes[1].set(xlabel='Recall',ylabel='Precision',title='Held-out precision–recall curve')
    for ax in axes:
        ax.set_xlim(0,1);ax.set_ylim(0,1.02);ax.legend(fontsize=8)
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.figure_dir/'figure_4_5_regression.png',dpi=200)
    plt.close(fig)
    coefficients = pd.read_csv(out/'regression_coefficients.csv').sort_values('odds_ratio')
    fig,ax=plt.subplots(figsize=(8,5))
    ax.scatter(coefficients.odds_ratio,coefficients.feature,color='#207D8D')
    ax.axvline(1,ls='--',color='gray')
    ax.set_xscale('log')
    ax.set_xticks([.3,.5,1,2,3],labels=['0.3','0.5','1','2','3'])
    ax.minorticks_off()
    ax.set_xlabel('Conditional odds ratio (log scale; penalised estimates)')
    ax.set_title('Skills associated with the lexical title outcome',loc='left')
    for i,row in enumerate(coefficients.itertuples()):
        ax.annotate(f'{row.odds_ratio:.2f}',(row.odds_ratio,i),xytext=(5,0),textcoords='offset points',va='center',fontsize=8)
    ax.set_xlim(coefficients.odds_ratio.min()*.7,coefficients.odds_ratio.max()*1.6)
    fig.tight_layout()
    fig.savefig(args.figure_dir/'figure_E_2_regression_odds.png',dpi=200)
    plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
