"""Qizhen Li / a1952021 — Appendix C analysis."""
from collections import Counter
from pathlib import Path
import json

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
TARGET = 'business analysis'
CHUNK_SIZE = 50_000


def find_input(name):
    for folder in (ROOT, ROOT.parent):
        for path in [folder / name, folder / (name + '.zip'), *sorted(folder.glob(name + '(*).zip'))]:
            if path.is_file():
                return path
    raise FileNotFoundError(f'Place {name} in {ROOT}')


def chunks(paths, name):
    yield from pd.read_csv(paths[name], dtype='string', keep_default_na=False,
                           chunksize=CHUNK_SIZE, compression='infer')


def blank(series):
    return series.isna() | series.str.strip().eq('')


def basic_checks(paths):
    audit, key_sets = [], {}
    for name in paths:
        keys, missing = Counter(), Counter()
        rows, invalid_dates = 0, 0
        first, last = None, None
        for frame in chunks(paths, name):
            rows += len(frame)
            columns = frame.columns.tolist()
            for column in columns:
                missing[column] += int(blank(frame[column]).sum())
            keys.update(frame.loc[~blank(frame['job_link']), 'job_link'])
            if 'first_seen' in frame:
                dates = pd.to_datetime(frame['first_seen'], errors='coerce', utc=True)
                invalid_dates += int((dates.isna() & ~blank(frame['first_seen'])).sum())
                if dates.notna().any():
                    first = dates.min() if first is None else min(first, dates.min())
                    last = dates.max() if last is None else max(last, dates.max())
        metrics = {'rows': rows, 'columns': len(columns), 'column_names': json.dumps(columns),
                   'unique_nonblank_job_links': len(keys),
                   'duplicate_job_link_excess_rows': sum(v - 1 for v in keys.values()),
                   'duplicated_job_link_keys': sum(v > 1 for v in keys.values())}
        metrics.update({f'missing:{k}': v for k, v in missing.items()})
        if first is not None:
            metrics.update(first_seen_min=str(first.date()), first_seen_max=str(last.date()),
                           invalid_nonblank_first_seen=invalid_dates)
        audit.extend((name, key, value) for key, value in metrics.items())
        key_sets[name] = set(keys)
    return pd.DataFrame(audit, columns=['scope', 'metric', 'value']), key_sets


def clean_skills(value):
    raw = [s.strip() for s in str(value).split(',') if s.strip()] if pd.notna(value) else []
    return raw, sorted({s.lower() for s in raw})


def identify_ba(paths):
    ba_keys, usable_keys, counts = set(), set(), Counter()
    for frame in chunks(paths, 'job_skills.csv'):
        for key, value in frame[['job_link', 'job_skills']].itertuples(index=False, name=None):
            raw, normalized = clean_skills(value)
            counts['rows_without_usable_skills'] += int(not normalized)
            counts['duplicate_tokens_removed'] += len(raw) - len(normalized)
            counts['empty_tokens_removed'] += sum(not s.strip() for s in value.split(','))
            if key.strip():
                if normalized:
                    usable_keys.add(key)
                if TARGET in normalized:
                    ba_keys.add(key)
            elif TARGET in normalized:
                counts['BA_rows_excluded_blank_job_link'] += 1
    records = {k: {'raw': [], 'skills': set()} for k in sorted(ba_keys)}
    for frame in chunks(paths, 'job_skills.csv'):
        selected = frame.loc[frame['job_link'].isin(ba_keys), ['job_link', 'job_skills']]
        for key, value in selected.itertuples(index=False, name=None):
            records[key]['raw'].append(value)
            records[key]['skills'].update(clean_skills(value)[1])
    cohort = pd.DataFrame([
        (k, r['raw'], sorted(r['skills']), len(r['skills'])) for k, r in records.items()
    ], columns=['job_link', 'job_skills_raw', 'skills_normalized', 'skill_count'])
    assert cohort['job_link'].is_unique
    assert cohort['skills_normalized'].map(lambda s: TARGET in s).all()
    return cohort, usable_keys, counts


def join_us(paths, cohort):
    parts = [f.loc[f['job_link'].isin(cohort['job_link'])].copy()
             for f in chunks(paths, 'linkedin_job_postings.csv')]
    metadata = pd.concat(parts, ignore_index=True)
    duplicates = metadata.loc[metadata['job_link'].duplicated(keep=False)]
    conflicts = (int(duplicates.groupby('job_link').nunique(dropna=False).gt(1).any(axis=1).sum())
                 if len(duplicates) else 0)
    joined = cohort.merge(metadata.drop_duplicates('job_link', keep='first'),
                          on='job_link', how='left', validate='one_to_one', indicator=True)
    us = joined.loc[joined['search_country'].eq('United States').fillna(False)].copy()
    if us.empty:
        raise ValueError('No usable US BA postings: cannot calculate percentages.')
    assert us['skill_count'].gt(0).all()
    return joined, us, conflicts


def category_counts(us, column):
    values = us[column].fillna('').str.strip().replace('', '(Missing)')
    table = values.value_counts().rename_axis(column).reset_index(name='frequency')
    table = table.sort_values(['frequency', column], ascending=[False, True]).reset_index(drop=True)
    table['percentage'] = table['frequency'] / len(us) * 100
    assert table['frequency'].sum() == len(us)
    return table


def save_chart(table, label, title, filename, n):
    plot = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, max(5, len(table) * .38)))
    bars = ax.barh(plot[label], plot['posting_count'], color='#28658A')
    ax.bar_label(bars, padding=3, fmt='%.0f')
    ax.set(title=title, xlabel='Number of US Business Analysis postings')
    ax.set_xlim(0, max(1, table['posting_count'].max() * 1.18) if len(table) else 1)
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(.01, .01, f'US BA postings: {n:,}. One count per posting; ties sorted alphabetically.', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(ROOT / 'figures' / filename, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    # Load data
    paths = {name: find_input(name) for name in ('job_skills.csv', 'linkedin_job_postings.csv')}
    for name, path in paths.items():
        print(f'{name}: {path}')

    # Basic checks
    basic, key_sets = basic_checks(paths)
    print(basic.to_string(index=False))
    profile = basic.set_index(['scope', 'metric'])['value']

    # Clean skills
    assert clean_skills(' Business Analysis,SQL, sql, ,')[1] == ['business analysis', 'sql']
    assert TARGET not in clean_skills('Business Analytics, Business Analysis Tools')[1]

    # Identify BA cohort
    cohort, usable_keys, cleaning = identify_ba(paths)

    # Join and filter US
    joined, us, conflicts = join_us(paths, cohort)
    n = len(us)
    print(f'Total Business Analysis postings: {len(cohort):,}')
    print(f'Total US Business Analysis postings: {n:,}')
    if n != 5466:
        print('Different from the reported 5,466. Possible technical causes: input version, '
              'token matching, duplicate handling or country filtering; cause not confirmed.')
    else:
        print('US count matches the reported 5,466.')

    # Co-occurring skills
    frequencies = Counter(s for skills in us['skills_normalized'] for s in set(skills) if s != TARGET)
    coskills = pd.DataFrame(frequencies.items(), columns=['skill', 'posting_count'])
    coskills = coskills.sort_values(['posting_count', 'skill'], ascending=[False, True])
    coskills['percentage_US_BA_postings'] = coskills['posting_count'] / n * 100
    assert coskills['posting_count'].le(n).all()
    assert coskills['posting_count'].sum() == (us['skill_count'] - 1).sum()

    # Job titles
    titles = category_counts(us, 'job_title').rename(columns={'frequency': 'posting_count'})
    known_titles = titles.loc[titles['job_title'].ne('(Missing)')]
    top_titles = known_titles.head(10)[['job_title', 'posting_count']]

    # Job level and type
    levels = category_counts(us, 'job_level').rename(columns={'frequency': 'posting_count'})
    types = category_counts(us, 'job_type').rename(columns={'frequency': 'posting_count'})

    # Data quality summary
    posts, keys = key_sets['linkedin_job_postings.csv'], key_sets['job_skills.csv']
    records = int(profile.loc[('job_skills.csv', 'rows')])
    overview = pd.DataFrame([{'total_skill_records': records,
                             'total_BA_postings': len(cohort), 'total_US_BA_postings': n}])
    stats = us['skill_count'].agg(['mean', 'median', 'min', 'max'])
    skill_overview = {
        'missing_job_skills_rows': int(profile.loc[('job_skills.csv', 'missing:job_skills')]),
        'usable_job_skills_rows': records - cleaning['rows_without_usable_skills'],
        'skill_count_scope': 'US BA postings; unique normalized tokens including business analysis',
        **{f'{key}_skill_count': value for key, value in stats.items()},
        'first_seen_scope': 'all job posting records',
        'first_seen_min': profile.get(('linkedin_job_postings.csv', 'first_seen_min'), ''),
        'first_seen_max': profile.get(('linkedin_job_postings.csv', 'first_seen_max'), '')}
    metrics = dict(cleaning)
    metrics.update(postings_without_skill_record=len(posts - keys),
                   postings_with_record_but_no_usable_skills=len((posts & keys) - usable_keys),
                   BA_postings_without_metadata=int(joined['_merge'].eq('left_only').sum()),
                   BA_conflicting_metadata_keys=conflicts,
                   duplicate_skill_policy='union tokens across same job_link',
                   duplicate_metadata_policy='first source row for same job_link',
                   US_BA_skill_count_above_100=int(us['skill_count'].gt(100).sum()),
                   US_BA_max_skill_count=int(us['skill_count'].max()))
    for skill in ['problem solving', 'problemsolving', 'communication', 'communication skills',
                  'business analytics', "bachelor's degree"]:
        metrics[f'US_BA_postings_with:{skill}'] = frequencies.get(skill, 0)
    for left, right in [('problem solving', 'problemsolving'), ('communication', 'communication skills')]:
        metrics[f'US_BA_postings_with_both:{left}|{right}'] = int(
            us['skills_normalized'].map(lambda s: left in s and right in s).sum())
    if len(top_titles):
        cutoff = int(top_titles['posting_count'].min())
        metrics['job_title_top10_cutoff_ties'] = json.dumps(
            known_titles.loc[known_titles['posting_count'].eq(cutoff), 'job_title'].tolist())
    extra = pd.DataFrame([('analysis', key, value) for key, value in metrics.items()], columns=basic.columns)
    quality = pd.concat([basic, extra], ignore_index=True)

    # Export outputs
    for folder in ('outputs', 'figures'):
        (ROOT / folder).mkdir(exist_ok=True)
    tables = {'ba_us_overview.csv': overview, 'ba_us_top_coskills.csv': coskills.head(15),
              'ba_us_top_titles.csv': top_titles, 'ba_us_level.csv': levels, 'ba_us_type.csv': types,
              'skills_overview.csv': pd.DataFrame([skill_overview]), 'data_quality_summary.csv': quality}
    for name, table in tables.items():
        table.to_csv(ROOT / 'outputs' / name, index=False)

    # Generate figures
    save_chart(coskills.head(15), 'skill',
               'Top 15 Co-occurring Skills in US Business Analysis Postings',
               'figure_1_top_coskill.png', n)
    save_chart(top_titles, 'job_title', 'Top 10 Job Titles in US Business Analysis Postings',
               'figure_2_top_titles.png', n)
    print('Generated all 7 CSV outputs and 2 figures successfully.')


if __name__ == '__main__':
    main()
