"""Reproduce Assignment 2 findings from local source CSVs; no sampling or imputation."""
import csv
import json
import argparse
from collections import Counter
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(
    description='Analyse US Business Analysis skill postings and export CSV evidence and charts.',
    epilog='Dependencies: pandas and matplotlib. No Word documents are generated.',
)
parser.add_argument('--data-dir', type=Path, required=True,
                    help='Folder containing linkedin_job_postings.csv and job_skills.csv')
parser.add_argument('--output-dir', type=Path, default=HERE / 'assignment2_outputs',
                    help='Directory for calculated CSV and JSON results')
parser.add_argument('--figure-dir', type=Path, default=HERE / 'assignment2_figures',
                    help='Directory for analysis charts')
args = parser.parse_args()
DATA = args.data_dir.expanduser().resolve()
OUT = args.output_dir.expanduser().resolve()
FIG = args.figure_dir.expanduser().resolve()
for filename in ('linkedin_job_postings.csv', 'job_skills.csv'):
    if not (DATA / filename).is_file():
        parser.error(f'Missing input file: {DATA / filename}')
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
BA = {'business analysis', 'business analyst', 'business analysts'}
def norm(s):
    return ' '.join(s.casefold().split())

# Explicit, deliberately small analyst-defined dictionaries, not exhaustive taxonomies.
FAMILIES = {
    'Communication': {'communication', 'communication skills', 'written communication', 'verbal communication', 'written and verbal communication'},
    'Project management': {'project management'},
    'Problem solving': {'problem solving', 'problemsolving', 'problem-solving'},
    'Data analysis': {'data analysis', 'data analytics'},
    'Leadership': {'leadership', 'leadership skills'},
    'SQL': {'sql', 'structured query language', 'structured query language (sql)'},
    'Requirements gathering': {'requirements gathering', 'requirements elicitation', 'requirements analysis'},
    'Excel': {'excel', 'microsoft excel', 'ms excel'},
    'Power BI': {'power bi', 'powerbi', 'microsoft power bi'},
    'Tableau': {'tableau'},
    'Python': {'python', 'python programming'},
    'Agile': {'agile', 'agile methodology', 'agile methodologies'},
    'Collaboration/teamwork': {'collaboration', 'teamwork', 'team work', 'team collaboration'},
    'Analytical skills': {'analytical skills', 'analytical thinking'},
    'Process improvement': {'process improvement', 'business process improvement', 'business process modeling', 'business process modelling', 'process mapping'},
}
CATEGORIES = {
    'Interpersonal': ['Communication', 'Leadership', 'Collaboration/teamwork'],
    'Business process / delivery': ['Project management', 'Requirements gathering', 'Agile', 'Process improvement'],
    'Analytical': ['Problem solving', 'Data analysis', 'Analytical skills'],
    'Technical tools': ['SQL', 'Excel', 'Power BI', 'Tableau', 'Python'],
}

def write(name, rows):
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / (name + '.csv'), index=False)
    return frame

def run():
    print('Reading posting metadata', flush=True)
    p = pd.read_csv(DATA / 'linkedin_job_postings.csv', dtype=str, keep_default_na=False)
    assert p.job_link.is_unique and p.job_link.ne('').all()
    us = p[p.search_country.eq('United States')].copy()
    us_links = set(us.job_link)
    all_links = set(p.job_link)
    seen = set()
    selected = set()
    token_sets = {}
    all_counts = Counter()
    fam_all = Counter()
    cat_all = Counter()
    strict = set()
    skill_rows = 0
    missing_tokens = 0
    unmatched = 0
    print('Scanning skills and calculating US baseline', flush=True)
    with (DATA / 'job_skills.csv').open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            skill_rows += 1
            link = row['job_link']
            assert link not in seen, 'Duplicate skills key requires explicit resolution'
            seen.add(link)
            unmatched += link not in all_links
            tokens = {norm(x) for x in row['job_skills'].split(',') if norm(x)}
            if tokens & BA:
                selected.add(link)
            if link not in us_links or not tokens:
                missing_tokens += link in us_links and not tokens
                continue
            families = {k for k,v in FAMILIES.items() if tokens & v}
            categories = {k for k,v in CATEGORIES.items() if families & set(v)}
            all_counts.update(tokens)
            fam_all.update(families)
            cat_all.update(categories)
            if tokens & BA:
                token_sets[link] = tokens
            if 'business analysis' in tokens:
                strict.add(link)
    b = us[us.job_link.isin(token_sets)].copy()
    n = len(b)
    observed_us = len(us_links & seen) - missing_tokens
    comparison_n = observed_us - n
    counts = Counter()
    family_counts = Counter()
    category_counts = Counter()
    per_record = []
    for r in b.to_dict('records'):
        tokens = token_sets[r['job_link']]
        families = {k for k,v in FAMILIES.items() if tokens & v}
        cats = {k for k,v in CATEGORIES.items() if families & set(v)}
        counts.update(tokens - BA)
        family_counts.update(families)
        category_counts.update(cats)
        per_record.append(dict(job_link=r['job_link'], title_mentions_business_analyst='business analyst' in norm(r['job_title']),
            strict_business_analysis=r['job_link'] in strict, distinct_skill_count=len(tokens),
            **{k: k in families for k in FAMILIES}, **{k: k in cats for k in CATEGORIES}))
    write('cohort_indicators', per_record)
    screenshot_terms = {'Communication':('communication',4230), 'Project management':('project management',2758),
        'Problem solving':('problem',2687), 'Data analysis':('data analysis',1884), 'Leadership':('leadership',1822),
        'SQL':('sql',1471),'Collaboration':('collaboration',1304),'Analytical skills':('analytical skills',1156),
        'Teamwork':('teamwork',1034),'Requirements gathering':('requirements gathering',1023)}
    write('tableau_skill_reconciliation',[dict(chart_label=k,substring_rule=term,tableau_n=expected,
        reproduced_substring_n=sum(any(term in token for token in ts) for ts in token_sets.values()))
        for k,(term,expected) in screenshot_terms.items()])
    def frequency(field):
        c = Counter(b[field])
        return write(field + '_frequency', [dict(value=k,n=v,pct=100*v/n) for k,v in c.most_common()])
    titles = frequency('job_title')
    companies = frequency('company')
    locations = frequency('job_location')
    frequency('first_seen')
    frequency('job_level')
    frequency('job_type')
    write('raw_skill_frequency', [dict(skill=k,n=v,pct=100*v/n) for k,v in counts.most_common()])
    famrows = []
    for k,v in family_counts.most_common():
        other = fam_all[k] - v
        famrows.append(dict(family=k,n=v,pct=100*v/n,other_us_n=other,other_us_denominator=comparison_n,
            other_us_pct=100*other/comparison_n,prevalence_ratio=(v/n)/(other/comparison_n) if other else None))
    families = write('skill_families_comparison',famrows)
    categories = write('skill_categories',[dict(category=k,n=v,pct=100*v/n) for k,v in category_counts.most_common()])
    write('skill_mapping',[dict(family=k,tokens=' | '.join(sorted(v))) for k,v in FAMILIES.items()])
    write('category_mapping',[dict(category=k,families=' | '.join(v)) for k,v in CATEGORIES.items()])
    country = p[p.job_link.isin(selected)].search_country.value_counts()
    write('cohort_search_country',[dict(search_country=k,n=int(v),pct=100*v/country.sum()) for k,v in country.items()])
    # Sensitivity proxy only: shared fields do not establish duplicate vacancies.
    proxy = b.assign(title=b.job_title.map(norm), advertiser=b.company.map(norm), location=b.job_location.map(norm))
    kept = proxy.drop_duplicates(['title','advertiser','location']).job_link
    proxy_counts = Counter()
    for link in kept:
        tokens = token_sets[link]
        proxy_counts.update({k for k,v in FAMILIES.items() if tokens & v})
    write('repeated_signature_sensitivity',[dict(family=k,main_n=family_counts[k],main_pct=100*family_counts[k]/n,
        collapsed_n=proxy_counts[k],collapsed_pct=100*proxy_counts[k]/len(kept)) for k in FAMILIES])
    # Selection contrast: overlap measures observed agreement, not accuracy or recall.
    title_links = set(us[us.job_title.map(norm).str.contains('business analyst',regex=False)].job_link)
    write('selection_overlap', [dict(group=k,n=v) for k,v in {
        'Both title phrase and selected skill':len(title_links & set(token_sets)),
        'Selected skill without title phrase':len(set(token_sets)-title_links),
        'Title phrase without selected skill':len(title_links-set(token_sets))}.items()])
    summary = dict(posting_rows=len(p),us_postings=len(us),skill_rows=skill_rows,skill_unique_links=len(seen),
        skills_not_in_postings=unmatched,postings_without_skills=len(all_links-seen),us_with_nonempty_skills=observed_us,
        other_us_denominator=comparison_n,selected_global=int(country.sum()),n=n,
        raw_titles=len(titles),normalised_titles=b.job_title.map(norm).nunique(),
        advertisers=len(companies),locations=len(locations),title_phrase_in_cohort=len(title_links & set(token_sets)),
        title_phrase_total_us=len(title_links),strict_n=len(strict),
        top10_titles=int(titles.head(10).n.sum()),top4_advertisers=int(companies.head(4).n.sum()),
        top4_locations=int(locations.head(4).n.sum()),signature_unique=len(kept),signature_excess=n-len(kept),
        missing_cohort_company=int(b.company.eq('').sum()),missing_cohort_location=int(b.job_location.eq('').sum()),
        first_seen_min=b.first_seen.min(),first_seen_max=b.first_seen.max())
    checks = [
        ('Unique posting keys',p.job_link.is_unique),('Unique skill keys',skill_rows==len(seen)),
        ('Every selected US key joins once',len(b)==len(token_sets)),
        ('Title frequency reconciles',int(titles.n.sum())==n),
        ('Advertiser frequency reconciles',int(companies.n.sum())==n),
        ('Location frequency reconciles',int(locations.n.sum())==n),
        ('Family counts bounded by cohort',all(0<=v<=n for v in family_counts.values())),
        ('Category counts bounded by cohort',all(0<=v<=n for v in category_counts.values())),
        ('Strict cohort contained in primary cohort',strict<=set(token_sets)),
        ('Comparison is disjoint and denominators reconcile',comparison_n+n==observed_us),
    ]
    write('validation_checks',[dict(check=k,passed=bool(v)) for k,v in checks])
    assert all(v for k,v in checks)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False})
    def bar(frame,label,title,file,limit=10):
        data=frame.head(limit).iloc[::-1]
        fig,ax=plt.subplots(figsize=(9,4.6))
        ax.barh(data[label],data.pct,color='#207D8D')
        ax.set_title(title,loc='left',weight='bold',pad=15)
        ax.set_xlabel(f'Percentage of selected US postings (n = {n:,})')
        ax.set_xlim(0,max(data.pct)*1.32)
        for y,(_,r) in enumerate(data.iterrows()):
            ax.text(r.pct+.25,y,f'{r.pct:.1f}% ({r.n:,})',va='center',fontsize=9)
        ax.grid(axis='x',alpha=.15)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(FIG/file,dpi=200)
        plt.close(fig)
    bar(titles,'value','Most frequent advertised job titles','figure_4_1_titles.png')
    bar(families,'family','Skills recorded alongside business analysis','figure_4_2_skills.png',12)
    bar(categories,'category','Coverage of the four defined skill categories','figure_4_3_categories.png')
    bar(companies,'value','Most frequent recorded advertisers','figure_4_4_advertisers.png')
    bar(locations,'value','Most frequent advertised location strings','figure_4_5_locations.png')
    fig,ax=plt.subplots(figsize=(9,4.5))
    d=families.sort_values('pct').copy()
    y=list(range(len(d)))
    ax.barh([v+.18 for v in y],d.pct,height=.35,label=f'BA skill cohort (n={n:,})',color='#207D8D')
    ax.barh([v-.18 for v in y],d.other_us_pct,height=.35,label=f'Other US postings with skills (n={comparison_n:,})',color='#97A9B7')
    ax.set_yticks(y,d.family)
    ax.set_xlabel('Posting prevalence (%)')
    ax.set_title('Within-extract comparison of skill prevalence',loc='left',weight='bold')
    ax.legend(loc='lower right',fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG/'figure_E_1_baseline.png',dpi=200)
    plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    run()
