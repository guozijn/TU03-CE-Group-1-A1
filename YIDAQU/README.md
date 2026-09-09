# Yida Qu - Appendix D Tableau Analysis

This folder contains my Tableau workbook and two figures for the individual data exploration in Appendix D. The analysis examines common job titles and locations among US LinkedIn postings with Business Analysis-related skills.

## Files

- `BA_analysis.twb`: the original workbook extracted from `BIAssignment.twbx`, with both worksheets and their settings retained.
- `figures/top_10_job_titles.png`: Top 10 Job Titles in US BA-related LinkedIn Postings.
- `figures/top_10_locations.png`: Top 10 Locations for US BA-related LinkedIn Postings.

## Data and Connection

The workbook uses `linkedin_job_postings.csv` and `job_skills.csv`, related through `job_link`. The assignment describes the source as LinkedIn job advertisements scraped in 2024. Use the original datasets supplied for the assignment. `job_summary.csv` is not used for these charts.

The raw CSV files are not included here. The supplied packaged workbook contains both large CSV files, so this folder contains the smaller `.twb` extracted from that package instead.

To open the workbook:

1. Place `linkedin_job_postings.csv` and `job_skills.csv` in the same local folder.
2. Open `BA_analysis.twb` in Tableau.
3. If Tableau reports a missing file, open the Data Source page and edit the `linkedin_job_postings` connection to point to the local `linkedin_job_postings.csv` file. The accompanying `job_skills.csv` must be in the same folder.
4. Refresh the connection and open the two worksheets.

## Filtering and Analysis

The calculated field `BA related` uses the following expression:

```text
CONTAINS(LOWER([job_skills]), "business analysis")
OR CONTAINS(LOWER([job_skills]), "business analyst")
OR CONTAINS(LOWER([job_skills]), "requirements gathering")
OR CONTAINS(LOWER([job_skills]), "business requirements")
```

This checks the skills text for at least one of the four phrases, ignoring case. The worksheets filter `BA related` to True and `search_country` to United States. Both filters are context filters, so the Top 10 is calculated within that filtered group.

For the title chart, `job_title` is placed on Rows and `COUNTD(job_link)` on Columns. Job titles are filtered to the top ten by `COUNTD(job_link)` and sorted in descending order. The location chart uses the same method with `job_location` on Rows.

`COUNTD(job_link)` counts distinct recruitment links within each category. It avoids counting repeated occurrences of a link within that category; it does not physically remove duplicate source records. The workflow filters and aggregates data in Tableau. No Python script or random sampling is used for these submitted charts.

## Interpretation

The charts describe advertisements matching the selected skill phrases. The filter does not establish a formal Business Analyst qualification requirement. LinkedIn advertisements may not represent the whole job market, and `search_country` reflects the search scope rather than independently verified job geography. Similar title or location labels can remain separate. These charts do not measure actual hiring or forecast future demand.

This folder documents the two visualisations. It does not independently reproduce every data-quality statistic reported elsewhere in the assignment.
