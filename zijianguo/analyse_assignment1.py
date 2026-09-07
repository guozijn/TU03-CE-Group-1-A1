#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "archive"
OUT = ROOT / "assignment1_individual" / "outputs"
FIG = ROOT / "assignment1_individual" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

POSTINGS = DATA / "linkedin_job_postings.csv"
SKILLS = DATA / "job_skills.csv"
SUMMARIES = DATA / "job_summary.csv"

P = f"read_csv('{POSTINGS}', header=true, all_varchar=true, parallel=true)"
S = f"read_csv('{SKILLS}', header=true, all_varchar=true, parallel=true)"


def save_query(con: duckdb.DuckDBPyConnection, name: str, sql: str) -> pd.DataFrame:
    frame = con.execute(sql).fetchdf()
    frame.to_csv(OUT / f"{name}.csv", index=False)
    return frame


def profile_broken_summary(parent_links: set[str]) -> dict[str, int | float]:
    """Profile summary records using a new-record URL marker.

    Each logical row starts with an unquoted LinkedIn job URL and a comma.
    Descriptions legitimately span physical lines. This remains an assumption,
    but it is deterministic and auditable, unlike permissive CSV recovery.
    """

    marker = re.compile(rb"^https://[^/]*linkedin\.com/jobs/view/")
    seen: set[str] = set()
    lengths: list[int] = []
    current_link: str | None = None
    current_length = 0
    malformed_starts = 0

    def finish() -> None:
        nonlocal current_link, current_length
        if current_link is not None:
            seen.add(current_link)
            lengths.append(max(0, current_length - 1))  # final CSV quote

    with SUMMARIES.open("rb") as handle:
        next(handle)  # header
        for line in handle:
            if marker.match(line):
                finish()
                first, sep, remainder = line.partition(b",")
                if not sep:
                    malformed_starts += 1
                    current_link = None
                    current_length = 0
                    continue
                current_link = first.decode("utf-8", errors="replace")
                current_length = len(remainder.rstrip(b"\r\n"))
            elif current_link is not None:
                current_length += 1 + len(line.rstrip(b"\r\n"))
        finish()

    series = pd.Series(lengths, dtype="int64")
    result = {
        "record_count_boundary_rule": int(len(lengths)),
        "unique_links_boundary_rule": int(len(seen)),
        "duplicate_excess": int(len(lengths) - len(seen)),
        "links_matching_postings": int(len(seen & parent_links)),
        "links_not_in_postings": int(len(seen - parent_links)),
        "postings_without_summary_record": int(len(parent_links - seen)),
        "empty_summary_count": int((series <= 2).sum()),
        "min_summary_bytes": int(series.min()),
        "median_summary_bytes": float(series.median()),
        "p95_summary_bytes": float(series.quantile(0.95)),
        "max_summary_bytes": int(series.max()),
        "malformed_record_starts": malformed_starts,
    }
    (OUT / "summary_boundary_profile.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def horizontal_bar(frame: pd.DataFrame, label: str, value: str, title: str,
                   xlabel: str, filename: str, colour: str = "#2B6CB0") -> None:
    plot = frame.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.barh(plot[label], plot[value], color=colour)
    ax.set_title(title, loc="left", weight="bold", pad=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for i, value_i in enumerate(plot[value]):
        ax.text(value_i, i, f"  {int(value_i):,}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sns.set_theme(style="whitegrid", font_scale=0.95)
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")
    con.execute("SET temp_directory='/tmp/duckdb_info5006_a1'")

    posting_columns = [
        "job_link", "last_processed_time", "got_summary", "got_ner",
        "is_being_worked", "job_title", "company", "job_location",
        "first_seen", "search_city", "search_country", "search_position",
        "job_level", "job_type",
    ]
    missing_expr = ", ".join(
        f"sum(CASE WHEN {column} IS NULL OR trim({column})='' THEN 1 ELSE 0 END) AS {column}"
        for column in posting_columns
    )

    overview = save_query(con, "dataset_overview", f"""
        SELECT count(*) AS posting_rows,
               count(DISTINCT job_link) AS unique_posting_links,
               min(first_seen) AS first_seen_min,
               max(first_seen) AS first_seen_max,
               min(last_processed_time) AS processed_min,
               max(last_processed_time) AS processed_max,
               count(DISTINCT company) AS companies,
               count(DISTINCT job_title) AS job_titles,
               count(DISTINCT job_location) AS job_locations
        FROM {P}
    """)
    posting_missing = save_query(con, "postings_missing_counts", f"SELECT {missing_expr} FROM {P}")
    save_query(con, "posting_flag_distributions", f"""
        SELECT got_summary, got_ner, is_being_worked, count(*) AS n
        FROM {P} GROUP BY ALL ORDER BY n DESC
    """)
    save_query(con, "posting_validity_metrics", f"""
        SELECT sum(regexp_matches(job_link, '^https://[^/]*linkedin\\.com/jobs/view/[^"]*-?[0-9]+$')) AS job_specific_links,
               count(*)-sum(regexp_matches(job_link, '^https://[^/]*linkedin\\.com/jobs/view/[^"]*-?[0-9]+$')) AS non_job_specific_links,
               count(DISTINCT search_country) AS country_categories,
               count(DISTINCT job_level) AS level_categories,
               count(DISTINCT job_type) AS type_categories
        FROM {P}
    """)
    save_query(con, "country_distribution", f"""
        SELECT search_country, count(*) AS n,
               round(100.0*count(*)/sum(count(*)) OVER(),2) AS pct
        FROM {P} GROUP BY search_country ORDER BY n DESC
    """)
    skills_overview = save_query(con, "skills_overview", f"""
        SELECT count(*) AS skill_rows,
               count(DISTINCT job_link) AS unique_skill_links,
               sum(CASE WHEN job_link IS NULL OR trim(job_link)='' THEN 1 ELSE 0 END) AS missing_link,
               sum(CASE WHEN job_skills IS NULL OR trim(job_skills)='' THEN 1 ELSE 0 END) AS missing_skills
        FROM {S}
    """)

    con.execute(f"""
        CREATE TEMP TABLE ba_links AS
        SELECT DISTINCT job_link
        FROM {S}, unnest(string_split(job_skills, ',')) AS token(skill)
        WHERE lower(trim(skill)) IN ('business analysis', 'business analyst', 'business analysts')
    """)
    con.execute(f"""
        CREATE TEMP TABLE ba_us AS
        SELECT p.* FROM {P} p JOIN ba_links b USING(job_link)
        WHERE p.search_country='United States'
    """)

    cohort = save_query(con, "ba_us_overview", """
        SELECT count(*) AS postings, count(DISTINCT job_title) AS titles,
               count(DISTINCT company) AS companies,
               count(DISTINCT job_location) AS locations,
               sum(lower(job_title) LIKE '%business analyst%') AS title_mentions_business_analyst
        FROM ba_us
    """)
    save_query(con, "ba_global_country_distribution", f"""
        SELECT p.search_country, count(*) AS n,
               round(100.0*count(*)/sum(count(*)) OVER(),2) AS pct
        FROM {P} p JOIN ba_links b USING(job_link)
        GROUP BY p.search_country ORDER BY n DESC
    """)
    titles = save_query(con, "ba_us_top_titles", """
        SELECT job_title, count(*) AS n,
               round(100.0*count(*)/(SELECT count(*) FROM ba_us),1) AS pct
        FROM ba_us GROUP BY job_title ORDER BY n DESC LIMIT 12
    """)
    companies = save_query(con, "ba_us_top_companies", """
        SELECT company, count(*) AS n,
               round(100.0*count(*)/(SELECT count(*) FROM ba_us),1) AS pct
        FROM ba_us GROUP BY company ORDER BY n DESC LIMIT 12
    """)
    locations = save_query(con, "ba_us_top_locations", """
        SELECT job_location, count(*) AS n,
               round(100.0*count(*)/(SELECT count(*) FROM ba_us),1) AS pct
        FROM ba_us GROUP BY job_location ORDER BY n DESC LIMIT 12
    """)
    skills = save_query(con, "ba_us_top_coskills", f"""
        WITH tokens AS (
          SELECT b.job_link, lower(trim(skill)) AS skill
          FROM ba_us b JOIN {S} s USING(job_link),
               unnest(string_split(job_skills, ',')) AS token(skill)
        )
        SELECT skill, count(DISTINCT job_link) AS n,
               round(100.0*count(DISTINCT job_link)/(SELECT count(*) FROM ba_us),1) AS pct
        FROM tokens
        WHERE skill NOT IN ('business analysis', 'business analyst', 'business analysts')
          AND skill <> ''
        GROUP BY skill ORDER BY n DESC LIMIT 15
    """)
    daily = save_query(con, "ba_us_daily", """
        SELECT first_seen, count(*) AS n FROM ba_us GROUP BY first_seen ORDER BY first_seen
    """)
    compare_type = save_query(con, "ba_us_vs_all_us_type", f"""
        SELECT cohort, job_type, count(*) AS n,
               round(100.0*count(*)/sum(count(*)) OVER(PARTITION BY cohort),2) AS pct
        FROM (
          SELECT 'BA-skill US' AS cohort, job_type FROM ba_us
          UNION ALL
          SELECT 'All US' AS cohort, job_type FROM {P} WHERE search_country='United States'
        ) GROUP BY cohort, job_type ORDER BY job_type, cohort
    """)
    compare_level = save_query(con, "ba_us_vs_all_us_level", f"""
        SELECT cohort, job_level, count(*) AS n,
               round(100.0*count(*)/sum(count(*)) OVER(PARTITION BY cohort),1) AS pct
        FROM (
          SELECT 'BA-skill US' AS cohort, job_level FROM ba_us
          UNION ALL
          SELECT 'All US' AS cohort, job_level FROM {P} WHERE search_country='United States'
        ) GROUP BY cohort, job_level ORDER BY job_level, cohort
    """)
    token_quality = save_query(con, "ba_us_skill_token_quality", f"""
        WITH tokens AS (
          SELECT b.job_link, lower(trim(skill)) AS skill
          FROM ba_us b JOIN {S} s USING(job_link),
               unnest(string_split(job_skills, ',')) AS token(skill)
        ), per_job AS (
          SELECT job_link, count(*) AS raw_n, count(DISTINCT skill) AS distinct_n
          FROM tokens GROUP BY job_link
        )
        SELECT min(raw_n) AS min_tokens,
               approx_quantile(raw_n,.25) AS q1_tokens,
               approx_quantile(raw_n,.5) AS median_tokens,
               approx_quantile(raw_n,.75) AS q3_tokens,
               max(raw_n) AS max_tokens,
               sum(raw_n>distinct_n) AS postings_with_duplicate_normalised_tokens,
               round(100.0*sum(raw_n>distinct_n)/count(*),2) AS duplicate_token_pct
        FROM per_job
    """)

    parent_links = {row[0] for row in con.execute(f"SELECT job_link FROM {P}").fetchall()}
    summary_profile = profile_broken_summary(parent_links)

    summary = {
        "postings": overview.iloc[0].to_dict(),
        "postings_missing": posting_missing.iloc[0].to_dict(),
        "skills": skills_overview.iloc[0].to_dict(),
        "ba_us": cohort.iloc[0].to_dict(),
        "token_quality": token_quality.iloc[0].to_dict(),
        "job_summary": summary_profile,
    }
    (OUT / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    horizontal_bar(titles, "job_title", "n", "Most frequent titles in the US BA-skill cohort",
                   "Postings", "figure_2_top_titles.png")
    horizontal_bar(skills, "skill", "n", "Skills most often listed with business analysis",
                   "Distinct US postings", "figure_3_top_coskills.png", "#2F855A")
    horizontal_bar(companies, "company", "n", "Employers with the most US BA-skill postings",
                   "Postings", "figure_4_top_employers.png", "#805AD5")
    horizontal_bar(locations, "job_location", "n", "Locations with the most US BA-skill postings",
                   "Postings", "figure_5_top_locations.png", "#C05621")

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(pd.to_datetime(daily["first_seen"]), daily["n"], color="#2B6CB0")
    ax.set_title("The supplied extract covers only six first-seen dates", loc="left", weight="bold", pad=14)
    ax.set_ylabel("US BA-skill postings")
    ax.set_xlabel("First seen (January 2024)")
    ax.spines[["top", "right"]].set_visible(False)
    for x, y in zip(pd.to_datetime(daily["first_seen"]), daily["n"]):
        ax.text(x, y, f"{y:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "figure_1_daily_coverage.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    sns.barplot(data=compare_level, x="job_level", y="pct", hue="cohort", ax=axes[0], palette="Set2")
    axes[0].set_title("Seniority mix")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Share of cohort (%)")
    sns.barplot(data=compare_type, x="job_type", y="pct", hue="cohort", ax=axes[1], palette="Set2")
    axes[1].set_title("Work-arrangement mix")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Share of cohort (%)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(title="")
    fig.suptitle("BA-skill postings closely mirror the extract's constrained category distributions",
                 x=0.02, ha="left", weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "figure_6_cohort_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Individual problem-ecosystem mind map. Text is deliberately explicit so
    # the relationships survive export to Word/PDF without interactive tooling.
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis("off")
    centre = (8, 5.5)
    branches = [
        ((2.1, 9.2), "BUSINESS NEED", "US consulting company\nneeds evidence on demand\nfor business-analysis capability", "#D6EAF8"),
        ((5.9, 9.5), "STAKEHOLDERS", "Consultants • employers\njob seekers • educators\nworkforce planners", "#E8DAEF"),
        ((10.4, 9.5), "DECISIONS", "Recruitment and workforce plans\ntraining/curriculum priorities\nmarket positioning", "#D5F5E3"),
        ((14.0, 8.8), "MARKET CONCEPTS", "Occupation vs job title vs skill\ndemand • geography • employer\nseniority • work arrangement", "#FCF3CF"),
        ((14.0, 3.0), "DATA & METADATA", "1,348,454 postings\nskills + descriptions + posting fields\nO*NET/BLS context", "#FADBD8"),
        ((10.5, 1.9), "ANALYTIC QUESTIONS", "Which roles and employers?\nWhere are postings?\nWhich skills co-occur?\nHow stable are patterns?", "#D6DBDF"),
        ((5.5, 1.9), "QUALITY & RISKS", "Six-day coverage • scraped sample\nmalformed descriptions • taxonomy noise\nselection bias • unknown accuracy", "#F5CBA7"),
        ((2.0, 3.1), "ETHICS & GOVERNANCE", "Platform terms/provenance\naggregate reporting • no personal data\ntransparent assumptions\navoid causal/annual claims", "#D2B4DE"),
    ]
    for (x, y), heading, body, colour in branches:
        ax.annotate("", xy=(x, y), xytext=centre,
                    arrowprops=dict(arrowstyle="-|>", color="#64748B", lw=1.8,
                                    shrinkA=50, shrinkB=55))
        ax.text(x, y, f"{heading}\n{body}", ha="center", va="center", fontsize=10.5,
                linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.65", facecolor=colour,
                          edgecolor="#475569", linewidth=1.2))
    ax.text(*centre,
            "US JOB-MARKET DEMAND\nFOR BUSINESS-ANALYSIS\nCAPABILITY\n\nFirst CRISP-DM iteration",
            ha="center", va="center", fontsize=14, weight="bold", color="white",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#1E3A5F",
                      edgecolor="#0F172A", linewidth=2))
    ax.set_title("Individual mind map: problem concept ecosystem", loc="left",
                 fontsize=18, weight="bold", pad=12)
    ax.text(0.15, 0.015,
            "Relationship logic: the business need defines decisions; decisions define questions; "
            "questions determine data and methods; quality and ethics constrain every inference.",
            transform=ax.transAxes, fontsize=10, color="#475569")
    fig.tight_layout()
    fig.savefig(FIG / "individual_mind_map.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
