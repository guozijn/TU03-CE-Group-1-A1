"""
Assignment 2 - Individual analysis (Laxman / A1959492)

Seniority (job_level) and work-arrangement (job_type) patterns within the
group's confirmed US Business Analysis cohort.

Input:
    Subin/Final Report/BA_US_6113_postings.csv
    (group-confirmed cohort: search_country = United States, exact token
    match on {business analysis, business analyst, business analysts},
    distinct job_link -> 6,113 postings)

Output (written to ./outputs/):
    seniority_distribution.csv
    worktype_distribution.csv
    worktype_by_seniority.csv
    skillcount_by_seniority.csv
    matched_skill_consistency_check.csv

Output (written to ./figures/):
    figure_I1_seniority.png
    figure_I2_worktype_by_level.png
    figure_I3_skillcount_boxplot.png
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COHORT_PATH = "../Subin/Final Report/BA_US_6113_postings.csv"  # path from laxman/ folder to Subin's folder

NAVY, TEAL, GRAY = "#185FA5", "#1D9E75", "#B4B2A9"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.edgecolor": "#888780",
    "axes.labelcolor": "#2C2C2A",
    "text.color": "#2C2C2A",
    "xtick.color": "#5F5E5A",
    "ytick.color": "#2C2C2A",
})


def main():
    df = pd.read_csv(COHORT_PATH)
    assert df["job_link"].nunique() == len(df) == 6113, (
        "Cohort size does not match the group-confirmed 6,113 postings - "
        "check the input file before trusting downstream numbers."
    )

    # 1. Seniority distribution
    level_counts = df["job_level"].value_counts()
    level_pct = (level_counts / level_counts.sum() * 100).round(1)
    level_out = pd.DataFrame({"postings": level_counts, "pct": level_pct})
    level_out.to_csv("outputs/seniority_distribution.csv")

    # 2. Work-arrangement distribution
    type_counts = df["job_type"].value_counts()
    type_pct = (type_counts / type_counts.sum() * 100).round(1)
    type_out = pd.DataFrame({"postings": type_counts, "pct": type_pct})
    type_out.to_csv("outputs/worktype_distribution.csv")

    # 3. Work arrangement by seniority (the "is it uniform?" check)
    ct_counts = pd.crosstab(df["job_level"], df["job_type"])
    ct_pct = (pd.crosstab(df["job_level"], df["job_type"], normalize="index") * 100).round(2)
    ct_out = ct_counts.add_suffix("_n").join(ct_pct.add_suffix("_pct"))
    ct_out.to_csv("outputs/worktype_by_seniority.csv")

    # 4. Skill-count distribution by seniority
    skill_stats = df.groupby("job_level")["unique_skill_count"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).round(1)
    skill_stats.to_csv("outputs/skillcount_by_seniority.csv")

    # 5. Consistency check: does the 3-token rule match the sensitivity figure
    #    reported elsewhere in the group's report (5,466 under strict single-token)?
    matched_counts = df["matched_ba_skill"].value_counts()
    single_token_total = matched_counts[
        matched_counts.index.str.contains("business analysis", case=False)
    ].sum()
    check = pd.DataFrame({
        "check": [
            "total_cohort",
            "matched_on_business_analysis_token_incl_combinations",
        ],
        "value": [len(df), int(single_token_total)],
        "expected_from_group_report": [6113, 5466],
    })
    check.to_csv("outputs/matched_skill_consistency_check.csv", index=False)

    # ---- Figures ----
    # Figure I.1: seniority distribution
    fig, ax = plt.subplots(figsize=(6.2, 2.4), dpi=200)
    order = ["Mid senior", "Associate"]
    vals = [level_counts[l] for l in order]
    pcts = [level_pct[l] for l in order]
    bars = ax.barh(order, vals, color=[NAVY, GRAY], height=0.55)
    for bar, v, p in zip(bars, vals, pcts):
        ax.text(v + 60, bar.get_y() + bar.get_height() / 2, f"{v:,} ({p}%)",
                va="center", ha="left", fontsize=10.5)
    ax.set_xlim(0, max(vals) * 1.28)
    ax.set_xlabel("Number of postings")
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()
    ax.set_title("Seniority distribution within the US Business Analysis cohort (n = 6,113)",
                 fontsize=11, loc="left", pad=10)
    plt.tight_layout()
    plt.savefig("figures/figure_I1_seniority.png", bbox_inches="tight")
    plt.close()

    # Figure I.2: work arrangement by seniority, log scale
    ct_share = ct_pct[["Onsite", "Hybrid", "Remote"]].loc[["Associate", "Mid senior"]]
    fig, ax = plt.subplots(figsize=(6.2, 2.9), dpi=200)
    x = range(len(ct_share.index))
    width = 0.25
    colors = {"Onsite": NAVY, "Hybrid": TEAL, "Remote": GRAY}
    for i, col in enumerate(["Onsite", "Hybrid", "Remote"]):
        offset = (i - 1) * width
        ax.bar([xi + offset for xi in x], ct_share[col], width=width,
               color=colors[col], label=col)
        for xi, v in zip(x, ct_share[col]):
            label_y = v * 1.15 if v > 0.5 else v + 0.15
            ax.text(xi + offset, label_y, f"{v:.1f}%", ha="center", fontsize=8.5)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 200)
    ax.set_xticks(list(x))
    ax.set_xticklabels(ct_share.index)
    ax.set_ylabel("Share of postings (%, log scale)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.22))
    ax.set_title("Work arrangement is near-uniform across seniority levels",
                 fontsize=11, loc="left", pad=30)
    plt.tight_layout()
    plt.savefig("figures/figure_I2_worktype_by_level.png", bbox_inches="tight")
    plt.close()

    # Figure I.3: skill-count box plot by seniority
    fig, ax = plt.subplots(figsize=(6.2, 2.8), dpi=200)
    data_assoc = df.loc[df["job_level"] == "Associate", "unique_skill_count"]
    data_mid = df.loc[df["job_level"] == "Mid senior", "unique_skill_count"]
    bp = ax.boxplot([data_mid, data_assoc], orientation="horizontal",
                     tick_labels=["Mid senior", "Associate"],
                     widths=0.5, patch_artist=True, showfliers=False,
                     medianprops=dict(color="#2C2C2A", linewidth=1.5))
    for patch, c in zip(bp["boxes"], [NAVY, GRAY]):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
        patch.set_edgecolor(c)
    ax.scatter([data_mid.mean()], [1], marker="D", color="#D85A30", zorder=5, s=35)
    ax.scatter([data_assoc.mean()], [2], marker="D", color="#D85A30", zorder=5, s=35)
    ax.set_xlabel("Distinct skill tokens per posting")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Mid senior postings list more distinct skills than Associate postings",
                 fontsize=11, loc="left", pad=10)
    ax.text(0.98, 0.04, "diamond = mean, line = median (outliers hidden)",
            transform=ax.transAxes, ha="right", fontsize=8.5, color="#5F5E5A")
    plt.tight_layout()
    plt.savefig("figures/figure_I3_skillcount_boxplot.png", bbox_inches="tight")
    plt.close()

    print("Done. Outputs written to ./outputs/, figures written to ./figures/")
    print(level_out)
    print(type_out)
    print(ct_out)
    print(skill_stats)
    print(check)


if __name__ == "__main__":
    main()
