import pandas as pd
from collections import Counter

print("=" * 60)
print("POSTINGS FILE")
print("=" * 60)

df_post = pd.read_csv("linkedin_job_postings.csv")
print("Shape:", df_post.shape)
print()
print("Missing values per column:")
print(df_post.isnull().sum())
print()
print("Missing % per column:")
print((df_post.isnull().sum() / len(df_post) * 100).round(3))
print()
for col in ["job_title", "company", "job_location", "search_city",
            "search_position", "job_level", "job_type", "search_country"]:
    print(f"{col}: {df_post[col].nunique()} distinct values")
print()
print("job_level value counts:")
print(df_post["job_level"].value_counts())
print()
print("job_type value counts:")
print(df_post["job_type"].value_counts())
print()
print("search_country value counts:")
print(df_post["search_country"].value_counts())
print()
print("first_seen range:", df_post["first_seen"].min(), "to", df_post["first_seen"].max())
print()
print("Duplicate job_link count:", df_post["job_link"].duplicated().sum())

del df_post  # free memory before next file

print()
print("=" * 60)
print("SKILLS FILE")
print("=" * 60)

df_skills = pd.read_csv("job_skills.csv")
print("Shape:", df_skills.shape)
print("Missing job_skills:", df_skills["job_skills"].isnull().sum())
print("Missing %:", round(df_skills["job_skills"].isnull().sum() / len(df_skills) * 100, 3))

skill_counts = df_skills["job_skills"].dropna().str.count(",").add(1)
print()
print("Skills per posting - descriptive stats:")
print(skill_counts.describe())

del df_skills

print()
print("=" * 60)
print("SUMMARY FILE (chunked - this is the big 5GB file)")
print("=" * 60)

total_rows = 0
missing_summary = 0
lengths = []

for chunk in pd.read_csv("job_summary.csv", chunksize=100000):
    total_rows += len(chunk)
    missing_summary += chunk["job_summary"].isnull().sum()
    lengths.extend(chunk["job_summary"].dropna().str.len().tolist())

print("Total rows:", total_rows)
print("Missing job_summary:", missing_summary)
print("Missing %:", round(missing_summary / total_rows * 100, 3))
print()
lengths_series = pd.Series(lengths)
print("Summary length (chars) - descriptive stats:")
print(lengths_series.describe())

print()
print("DONE - copy the numbers above into your data dictionary")