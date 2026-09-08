import pandas as pd

df1 = pd.read_csv("linkedin_job_postings.csv", nrows=5000)
df1.to_csv("sample_postings.csv", index=False)
print("Postings done:", df1.shape)

df2 = pd.read_csv("job_summary.csv", nrows=5000)
df2.to_csv("sample_summary.csv", index=False)
print("Summary done:", df2.shape)