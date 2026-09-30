from pathlib import Path
import pandas as pd

DATASET_ROOT = Path("/path/to/PFR_dataset/Ratings/subjective_ratings_all_n168")
OUTPUT = Path("outputs/demographics/pfr_demographics_summary.csv")

EXPECTED_SEX = {"w": "female", "m": "male", "d": "diverse"}

rows = []
unexpected_sex_values = []

for file in sorted(DATASET_ROOT.rglob("*.csv")):
    try:
        df = pd.read_csv(file, dtype=str, sep=None, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(file, dtype=str, sep=None, engine="python", encoding="latin1")

    df.columns = [c.strip() for c in df.columns]

    if "Sex" not in df.columns or "Age" not in df.columns:
        print(f"Skipped, missing Sex/Age: {file}")
        continue

    sex_values = df["Sex"].dropna().astype(str).str.strip().str.lower().unique()
    age_values = df["Age"].dropna().astype(str).str.strip().unique()

    sex = sex_values[0] if len(sex_values) > 0 else None
    age = age_values[0] if len(age_values) > 0 else None

    if sex not in EXPECTED_SEX:
        unexpected_sex_values.append((file.name, sex))

    rows.append({
        "file": file.name,
        "subject_id": file.name.split("_")[1] if "_" in file.name else file.stem,
        "sex_raw": sex,
        "sex_label": EXPECTED_SEX.get(sex, "unexpected"),
        "age": pd.to_numeric(age, errors="coerce"),
        "n_sex_values_in_file": len(sex_values),
        "n_age_values_in_file": len(age_values),
        "all_sex_values": ", ".join(sex_values),
        "all_age_values": ", ".join(age_values),
    })

summary = pd.DataFrame(rows)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUTPUT, index=False)

print("\nSaved to:", OUTPUT)
print("\nNumber of participants/files:", len(summary))

print("\nSex distribution:")

female = (summary["sex_label"] == "female").sum()
male = (summary["sex_label"] == "male").sum()
diverse = (summary["sex_label"] == "diverse").sum()
unexpected = (summary["sex_label"] == "unexpected").sum()

print(f"Female:     {female}")
print(f"Male:       {male}")
print(f"Diverse:    {diverse}")
print(f"Unexpected: {unexpected}")
print(f"Total:      {len(summary)}")

print("\nAge statistics:")

mean_age = summary["age"].mean()
std_age = summary["age"].std()
median_age = summary["age"].median()
min_age = summary["age"].min()
max_age = summary["age"].max()

print(f"Mean age:              {mean_age:.2f}")
print(f"Standard deviation:    {std_age:.2f}")
print(f"Median age:            {median_age:.0f}")
print(f"Youngest participant:  {min_age:.0f}")
print(f"Oldest participant:    {max_age:.0f}")

print("\nAge by sex:")
print(summary.groupby("sex_label")["age"].describe())

if unexpected_sex_values:
    print("\nUnexpected sex values found:")
    for filename, value in unexpected_sex_values:
        print(f"{filename}: {value}")
else:
    print("\nNo unexpected sex values found.")