import pandas as pd
import numpy as np
import os

# ==========================================
# 1. PROJECT TITLE
# ==========================================

print("=" * 55)
print(" STUDENT PERFORMANCE DATA CLEANING PROJECT")
print("=" * 55)

# ==========================================
# 2. LOAD DATASET
# ==========================================

file_name = "student-mat.csv"

if not os.path.exists(file_name):
    print("Error: student-mat.csv file not found!")
    exit()

df = pd.read_csv(file_name, sep=";")

print("\nSTEP 1: Dataset Loaded Successfully")
print("Original Dataset Shape:", df.shape)

# Select required columns
columns = [
    "school", "sex", "age", "studytime",
    "absences", "G1", "G2", "G3"
]

df = df[columns].copy()

# Save original selected data
df.to_csv("student_performance_raw.csv", index=False)

# ==========================================
# 3. DATA INSPECTION
# ==========================================

print("\nSTEP 2: DATA INSPECTION")

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ==========================================
# 4. CHECK DUPLICATE RECORDS
# ==========================================

print("\nSTEP 3: DUPLICATE CHECKING")

duplicate_count = df.duplicated().sum()

print("Duplicate Records Found:", duplicate_count)

df.drop_duplicates(inplace=True)

print("Duplicates Removed Successfully.")

# ==========================================
# 5. STANDARDIZE TEXT DATA
# ==========================================

print("\nSTEP 4: TEXT STANDARDIZATION")

df["school"] = df["school"].astype(str).str.strip().str.upper()
df["sex"] = df["sex"].astype(str).str.strip().str.upper()

print("Text values standardized successfully.")

# ==========================================
# 6. CONVERT DATA TYPES
# ==========================================

print("\nSTEP 5: DATA TYPE CONVERSION")

numeric_columns = [
    "age", "studytime", "absences", "G1", "G2", "G3"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("Numeric columns converted successfully.")

# ==========================================
# 7. CHECK MISSING VALUES
# ==========================================

print("\nSTEP 6: MISSING VALUE HANDLING")

missing_before = df.isnull().sum()

print("\nMissing Values Before Cleaning:")
print(missing_before)

# Fill missing categorical values
for col in ["school", "sex"]:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# Fill missing numerical values with median
for col in numeric_columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

print("\nMissing values handled successfully.")

# ==========================================
# 8. VALIDATE DATA RANGES
# ==========================================

print("\nSTEP 7: DATA VALIDATION")

# Valid age range: 15 to 22
df.loc[~df["age"].between(15, 22), "age"] = np.nan

# Valid study time range: 1 to 4
df.loc[~df["studytime"].between(1, 4), "studytime"] = np.nan

# Absences cannot be negative
df.loc[df["absences"] < 0, "absences"] = np.nan

# Grades must be between 0 and 20
for col in ["G1", "G2", "G3"]:
    df.loc[~df[col].between(0, 20), col] = np.nan

# Fill any invalid values with median
for col in numeric_columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

print("Invalid values checked and corrected.")

# ==========================================
# 9. OUTLIER DETECTION USING IQR
# ==========================================

print("\nSTEP 8: OUTLIER DETECTION")

Q1 = df["absences"].quantile(0.25)
Q3 = df["absences"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df[
    (df["absences"] < lower_limit) |
    (df["absences"] > upper_limit)
]

outlier_count = len(outliers)

print("Outliers Found in Absences:", outlier_count)

print("Outliers are identified but not deleted.")

# ==========================================
# 10. FINAL DATA VALIDATION
# ==========================================

print("\nSTEP 9: FINAL VALIDATION")

print("Remaining Missing Values:")
print(df.isnull().sum())

print("Remaining Duplicate Records:", df.duplicated().sum())

# ==========================================
# 11. SAVE CLEANED DATASET
# ==========================================

df.to_csv("student_performance_cleaned.csv", index=False)

print("\nCleaned Dataset Saved Successfully.")

# ==========================================
# 12. GENERATE CLEANING SUMMARY
# ==========================================

summary = pd.DataFrame({
    "Cleaning Step": [
        "Original Records",
        "Duplicate Records Found",
        "Duplicate Records Removed",
        "Missing Values Before Cleaning",
        "Missing Values After Cleaning",
        "Outliers Detected",
        "Final Records"
    ],
    "Count": [
        len(pd.read_csv("student-mat.csv", sep=";")),
        duplicate_count,
        duplicate_count,
        int(missing_before.sum()),
        int(df.isnull().sum().sum()),
        outlier_count,
        len(df)
    ]
})

summary.to_csv("student_cleaning_summary.csv", index=False)

print("\nCleaning Summary:")
print(summary)

print("\n" + "=" * 55)
print(" DATA CLEANING PROCESS COMPLETED SUCCESSFULLY!")
print("=" * 55)