import pandas as pd
import shutil
import os

# ============================================
# STEP 1: Safety - master backup check karo
# ============================================
if not os.path.exists("embeddings_MASTER_SAFE.parquet"):
    shutil.copy("embeddings.parquet", "embeddings_MASTER_SAFE.parquet")
    print("Master safe backup bana diya")
else:
    print("Master safe backup pehle se maujood hai, skip kiya")

# ============================================
# STEP 2: Original data load karo (master safe se, taaki hamesha fresh/correct data mile)
# ============================================
df = pd.read_parquet("embeddings_MASTER_SAFE.parquet").reset_index(drop=True)
print(f"\nLoaded rows: {len(df)}")
print("Pehle ke numbers:", sorted(df['number'].unique(), key=lambda x: int(x)))

# ============================================
# STEP 3: Correct numbers assign karo (playlist ke hisaab se)
# ============================================
correct_number_map = {
    "Basic Structure of an HTML Website": "3",
    "SEO and Core Web Vitals in HTML": "6",
    "Forms and input tags in HTML": "7",
    "Inline & Block Elements in HTML": "8",
    "Id & Classes in HTML": "9",
    "Video, Audio & Media in HTML": "10",
    "Semantic Tags  in HTML": "11",
    "Exercise 1 - Pure HTML Media Player": "12",
    "Entities, Code tag and more on HTML": "13",
    "Introduction to CSS": "14",
    "Inline, Internal & External CSS": "15",
    "Exercise 1 - Solution & Shoutouts": "16",
    "CSS Selectors MasterClass": "17",
    "CSS Box Model - Margin, Padding & Borders": "18",
}

df['number'] = df['title'].map(correct_number_map)

nan_count = df['number'].isna().sum()
print(f"\nTitle match nahi hua (NaN count): {nan_count}")
if nan_count > 0:
    print("WARNING: Ye titles map nahi hue:")
    print(df[df['number'].isna()]['title'].unique())

# ============================================
# STEP 4: Duplicate rows hatao
# ============================================
before = len(df)
df = df.drop_duplicates(subset=['title', 'start', 'text']).reset_index(drop=True)
after = len(df)
print(f"\nDuplicates hataye: {before - after} rows")
print(f"Final rows: {after}")

print("\nFinal mapping:")
print(df.groupby('title')['number'].unique().to_string())

# ============================================
# STEP 5: Naya clean file save karo (original touch nahi hua)
# ============================================
df.to_parquet("embeddings_fixed.parquet", index=False)
print("\n✅ Saved: embeddings_fixed.parquet")
print("✅ embeddings.parquet aur embeddings_MASTER_SAFE.parquet dono untouched hain")