import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm
import os

# ========== CORRECT PATHS ==========
DATASET_ROOT = r"C:\Users\HP\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset"
CANDIDATE_FILE = r"C:\Users\HP\Documents\ML challenge\AML\output\candidate_pairs.tsv"
OUTPUT_FILE = r"C:\Users\HP\Documents\ML challenge\AML\output\matching_results.tsv"
# ===================================

print("=== Fast Submission Script ===")

print("Loading Test Source 1...")
s1 = pd.read_csv(os.path.join(DATASET_ROOT, "test", "test_source1.tsv"), sep="\t", dtype=str)
s1_names = dict(zip(s1["entity_id"], s1["business_name"].fillna("").str.lower()))

print("Loading Test Source 2...")
s2 = pd.read_csv(os.path.join(DATASET_ROOT, "test", "test_source2.tsv"), sep="\t", dtype=str)
s2_names = dict(zip(s2["entity_id"], s2["business_name"].fillna("").str.lower()))

print("Loading Test Source 3...")
s3 = pd.read_csv(os.path.join(DATASET_ROOT, "test", "test_source3.tsv"), sep="\t", dtype=str)
s3_names = dict(zip(s3["entity_id"], s3["business_name"].fillna("").str.lower()))

all_names = {**s2_names, **s3_names}
print(f"Loaded names: S1={len(s1_names)}, S2+S3={len(all_names)}")

print("\nProcessing candidates...")

output_rows = []
chunksize = 20000

reader = pd.read_csv(CANDIDATE_FILE, sep="\t", dtype=str, chunksize=chunksize)

for chunk in tqdm(reader):
    for _, row in chunk.iterrows():
        s1_id = row["source1_entity_id"]
        candidates = str(row.get("candidate_entity_ids", "")).strip()

        if candidates == "" or candidates.lower() == "nan":
            output_rows.append((s1_id, ""))
            continue

        cand_list = [c.strip() for c in candidates.split(",") if c.strip()]
        s1_name = s1_names.get(s1_id, "")

        scored = []
        for cid in cand_list:
            cname = all_names.get(cid, "")
            if not cname:
                continue
            score = fuzz.token_set_ratio(s1_name, cname)
            if score >= 78:
                scored.append((score, cid))

        scored.sort(reverse=True)
        top_matches = [cid for score, cid in scored[:3]]
        output_rows.append((s1_id, ",".join(top_matches)))

print("\nSaving matching_results.tsv ...")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s1_id, matches in output_rows:
        f.write(f"{s1_id}\t{matches}\n")

print("Done!")
print("File saved at:", OUTPUT_FILE)
print(f"Total Source 1 entities processed: {len(output_rows)}")