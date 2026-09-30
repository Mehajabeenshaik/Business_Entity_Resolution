import csv
import os

matching_path = r'C:\Users\HP\Documents\ML challenge\AML\output\matching_results.tsv'
candidates_path = r'C:\Users\HP\Documents\ML challenge\AML\output\test_candidate_pairs.tsv'

existing = {}
with open(matching_path) as f:
    r = csv.reader(f, delimiter='\t')
    next(r)
    for row in r:
        if row:
            existing[row[0]] = row[1] if len(row) > 1 else ''

all_test_ids = set()
with open(candidates_path) as f:
    r = csv.reader(f, delimiter='\t')
    next(r)
    for row in r:
        if row:
            all_test_ids.add(row[0])

all_test_ids.update(existing.keys())

output_path = r'C:\Users\HP\Documents\ML challenge\AML\output\matching_results_FIXED.tsv'

with open(output_path, 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['source1_entity_id', 'matched_entity_ids'])
    for sid in sorted(all_test_ids):
        w.writerow([sid, existing.get(sid, '')])

print("Total IDs written:", len(all_test_ids))
print("Had predictions:", len(existing))
print("Output saved to:", output_path)