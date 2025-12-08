import os
import json
import csv

# Folders
RESULTS_DIR = "results"
OUTPUT_DIR = "data"
LANGS = ["en", "pt", "it"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

for lang in LANGS:
    input_path = os.path.join(RESULTS_DIR, lang, "mined_data.jsonl")
    output_path = os.path.join(OUTPUT_DIR, f"{lang}.csv")

    if not os.path.exists(input_path):
        print(f"⚠️ File not found: {input_path}")
        continue

    rows = []
    with open(input_path, "r", encoding="utf-8") as infile:
        for line in infile:
            data = json.loads(line)
            s1 = data.get("s1", "").strip()
            s2 = data.get("s2", "").strip()
            dm = data.get("dm_label", "").strip()
            article_id = data.get("article_id", "")

            # Remove "dm," from the beginning of s2
            if s2.lower().startswith(dm.lower() + ","):
                s2 = s2[len(dm) + 1:].strip()
            elif s2.lower().startswith(dm.lower()):
                s2 = s2[len(dm):].strip()

            if s2:
                s2 = s2[0].upper() + s2[1:]

            rows.append({
                "s1": s1,
                "s2": s2,
                "dm": dm,
                "article_id": article_id
            })

    with open(output_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["s1", "s2", "dm", "article_id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ {output_path} created with {len(rows)} lines.")