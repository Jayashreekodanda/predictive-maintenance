import pandas as pd
import glob
import json

# Path to your results directory
files = glob.glob("results/*.jsonl")

records = []
for f in files:
    with open(f, "r") as infile:
        for line in infile:
            records.append(json.loads(line))

# Convert to DataFrame
df = pd.DataFrame(records)

# Save to CSV
df.to_csv("benchmark_results.csv", index=False)

print(f"Combined {len(files)} files into benchmark_results.csv")
