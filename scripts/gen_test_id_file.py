import os
import glob
import csv
import argparse

# Generate test_ids.csv file based on single dfs present in specified directory

def gen_file(path: str):

    files = glob.glob(os.path.join(path, "*_single_df.csv"))
    print(f"Test count: {len(files)}")
    files.sort()
    with open(os.path.join(path, "test_ids.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        for fi in files:
            writer.writerow([fi[len(path)+1:-14]])
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", dest="dir")
    args = parser.parse_args()
    gen_file(args.dir)