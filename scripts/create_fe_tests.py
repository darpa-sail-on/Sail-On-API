# Create Feature Extraction Tests

# Outline:
# -Scan through tests in specified folder
# -create list of unique ids as they are found
# -create test of defined length
# -create metadata file for each test

# Command line args:
# -test directory
# -output test length
# -test name convention

import glob
import csv
import json


def create_fe_tests(directory: str, length: int, name: str):
    # Read in all test files
    test_files = glob.glob(
            f"{directory}/*.csv", recursive=True
        )

    # Grab all video ids
    print(f"Checking {len(test_files)} tests for video ids")
    unique_video_ids = {}
    for tf in test_files:
        if "test_ids" not in tf:
            with open(tf, "r") as f:
                reader = csv.reader(f, delimiter=",")
                lines = [x for x in reader][1:]
            for l in lines:
                unique_video_ids[l[0]] = l[0]

    uniques_list = unique_video_ids.keys()
    print(f"{len(uniques_list)} unique video ids found")

    # Create tests of specified length
    test_list = [["anonymous_id"]]
    created_index = 0
    for vid in uniques_list:
        test_list.append([vid])
        if len(test_list) > length:
            test_path = f"{directory}/{name}.{created_index}"
            create_test(test_list, test_path)
            test_list = [["anonymous_id"]]
            created_index += 1

    if len(test_list) > 1:
        test_path = f"{directory}/{name}.{created_index}"
        create_test(test_list, test_path)
    print("Done")


def create_test(test_list: [], test_path: str):
    with open(f"{test_path}_single_df.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(test_list)
    metadata = {
                "protocol": 'OND',
                "known_classes": 88,
                "max_novel_classes": 88,
                "round_size": 100,
                "threshold": 0.5,
                "red_light": 0
            }
    with open(f"{test_path}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"{test_path} created")


def command_line() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", help="The directory with all of the tests", dest="directory")
    parser.add_argument("--length", help="The output test length", dest="length")
    parser.add_argument("--name", help="The naming convention to use for output test names", dest="name")

    args = parser.parse_args()
    create_fe_tests(args.directory, int(args.length), args.name)


if __name__ == "__main__":
    command_line()