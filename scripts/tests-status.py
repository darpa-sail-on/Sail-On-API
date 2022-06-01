import argparse
import glob
import os
import json
import csv
import tarfile
import datetime

"""
    This script allows for tracking the completion status/progress of a specified detector over the range of tests.
    If specified, users can also use this script to tar up all relevant files which can be used for evaluation.
    Arguments:
    --detector: The name of the detector as specified in the session-id.json log on session creation
    --sessions: Alternative to detector, user can provide a list of session ids to search against
    --path: directory path containing both the TESTS and RESULTS folders for the experiment runs
    --domain: The domain the runs are under
    --f-types: Optional override for the file types to search for
    --zip: Include if you want to tar up all relevant files to isolate for evaluation
    --port: optional port number that the run was made against if the results arent in a folder named RESULTS

    Also when finished produces 3 text files:
    -List of all completed tests found
    -List of all incomplete tests found
    -List of any missing result files

    All files, including the optional tar.gz, are created in the base directory this script is run out of
"""

def tests_status(detector: str, sessions: list, path: str, domain: str, ft: list, do_zip: bool, port: str):
    # make sure detector or session ids were provided
    if not detector and not sessions:
        raise ValueError("detector or sessions must be defined")

    filename = "filename"
    if detector:
        filename = detector
    else:
        filename = datetime.datetime.now()

    # get list of test ids
    cur_dir = os.getcwd()
    with open(f"{path}/TESTS/OND/{domain}/test_ids.csv", "r") as f:
        reader = csv.reader(f)
        test_ids = [i[0] for i in reader]

    # get session files and ids
    os.chdir(f"{path}/RESULTS{port}")
    session_files = glob.glob("*-*-*-*-????????????.json")
    print(len(session_files))
    session_ids = []
    good_session_files = []
    session_test_files = []
    for f_path in session_files:
        try:
            if detector:
                with open(f_path, "r") as session_file:
                    structure = json.load(session_file)
                if structure["created"]["detector"] == detector:
                    session_id = os.path.basename(f_path).split(".")[0]
                    good_session_files.append(f_path)
                    session_test_files.extend(glob.glob(f"{str(session_id)}.*.json"))
                    session_ids.append(session_id)
            elif sessions:
                session_id = os.path.basename(f_path).split(".")[0]
                if session_id in sessions:
                    good_session_files.append(f_path)
                    session_test_files.extend(glob.glob(f"{str(session_id)}.*.json"))
                    session_ids.append(session_id)
        except:
            print(f"failed to load: {f_path}")

    print(f"{len(session_ids)} relevant session ids found")

    # Set up file type lists to check each domain for
    file_types = {
        "activity_recognition": ["classification", "detection"],
        "nlt": ["labels", "detection"],
        "transcripts": [
            "classification", "detection", "transcription", "word_spacing", 
            "slant_angle", "pen_pressure", "letter_size", "appearance"
        ],
        "image_classification": ["classification", "detection"]
    }
    # Check and replace with the override list if specified
    if ft:
        file_types[domain] = ft

    # look through result files for relevant session ids
    all_results_files = glob.glob(f"OND/{domain}/*.csv", recursive=True)
    print(f"{len(all_results_files)} result files found")
    result_files = []
    for rf in all_results_files:
        if os.path.basename(rf).split(".")[0] in session_ids:
            result_files.append(rf)
    print(f"{len(result_files)} relevant result files found")

    # check to make sure all result files are the proper length
    print("checking result files for appropriate length:")
    finished_result_files = []
    for i, r in enumerate(result_files):
        with open(r, "r") as f:
            reader = csv.reader(f)
            lines = [i for i in reader]
            if len(lines) == 1280 or "characterization" in r:
                finished_result_files.append(r)
            if i % 1000 == 0:
                print(f"{i}/{len(result_files)} files checked")
    print(f"{len(finished_result_files)} finished result files")

    # Sort through completed test files for unique test ids
    completed_test_ids = [os.path.basename(rf).split("_")[0].split(".", 1)[1] for rf in finished_result_files]
    unique_cti = list(dict.fromkeys(completed_test_ids))
    for i in unique_cti:
        c = completed_test_ids.count(i)
        proper_count = len(file_types[domain])
        if c < proper_count:
            unique_cti = [x for x in unique_cti if x != i]
            print(f"Test {i} missing files {c}/{proper_count}")
    print(f"test ids completed: {len(unique_cti)}/{len(test_ids)}")
    unfinished = []
    for t in test_ids:
        if t not in unique_cti:
            unfinished.append(t)

    # get the latest files for each test id to tar up
    print("\nSelecting most up to date files for each test...")
    latest_files_dict = dict.fromkeys(unique_cti)
    tests_missing = []
    r_files_to_tar = []
    max_session_ids = []
    for n, i in enumerate(latest_files_dict.keys()):
        if i in test_ids:
            latest_files_dict[i] = []
            for f in finished_result_files:
                if i in f:
                    latest_files_dict[i].append(f)

            for f_type in file_types[domain]:
                f_files = [t for t in latest_files_dict[i] if f_type in t]
                if len(f_files) > 0:
                    f_max = max(f_files, key=os.path.getmtime)
                    r_files_to_tar.append(f_max)
                    max_session_ids.append(os.path.basename(f_max).split(".")[0])
                else:
                    tests_missing.append(f"{i}_{f_type}")

            if all(sid == max_session_ids[0] for sid in max_session_ids):
                r_files_to_tar.append(f"{str(max_session_ids[0])}.{i}.json")
            else:
                print(f"Non matching session ids for max files for test: {i}, please decide and add manually")
                for ms in max_session_ids:
                    print(f"{ms}")
        max_session_ids.clear()
        if n % 100 == 0:
            print(f"{n}/{len(latest_files_dict.keys())} test ids checked")
    
    print(f"{len(set(r_files_to_tar))} files picked out for unique ids")
    
    # tar up all the files if specified
    if(do_zip):
        print("tarring up files...")
        files_to_tar = r_files_to_tar.copy()
        files_to_tar.extend(good_session_files)
        with tarfile.open(f"{cur_dir}/{filename}.tar.gz", mode="w:gz") as tf:
            for i, f in enumerate(files_to_tar):
                tf.add(f)
                if i % 100 == 0:
                    print(f"{i}/{len(files_to_tar)}")
        print("Done tarring")
    
    os.chdir(cur_dir)
    with open(f"{filename}_unfinished.txt", "w+") as f:
        for i in unfinished:
            f.write(f"{i}\n")

    with open(f"{filename}_finished.txt", "w+") as f:
        for i in unique_cti:
            f.write(f"{i}\n")

    with open(f"{filename}_missing.txt", "w+") as f:
        for i in tests_missing:
            f.write(f"{i}\n")

    print("Done")


def command_line() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", help="Detector version to filter for", dest="detector", default = "")
    parser.add_argument("--sessions", help="list of session ids to check", nargs="*", dest="sessions", default=[])
    parser.add_argument("--path", help="The results folder path", dest="path")
    parser.add_argument("--domain", help="The domain being searched", dest="domain")
    parser.add_argument("--f-types", help="Optional override for file types to search for", nargs="*", dest="ft", default=[])
    parser.add_argument("--zip", help="tar up the files?", dest="zip")
    parser.add_argument("--port", help="port override for non default results directories", dest="port", default="")

    args = parser.parse_args()
    if args.port:
        args.port = f"_{args.port}"

    tests_status(args.detector, args.sessions, args.path, args.domain, args.ft, bool(args.zip), args.port)


if __name__ == "__main__":
    command_line()
