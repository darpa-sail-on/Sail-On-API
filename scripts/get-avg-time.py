import os
import json
import glob
from datetime import datetime, timedelta

def get_time(path, detector, sessions, save):
    if detector:
        session_files = glob.glob(f"{path}/*-*-*-*-????????????.json")
        print(f"session files found: {len(session_files)}")
        print(f"Detector: {detector}")
    elif sessions:
        print(f"Sessions: {sessions}")
        session_files = [x for x in glob.glob(f"{path}/*-*-*-*-????????????.json") if os.path.basename(x[:-5]) in sessions]
    times = []
    good_session_files = []
    session_test_files = []
    for f_path in session_files:
        try:
            with open(f_path, "r") as session_file:
                structure = json.load(session_file)
            session_id = os.path.basename(f_path).split(".")[0]
            if structure["created"]["detector"] == detector or detector == "":
                session_test_files.extend(glob.glob(f"{path}/{str(session_id)}.*.json"))
                good_session_files.append(f_path)
        except:
            print(f"failed to load: {f_path}")


    print(f"Tests found: {len(session_test_files)}")
    for i, t_path in enumerate(session_test_files):
        try:
            with open(t_path, "r") as session_file:
                test_structure = json.load(session_file)
        except:
            print(f"failed to load: {t_path}")

        try:
            t_start = datetime.strptime(test_structure["data_request"]["time"][0], "%Y-%m-%d %H:%M:%S.%f")
            t_end = datetime.strptime(test_structure["completion"]["time"][0], "%Y-%m-%d %H:%M:%S.%f")
            times.append(t_end - t_start)
        except:
            print(f"{t_path} was unfinished or improperly formatted")
            continue

        if i % 100 == 0:
            print(f"{i}/{len(session_test_files)}")
    
    print(f"Total tests used: {len(times)}")
    print(f"Avg time: {sum(times, timedelta(0)) / len(times)}")

    if save:
        with open(f"{detector if detector else path.split('/')[-1]}_avg_runtime.txt", "w", newline="") as f:
            f.write(f"Total tests used: {len(times)}\nAvg time: {sum(times, timedelta(0)) / len(times)}")


def command_line() -> None:
    """Run the `sail_on_evaluate` command."""
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--session-id", help="session-id", dest="session_id")
    parser.add_argument("--path", help="session files folder", dest="path")
    parser.add_argument("--detector", help="detector", dest="detector", default="")
    parser.add_argument("--sessions", help="list of session ids to check", nargs="*", dest="sessions", default=[])
    parser.add_argument("--save", help="include to save results to a txt file", dest="save", default=False)

    args = parser.parse_args()
    get_time(args.path, args.detector, args.sessions, args.save)


if __name__ == "__main__":
    command_line()
