"""A mock provider class for testing the Server class and Provider interface."""

from .provider import Provider, FileResult
from .errors import ServerError, ProtocolError, RoundError
from .constants import ProtocolConstants

import os
import glob
import csv
import uuid
import datetime
import json
import traceback
import nltk
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from dateutil import parser
import zipfile

from typing import List, Optional, Dict, Any
from csv import reader
from io import BytesIO




def get_session_info(folder: str, session_id: str) -> Dict[str, Any]:
    """Retrieve session info."""
    path = os.path.join(folder, f"{str(session_id)}.json")
    if os.path.exists(path):
        with open(path, "r") as session_file:
            return json.load(session_file)
    return {}


def log_session(
    folder: str,
    session_id: str,
    activity: str,
    test_id: str = None,
    round_id: Optional[int] = None,
    content: Optional[Dict[str, Any]] = None,
) -> None:
    """Create a log file of all session activity."""
    structure = get_session_info(folder, session_id)
    activities = structure.get("activity", {})
    if test_id is None:
        activities[activity] = {"time": [str(datetime.datetime.now())]}
        if content is not None:
            activities[activity].update(content)
    else:
        if activity not in activities:
            activities[activity] = {"time": [str(datetime.datetime.now())]}
        tests = activities[activity].get("tests", {})
        if test_id not in tests:
            tests[test_id] = {}
        if round_id is None:
            if "time" in tests[test_id]:
                tests[test_id]["time"].append(str(datetime.datetime.now()))
            else:
                tests[test_id]["time"] = [str(datetime.datetime.now())]
            if content is not None:
                tests[test_id].update(content)
        else:
            round_id = str(round_id)
            rounds = tests[test_id].get("rounds", {})
            if round_id not in rounds:
                rounds[round_id] = {"time": [str(datetime.datetime.now())]}
            else:
                rounds[round_id]["time"].append(str(datetime.datetime.now()))
            if content is not None:
                rounds[round_id].update(content)
            tests[test_id]["rounds"] = rounds
            tests[test_id]["last round"] = round_id
        activities[activity]["tests"] = tests
    structure["activity"] = activities
    with open(os.path.join(folder, f"{str(session_id)}.json"), "w") as session_file:
        json.dump(structure, session_file, indent=2)

# region Feedback related functions
def read_feedback_file(
        csv_reader: reader,
        feedback_ids: List[str],
        metadata: Dict[str, Any],
        is_ground_truth: bool,
        round_id: Optional[int] = None,
) -> Dict[str, str]:
    """
        Gets the feedback data out of the provided
        csv feedback file for the specified ids in 
        the last submitted round.
    """

    try:
        lines = [x for x in csv_reader]
        if is_ground_truth:
            round_pos = int(round_id) * int(metadata["round_size"])
        else:
            round_pos = len(lines) - int(metadata["round_size"])
    except KeyError:
        raise RoundError(
            "no_defined_rounds",
            "round_size not defined in metadata.",
            traceback.format_stack(),
        )
    try:
        if feedback_ids is not None:
            return {
                x[0]: [float(n) for n in x[1:]]
                for x in [[n.strip(" \"'") for n in y] for y in lines][round_pos:round_pos + int(metadata["round_size"])]
                if x[0] in feedback_ids
            }
        else:
            return {
                x[0]: [float(n) for n in x[1:]]
                for x in [[n.strip(" \"'") for n in y] for y in lines][round_pos:round_pos + int(metadata["feedback_max_ids"])]
            }
    except ValueError:
        if feedback_ids is not None:
            return {
                x[0]: [n for n in x[1:]]
                for x in [[n.strip(" \"'") for n in y] for y in lines][round_pos:round_pos + int(metadata["round_size"])]
                if x[0] in feedback_ids
            }
        else:
            return {
                x[0]: [n for n in x[1:]]
                for x in [[n.strip(" \"'") for n in y] for y in lines][round_pos:round_pos + int(metadata["feedback_max_ids"])]
            }

def get_classification_feedback(
        gt_files: List[str],
        result_files: List[str],
        feedback_ids: List[str],
        metadata: Dict[str, Any],
        round_id: int,
) -> Dict[str, Any]:
    """Calculates and returns the proper feedback for classification type feedback"""
    with open(gt_files[0], "r") as f:
        gt_reader = csv.reader(f, delimiter=",")
        ground_truth = read_feedback_file(gt_reader, feedback_ids, metadata, True, round_id)
    with open(result_files[0], "r") as rf:
        result_reader = csv.reader(rf, delimiter=",")
        results = read_feedback_file(result_reader, feedback_ids, metadata, False)

    return {
        x: 0
        if ground_truth[x][1:].index(max(ground_truth[x][1:])) !=
           results[x][1:].index(max(results[x][1:]))
        else 1
        for x in ground_truth.keys()
    }


def get_detection_feedback(
        gt_files: List[str],
        result_files: List[str],
        feedback_ids: List[str],
        metadata: Dict[str, Any],
        round_id: int,
) -> Dict[str, Any]:
    """Calculates and returns the proper feedback for detection type feedback"""
    threshold = float(metadata["threshold"])

    with open(gt_files[0], "r") as f:
        gt_reader = csv.reader(f, delimiter=",")
        ground_truth = read_feedback_file(gt_reader, feedback_ids, metadata, True, round_id)
    with open(result_files[0], "r") as rf:
        result_reader = csv.reader(rf, delimiter=",")
        results = read_feedback_file(result_reader, feedback_ids, metadata, False)

    return {
        x: 0 if abs(ground_truth[x][0] - results[x][0]) > threshold else 1
        for x in ground_truth.keys()
    }

def get_characterization_feedback(
        gt_files: List[str],
        result_files: List[str],
        feedback_ids: List[str],
        metadata: Dict[str, Any],
        round_id: int,
) -> Dict[str, Any]:
    """Calculates and returns the proper feedback for characterization type feedback"""
    known_classes = int(metadata["known_classes"]) + 1

    with open(gt_files[0], "r") as f:
        gt_reader = csv.reader(f, delimiter=",")
        ground_truth = read_feedback_file(gt_reader, feedback_ids, metadata, True, round_id)
    with open(result_files[0], "r") as rf:
        result_reader = csv.reader(rf, delimiter=",")
        results = read_feedback_file(result_reader, feedback_ids, metadata, False)

    # If ground truth is not novel, returns 1 is prediction is correct, 
    # otherwise returns 1 if prediction is not a known class
    return {
        x: 0
        if (sum(ground_truth[x][1:known_classes]) > (ground_truth[x][0] + sum(ground_truth[x][known_classes:]))
            and ground_truth[x].index(max(ground_truth[x])) != results[x].index(max(results[x])))
        or (sum(ground_truth[x][1:known_classes]) < (ground_truth[x][0] + sum(ground_truth[x][known_classes:]))
            and (results[x].index(max(results[x])) in range(1, known_classes)))
        else 1
        for x in ground_truth.keys()
    }

def get_levenshtein_feedback(
        gt_files: List[str],
        result_files: List[str],
        feedback_ids: List[str],
        metadata: Dict[str, Any],
        round_id: int,
) -> Dict[str, Any]:
    """Calculates and returns the proper feedback for levenshtein type feedback"""
    with open(gt_files[0], "r") as f:
        gt_reader = csv.reader(f, delimiter=",")
        ground_truth = read_feedback_file(gt_reader, feedback_ids, metadata, True, round_id)
    with open(result_files[0], "r") as rf:
        result_reader = csv.reader(rf, delimiter=",")
        results = read_feedback_file(result_reader, feedback_ids, metadata, False)

    return {
        x: [nltk.edit_distance(ground_truth[x][i], results[x][i]) for i,_ in enumerate(ground_truth[x])]
        for x in ground_truth.keys()
    }

def get_cluster_feedback(
        gt_files: List[str],
        result_files: List[str],
        feedback_ids: List[str],
        metadata: Dict[str, Any],
        round_id: int,
) -> Dict[str, Any]:
    """Calculates and returns the proper feedback for levenshtein type feedback"""
    with open(gt_files[0], "r") as f:
        gt_reader = csv.reader(f, delimiter=",")
        ground_truth = read_feedback_file(gt_reader, feedback_ids, metadata, True, round_id)
    with open(result_files[0], "r") as rf:
        result_reader = csv.reader(rf, delimiter=",")
        results = read_feedback_file(result_reader, feedback_ids, metadata, False)

    if feedback_ids is None:
        feedback_ids = ground_truth.keys()

    gt_list = []
    r_list = []
    try:
        for key in sorted(feedback_ids):
            gt_list.append(ground_truth[key])
            r_list.append(results[key])
    except:
        raise ServerError("MissingIds", "Some requested Ids are missing from either ground truth or results file for the current round")
    
    gt_np = np.argmax(np.array(gt_list), axis=1)
    r_np = np.argmax(np.array(gt_list), axis=1)

    return_dict = {
        "cluster": normalized_mutual_info_score(gt_np, r_np)
    }

    for i in np.unique(r_np):
        places = np.where(r_np == i)[0]
        return_dict[str(i)] = (max(np.unique(gt_np[places],return_counts=True)[1])/places.shape[0])
    
    return return_dict

def psuedo_label_feedback(
        gt_files: List[str],
        feedback_ids: List[str],
        feedback_type: str,
        metadata: Dict[str, Any],
        folder: str,
        session_id: str,
        round_id: int,
) -> Dict[str, Any]:
    "Grabs psuedo label feedback for requested ids"
    with open(gt_files[0], "r") as f:
        gt_reader = csv.reader(f, delimiter=",")
        ground_truth = read_feedback_file(gt_reader, feedback_ids, metadata, True, round_id)

    structure = get_session_info(folder, session_id)

    if "psuedo_labels" in structure["activity"]:
        if feedback_type in structure["activity"]["psuedo_labels"]:
            labels = structure["activity"]["psuedo_labels"][feedback_type]
        else:
            structure["activity"]["psuedo_labels"][feedback_type] = []
            labels = []
    else:
        structure["activity"]["psuedo_labels"] = {feedback_type: []}
        labels = []

    return_dict = {}
    for x in ground_truth.keys():
        col = ground_truth[x].index(max(ground_truth[x]))
        if col not in labels:
            labels.append(col)
        return_dict[x] = labels.index(col)

    structure["activity"]["psuedo_labels"][feedback_type] = labels
    with open(os.path.join(folder, f"{str(session_id)}.json"), "w") as session_file:
        json.dump(structure, session_file, indent=2)

    return return_dict

# endregion


class FileProvider(Provider):
    """File-based service provider."""

    def __init__(self, folder: str, results_folder: str):
        """Initialize."""
        self.folder = folder
        self.results_folder = results_folder
        os.makedirs(results_folder, exist_ok=True)

    def get_test_metadata(self, session_id: str, test_id: str, api_call: bool = True) -> Dict[str, Any]:
        """Get test metadata"""
        metadata_location = glob.glob(
            os.path.join(self.folder, "**", f"{test_id}_metadata.json"), recursive=True
        )

        if len(metadata_location) == 0:
            raise ServerError(
                "metadata_not_found",
                f"Metadata file for Test Id {test_id} could not be found",
                traceback.format_stack(),
            )

        hints = []

        # List of metadata vars approved to be sent to the client
        approved_metadata = [
            "protocol",
            "known_classes",
            "max_novel_classes",
            "round_size"
        ]

        if session_id is not None:
            structure = get_session_info(self.results_folder, session_id)
            hints = structure.get('activity',{}).get('created', {}).get('hints',[])

        approved_metadata.extend([data for data in ["red_light"] if data in hints])

        with open(metadata_location[0], "r") as md:
            if api_call:
                return {
                    k: v for k, v in json.load(md).items()
                    if k in approved_metadata
                }
            else:
                return json.load(md)

    def test_ids_request(
        self, protocol: str, domain: str, detector_seed: str, test_assumptions: str
    ) -> Dict[str, str]:
        def _strip_id(filename):
            return os.path.splitext(os.path.basename(filename))[0]
        """Request test IDs."""
        file_location = os.path.join(self.folder, protocol, domain, "test_ids.csv")
        if not os.path.exists(file_location):
            if not os.path.exists(os.path.join(self.folder, protocol)):
                msg = f"{protocol} not configured"
            elif not os.path.exists(os.path.join(self.folder, protocol, domain)):
                msg = f"domain {domain} for {protocol} not configured"
            else:
                test_ids = [_strip_id(f) for f in glob.glob(os.path.join(self.folder, protocol, domain,'*.csv'))]
                return {"test_ids": test_ids, "generator_seed": "1234"}
            raise ProtocolError(
                "BadDomain",
                msg,
                traceback.format_stack(),
            )

        return {"test_ids": file_location, "generator_seed": "1234"}

    def new_session(
        self, test_ids: List[str], protocol: str, novelty_detector_version: str, hints: List[str]
    ) -> str:
        """Create a session."""
        # Verify's that all given test id's are valid and have associated csv files
        for test_id in test_ids:
            file_locations = glob.glob(
                os.path.join(self.folder, protocol, "**", f"{test_id}.csv"), recursive=True
            )
            if len(file_locations) == 0:
                raise ServerError(
                    "test_id_invalid",
                    f"Test Id {test_id} could not be matched to a specific file",
                    traceback.format_stack(),
                )

            domain = (file_locations[0]).split(os.path.sep)[-2]

        session_id = str(uuid.uuid4())

        log_session(
            self.results_folder,
            session_id,
            activity="created",
            content={
                "protocol": protocol,
                "domain": domain,
                "detector": novelty_detector_version,
                "hints": hints
            },
        )

        return session_id

    def dataset_request(self, session_id: str, test_id: str, round_id: int) -> FileResult:
        """Request a dataset."""
        file_locations = glob.glob(
            os.path.join(self.folder, "**", f"{test_id}.csv"), recursive=True
        )
        if len(file_locations) == 0:
            raise ServerError(
                "test_id_invalid",
                f"Test Id {test_id} could not be matched to a specific file",
                traceback.format_stack(),
            )

        metadata = self.get_test_metadata(session_id, test_id, False)

        if round_id is not None:
            temp_file_path = BytesIO()

            with open(file_locations[0], "r") as f:
                lines = f.readlines()
                lines = [x for x in lines if x.strip("\n\t\"',.") != ""]
                try:
                    round_pos = int(round_id) * int(metadata["round_size"])
                except KeyError:
                    raise RoundError(
                        "no_defined_rounds",
                        f"round_size not defined in metadata for test id {test_id}",
                        traceback.format_stack(),
                    )
                if round_pos >= len(lines):
                    raise RoundError(
                        "round_id_invalid",
                        f"Round id {str(round_id)} is out of scope for test id {test_id}. Check the metadata round_size.",  # noqa: E501,
                        traceback.format_stack(),
                    )
                temp_file_path.write(''.join(lines[round_pos:round_pos + int(metadata["round_size"])]).encode('utf-8'))
                temp_file_path.seek(0)
        else:
            temp_file_path = open(file_locations[0], 'rb')

        log_session(
            self.results_folder,
            session_id,
            test_id=test_id,
            round_id=round_id,
            activity="data_request",
        )

        return temp_file_path

    # Sets up the various feedback algorithms that can be used with
    # this implementation of FileProvider
    # {
    #   domain: {
    #       feedback_type: {
    #           function: ...
    #           files: [...]
    #       }
    #   }
    # }
    feedback_request_mapping = {
        "image_classification" : {
            ProtocolConstants.CLASSIFICATION:  {
                "function" : get_classification_feedback,
                "files" : [ProtocolConstants.DETECTION, ProtocolConstants.CLASSIFICATION]
            },
            ProtocolConstants.PSUEDO_CLASSIFICATION: {
                "function" : psuedo_label_feedback,
                "files" : [ProtocolConstants.CLASSIFICATION]
            }
        },
        "transcripts" : {
            ProtocolConstants.CLASSIFICATION:  {
                "function" : get_cluster_feedback,
                "files" : [ProtocolConstants.CLASSIFICATION]
            },
            ProtocolConstants.TRANSCRIPTION: {
                "function": get_levenshtein_feedback,
                "files": [ProtocolConstants.TRANSCRIPTION]
            },
            ProtocolConstants.CHARACTERIZATION: {
                "function": get_cluster_feedback,
                "files": [ProtocolConstants.CHARACTERIZATION]
            },
            ProtocolConstants.PSUEDO_CLASSIFICATION: {
                "function" : psuedo_label_feedback,
                "files" : [ProtocolConstants.CLASSIFICATION]
            }
        },
        "activity" : {
            ProtocolConstants.CLASSIFICATION:  {
                "function" : get_cluster_feedback,
                "files" : [ProtocolConstants.CLASSIFICATION]
            },
            ProtocolConstants.TEMPORAL: {
                "function": get_cluster_feedback,
                "files": [ProtocolConstants.TEMPORAL]
            },
            ProtocolConstants.SPATIAL: {
                "function": get_cluster_feedback,
                "files": [ProtocolConstants.SPATIAL]
            },
            ProtocolConstants.PSUEDO_CLASSIFICATION: {
                "function" : psuedo_label_feedback,
                "files" : [ProtocolConstants.CLASSIFICATION]
            }
        }
    }

    def get_feedback(
        self,
        feedback_ids: List[str],
        feedback_type: str,
        session_id: str,
        test_id: str,
        round_id: int,
    ) -> BytesIO:
        """Get feedback of the specified type"""
        metadata = self.get_test_metadata(session_id, test_id, False)
        structure = get_session_info(self.results_folder, session_id)

        # Gets the amount of ids already requested for this type of feedback this round and 
        # determines whether the limit has alreayd been reached
        try:
            feedback_count = structure["activity"]["get_feedback"]["tests"][test_id]["rounds"][round_id].get(feedback_type, 0)
            if feedback_count >= metadata["feedback_max_ids"]:
                raise ProtocolError(
                    "FeedbackBudgetExceeded", 
                    f"Feedback of type {feedback_type} has already been requested on the maximum number of ids",
                    traceback.format_exc()
                )
        except KeyError:
            feedback_count = 0

        # Makes sure that feedback isnt gathered on more than the allowed number of ids per round
        metadata["feedback_max_ids"] -= feedback_count
        if "feedback_max_ids" in metadata and feedback_ids is not None:
            if len(feedback_ids) > metadata["feedback_max_ids"]:
                feedback_ids = feedback_ids[0:metadata["feedback_max_ids"]]

        # Ensure feedback type works with session domain
        # and if so, grab the proper files
        domain = metadata["domain"]
        if domain in self.feedback_request_mapping:
            try:
                file_types = self.feedback_request_mapping[domain][feedback_type]["files"]
            except:
                raise ProtocolError(
                    "InvalidFeedbackType", 
                    f"Invalid feedback type requested for the test id {test_id} with domain {domain}",
                    traceback.format_stack(),
                )
            ground_truth_files = []
            for t in file_types:
                ground_truth_files.extend(glob.glob(
                    os.path.join(self.folder, "**", f"{test_id}_{t}.csv"),
                    recursive=True,
                ))

                if len(ground_truth_files) < 1:
                    raise ServerError(
                        "test_id_invalid",
                        f"Could not find ground truth file(s) for test Id {test_id} with feedback type {feedback_type}",  # noqa: E501
                        traceback.format_stack(),
                    )

            results_files = []
            for t in file_types:
                results_files.extend(glob.glob(
                    os.path.join(self.results_folder,"**",f"{str(session_id)}.{str(test_id)}_{t}.csv"),
                    recursive=True,
                ))
        else:
            raise ProtocolError(
                "BadDomain", 
                f"The set domain does not match a proper domain type. Please check the metadata file for test {test_id}",
                traceback.format_stack(),
            )


        # Check to make sure the round id being requested is both the latest and the highest round submitted
        try:
            if structure["activity"]["post_results"]["tests"][test_id]["last round"] != str(round_id):
                raise RoundError("NotLastRound", "Attempted to get feedback on an older round. Feedback can only be retrieved on the most recent round submission.")
            
            rounds_subbed = [int(r) for r in structure["activity"]["post_results"]["tests"][test_id]["rounds"].keys()]
            if int(round_id) != max(rounds_subbed):
                raise RoundError("NotMaxRound", "Attempted to get feedback on a round that wasn't the max round submitted (most likely a resubmitted round).")
        except RoundError as e:
            raise e
        except Exception as e:
            raise RoundError("SessionLogError", "Error checking session log for round history. Ensure results have been posted before requesting feedback")
        
        if len(results_files) < 1:
            raise ServerError(
                "result_file_not_found",
                f"No result file(s) could be found in {self.results_folder}. Make sure results are posted before calling get feedback.",  # noqa: E501
                traceback.format_stack(),
            )


        # Get feedback from specified test
        try:
            if "psuedo" in feedback_type:
                feedback = psuedo_label_feedback(
                    ground_truth_files,
                    feedback_ids,
                    self.feedback_request_mapping[domain][feedback_type]["files"][0],
                    metadata,
                    self.results_folder,
                    session_id,
                    round_id
                )
            else:
                feedback = self.feedback_request_mapping[domain][feedback_type]["function"](
                    ground_truth_files,
                    results_files,
                    feedback_ids,
                    metadata,
                    round_id
                )
        except KeyError:
            raise ProtocolError(
                "feedback_type_invalid",
                f"Feedback type {feedback_type} is not valid. Make sure the provider's feedback_algorithms variable is properly set up",  # noqa: E501
                traceback.format_exc()
            )

        # Log call
        if feedback_ids is None:
            feedback_count += metadata["feedback_max_ids"]
        else:
            feedback_count += len(feedback_ids)

        log_session(
            self.results_folder,
            session_id=session_id,
            activity="get_feedback",
            test_id=test_id,
            round_id=round_id,
            content={feedback_type: feedback_count},
        )

        feedback_csv = BytesIO()
        if feedback_type == "cluster":
            feedback_csv.write("cluster".encode('utf-8'))
            for val in feedback:
                feedback_csv.write(f",{val}".encode('utf-8'))
            feedback_csv.write("\n".encode('utf-8'))
        else:
            for key in feedback.keys():
                if type(feedback[key]) is not list:
                    feedback_csv.write(f"{key},{feedback[key]}\n".encode('utf-8'))
                else:
                    feedback_csv.write(f"{key},{','.join(str(x) for x in feedback[key])}\n".encode('utf-8'))

        feedback_csv.seek(0)

        return feedback_csv

    def post_results(
        self,
        session_id: str,
        test_id: str,
        round_id: int,
        result_files: Dict[str, str],
    ) -> None:
        """Post results."""
        # Modify session file with posted results
        info = get_session_info(self.results_folder, session_id)
        protocol = info["activity"]["created"]["protocol"]
        domain = info["activity"]["created"]["domain"]
        os.makedirs(os.path.join(self.results_folder, protocol, domain), exist_ok=True)
        log_content = {}
        for r_type in result_files.keys():
            filename = f"{str(session_id)}.{str(test_id)}_{r_type}.csv"
            path = os.path.join(self.results_folder, protocol, domain, filename)
            log_content[f"{r_type} file path"] = path
            lines = []
            if os.path.exists(path):
                with open(path, "r") as result_file:
                    lines = result_file.readlines()
            lines = [x for x in lines if x.strip("\n\t\"',.") != ""]
            with open(path, "w") as result_file:
                result_file.writelines(lines)
                result_file.write(result_files[r_type])

        # Log call
        log_session(
            self.results_folder,
            activity="post_results",
            session_id=session_id,
            test_id=test_id,
            round_id=round_id,
            content=log_content,
        )

    def evaluate(self, session_id: str, test_id: str, round_id: int) -> str:
        """Perform evaluation."""
        # TODO: Get rid of this, evaluate is in separate code base now
        log_session(
            self.results_folder,
            session_id=session_id,
            test_id=test_id,
            round_id=round_id,
            activity="evaluation",
        )

        return os.path.join(self.folder, "evaluation.csv")

    def terminate_session(self, session_id: str) -> None:
        """Terminate the session."""
        # Modify session file to indicate session has been terminated
        log_session(self.results_folder, session_id=session_id, activity="termination")

    def session_status(self, after: str = None, session_id: str = None, include_tests: bool = False) -> str:
        """
        Retrieve Session Names
        :param after: Date Time String lower
        :param include_tests if True, then add the completed test
        :return: CSV of session id and start date time and, termination date time stamp in iso format
        if include tests, then add a second column of tests, thus the format is:
           session_id, test_id, start date time, termination date time stamp
        """
        lower_bound = parser.isoparse(after) if after is not None else None

        session_files_locations = glob.glob(
            os.path.join(os.path.join(self.results_folder, "*.json"))
        )

        results = []
        for session_file in session_files_locations:
            with open(session_file, 'r') as fp:
                session_data = json.load(fp)
                terminated = 'activity' in session_data and 'termination' in session_data['activity']
                created = 'activity' in session_data and 'created' in session_data['activity']
                if terminated:
                    terminate_time = session_data['activity']['termination']['time']
                else:
                    terminate_time = 'Incomplete'
                if created:
                    creation_time = session_data['activity']['created']['time']
                else:
                    creation_time = 'N/A'
                session_name = os.path.splitext(os.path.basename(session_file))[0]
                test_ids = session_data['activity'].get('post_results', {}).get('tests', {}) \
                    if include_tests and created else None
                if (session_id is None and (not lower_bound or
                                            lower_bound <= parser.isoparse(terminate_time))) or \
                        session_name == session_id:
                    if include_tests:
                        if test_ids:
                            for test_id in test_ids:
                                results.append(f'{session_name}, {test_id}, {creation_time},{terminate_time}')
                        else:
                            results.append(f'{session_name}, NA, {creation_time}, {terminate_time}')
                    else:
                        results.append(f'{session_name},{creation_time},{terminate_time}')
        results = sorted(results, key=lambda x: (x.split(',')[1], x.split(',')[0]))
        return '\n'.join(results)

    def get_session_zip(self, session_id) -> str:
        """
        Retrieve Completed Session Names
        :param session_id
        :return: zip file path
        """
        zip_file_name = os.path.join(self.results_folder, f'{session_id}.zip')
        with zipfile.ZipFile(zip_file_name, 'w', compression= zipfile.ZIP_BZIP2) as zip:
            zip.write(os.path.join(self.results_folder, f'{session_id}.json'), arcname=f'{session_id}.json')

            for protocol in os.listdir(self.results_folder):
                if os.path.isdir(os.path.join(self.results_folder, protocol)):
                    for test_file in glob.glob(
                            os.path.join(self.results_folder, protocol, "**", f"{session_id}.*.csv"),
                            recursive=True,
                    ):
                        zip.write(test_file, arcname=test_file[len(self.results_folder) + 1:])

        return zip_file_name