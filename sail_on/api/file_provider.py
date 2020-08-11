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
        activities[activity] = {"time": str(datetime.datetime.now())}
        if content is not None:
            activities[activity].update(content)
    else:
        if activity not in activities:
            activities[activity] = {"time": str(datetime.datetime.now())}
        tests = activities[activity].get("tests", {})
        if test_id not in tests:
            tests[test_id] = {}
        rounds = tests[test_id].get("rounds", {})
        if round_id not in rounds:
            rounds[round_id] = {"time": str(datetime.datetime.now())}
        if content is not None:
            rounds[round_id].update(content)
        tests[test_id]["rounds"] = rounds
        activities[activity]["tests"] = tests
    structure["activity"] = activities
    with open(os.path.join(folder, f"{str(session_id)}.json"), "w") as session_file:
        json.dump(structure, session_file, indent=2)


# region Feedback related functions
def read_feedback_file(
        csv_reader: reader,
        feedback_ids: List[str],
        metadata: Dict[str, Any],
        round_id: int,
) -> Dict[str, str]:
    """
        Gets the feedback data out of the provided
        csv feedback file for the specified ids.
    """
    if round_id is not None:
        try:
            round_pos = int(round_id) * int(metadata["round_size"])
        except KeyError:
            raise RoundError(
                "no_defined_rounds",
                "round_size not defined in metadata.",
                traceback.format_stack(),
            )
        lines = [x for x in csv_reader]
        if round_pos >= len(lines):
            raise RoundError(
                "round_id_invalid",
                f"Round id {str(round_id)} is out of scope. Check the metadata round_size.",  # noqa: E501
                traceback.format_stack(),
            )
        return {
                    x[0]: [float(n) for n in x[1:]]
                    for x in [
                                 [n.strip(" \"'") for n in y] for y in lines
                             ][round_pos:round_pos + int(metadata["round_size"])]
                    if x[0] in feedback_ids
                }
    else:
        return {
            x[0]: [float(n) for n in x[1:]]
            for x in [[n.strip(" \"'") for n in y] for y in csv_reader]
            if x[0] in feedback_ids
        }


def get_classification_feedback(
        gt_reader: reader,
        result_reader: reader,
        feedback_ids: List[str],
        metadata: Dict[str, Any],
        round_id: int,
) -> Dict[str, Any]:
    """Calculates and returns the proper feedback for classification type feedback"""
    ground_truth = read_feedback_file(gt_reader, feedback_ids, metadata, round_id)
    results = read_feedback_file(result_reader, feedback_ids, metadata, round_id)
    return {
        x: 0
        if ground_truth[x][1:].index(max(ground_truth[x][1:])) !=
           results[x][1:].index(max(results[x][1:]))
        else 1
        for x in ground_truth.keys()
    }


def get_detection_feedback(
        gt_reader: reader,
        result_reader: reader,
        feedback_ids: List[str],
        metadata: Dict[str, Any],
        round_id: int,
) -> Dict[str, Any]:
    """Calculates and returns the proper feedback for detection type feedback"""
    threshold = float(metadata["threshold"])

    ground_truth = read_feedback_file(gt_reader, feedback_ids, metadata, round_id)
    results = read_feedback_file(result_reader, feedback_ids, metadata, round_id)

    return {
        x: 0 if abs(ground_truth[x][0] - results[x][0]) > threshold else 1
        for x in ground_truth.keys()
    }


def get_characterization_feedback(
        gt_reader: reader,
        result_reader: reader,
        feedback_ids: List[str],
        metadata: Dict[str, Any],
        round_id: int,
) -> Dict[str, Any]:
    """Calculates and returns the proper feedback for characterization type feedback"""
    known_classes = int(metadata["known_classes"]) + 1

    ground_truth = read_feedback_file(gt_reader, feedback_ids, metadata, round_id)
    results = read_feedback_file(result_reader, feedback_ids, metadata, round_id)

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
# endregion


class FileProvider(Provider):
    """File-based service provider."""

    def __init__(self, folder: str, results_folder: str):
        """Initialize."""
        self.folder = folder
        self.results_folder = results_folder
        os.makedirs(results_folder, exist_ok=True)

    def get_test_metadata(self, test_id: str, api_call: bool = True) -> Dict[str, Any]:
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

        # List of metadata vars approved to be sent to the client
        approved_metadata = [
            "protocol",
            "known_classes",
            "max_novel_classes",
            "round_size",
            "red_light"
        ]

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
        """Request test IDs."""
        file_location = os.path.join(self.folder, protocol, domain, "test_ids.csv")
        return {"test_ids": file_location, "generator_seed": "1234"}

    def new_session(
        self, test_ids: List[str], protocol: str, novelty_detector_version: str
    ) -> str:
        """Create a session."""
        # Verify's that all given test id's are valid and have associated csv files
        for test_id in test_ids:
            file_locations = glob.glob(
                os.path.join(self.folder, "**", f"{test_id}.csv"), recursive=True
            )
            if len(file_locations) != 1:
                raise ServerError(
                    "test_id_invalid",
                    f"Test Id {test_id} could not be matched to a specific file",
                    traceback.format_stack(),
                )

            domain = (file_locations[0]).split(os.path.sep)[-2]
            found_protocol = (file_locations[0]).split(os.path.sep)[-3]
            if found_protocol != protocol:
                raise ValueError(
                    f"Test file not associated with given protocol {protocol}"
                )

        session_id = str(uuid.uuid4())

        log_session(
            self.results_folder,
            session_id,
            activity="created",
            content={
                "protocol": protocol,
                "domain": domain,
                "detector": novelty_detector_version,
            },
        )

        return session_id

    def dataset_request(self, session_id: str, test_id: str, round_id: int) -> FileResult:
        """Request a dataset."""
        file_locations = glob.glob(
            os.path.join(self.folder, "**", f"{test_id}.csv"), recursive=True
        )
        if len(file_locations) != 1:
            raise ServerError(
                "test_id_invalid",
                f"Test Id {test_id} could not be matched to a specific file",
                traceback.format_stack(),
            )

        metadata = self.get_test_metadata(test_id, False)

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
            temp_file_path = open(file_locations[0],'rb')

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
    feedback_algorithms = {
        ProtocolConstants.CLASSIFICATION: get_classification_feedback,
        ProtocolConstants.DETECTION: get_detection_feedback,
        ProtocolConstants.CHARACTERIZATION: get_characterization_feedback,
    }

    def get_feedback(
        self,
        feedback_ids: List[str],
        feedback_type: str,
        session_id: str,
        test_id: str,
        round_id: int,
    ) -> Dict[str, Any]:
        """Get feedback of the specified type"""
        # Find label file for specified test
        file_locations = glob.glob(
            os.path.join(self.folder, "**", f"{test_id}_{feedback_type}.csv"),
            recursive=True,
        )

        if len(file_locations) != 1:
            raise ServerError(
                "test_id_invalid",
                f"Test Id {test_id} could not be matched to a specific ground truth file for feedback type {feedback_type}",  # noqa: E501
                traceback.format_stack(),
            )

        metadata = self.get_test_metadata(test_id, False)

        results_files = glob.glob(
            os.path.join(
                self.results_folder,
                "**",
                f"{str(session_id)}.{str(test_id)}_{feedback_type}.csv"
            ),
            recursive=True,
        )

        if len(results_files) < 1:
            raise ServerError(
                "result_file_not_found",
                f"No result file(s) could be found in {self.results_folder}. Make sure results are posted before calling get feedback.",  # noqa: E501
                traceback.format_stack(),
            )

        # Get feedback from specified test
        with open(file_locations[0], "r") as f:
            gt_reader = csv.reader(f, delimiter=",")
            with open(results_files[0], "r") as rf:
                result_reader = csv.reader(rf, delimiter=",")

                try:
                    feedback = self.feedback_algorithms[feedback_type](
                        gt_reader,
                        result_reader,
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
        count = len(feedback)
        log_session(
            self.results_folder,
            session_id=session_id,
            activity="get_feedback",
            test_id=test_id,
            round_id=round_id,
            content={feedback_type: feedback_ids},
        )

        # Update total count
        structure = get_session_info(self.results_folder, session_id)
        total_count = structure["activity"]["get_feedback"]["tests"][test_id].get(
            "total count", 0
        )
        total_count += count
        structure["activity"]["get_feedback"]["tests"][test_id][
            "total count"
        ] = total_count
        with open(
            os.path.join(self.results_folder, f"{str(session_id)}.json"), "w"
        ) as session_file:
            json.dump(structure, session_file, indent=2)

        temp_csv_path = BytesIO()

        for key in feedback.keys():
            temp_csv_path.write(f"{key},{feedback[key]}\n".encode('utf-8'))

        temp_csv_path.seek(0)

        return temp_csv_path

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
        # TODO: FOR NOW, use true/false predictions of novelty to calculate
        # precision, accuracy, recall and F1 Score
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
