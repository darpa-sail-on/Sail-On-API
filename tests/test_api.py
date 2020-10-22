"""Unit tests for server, and file_provider modules and related functions."""
import os
import io
import requests
import json
import unittest
import uuid
import threading
from sail_on.api import server, errors, ProtocolConstants
from sail_on.api.file_provider import FileProvider
import shutil
import time
from requests_toolbelt.multipart import decoder

from typing import Any, Generator, Optional, Dict
from requests import Response
from json import JSONDecodeError

# Helpers


def _check_response(response: Response) -> None:
    """
    Raise the appropriate ApiError based on response error code.

    :param response:
    :return: True
    """
    if response.status_code != 200:
        try:
            response_json = response.json()
            # Find the appropriate error class based on error code.
            for subclass in errors.ApiError.error_classes():
                if subclass.error_code == response.status_code:
                    raise subclass(
                        response_json["reason"],
                        response_json["message"],
                        response_json["stack_trace"],
                    )
        except JSONDecodeError:
            raise errors.ServerError("Unknown", response.content.decode("UTF-8"), "")


def get(path: str, **params: Dict[str, Any]) -> Response:
    return requests.get(f"http://localhost:12345{path}", **params)


def post(path: str, **params: Dict[str, Any]) -> Response:
    return requests.post(f"http://localhost:12345{path}", **params)


def delete(path: str, **params: Dict[str, Any]) -> Response:
    return requests.delete(f"http://localhost:12345{path}", **params)


SERVER_RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), f"server_results_unit_tests")


class TestApi(unittest.TestCase):
    """Test the API."""

    @classmethod
    def tearDownClass(cls) -> None:
        """Tear down test fixtures."""
        shutil.rmtree(SERVER_RESULTS_DIR)

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        if os.path.exists(SERVER_RESULTS_DIR):
            shutil.rmtree(SERVER_RESULTS_DIR)
        os.mkdir(SERVER_RESULTS_DIR)
        server.set_provider(
            FileProvider(os.path.join(os.path.dirname(
                __file__), "data"), SERVER_RESULTS_DIR)
        )
        api_thread = threading.Thread(target=server.init, args=("localhost", 12345))
        api_thread.daemon = True
        api_thread.start()
        directory = os.path.join(os.path.dirname(__file__), "session_state_files")
        for filename in os.listdir(directory):
            shutil.copy(os.path.join(directory, filename),
                        os.path.join(SERVER_RESULTS_DIR))

    # Test Ids Request Tests
    def test_test_ids_request_success(self):
        """Test test_ids_request."""
        payload = {
            "protocol": "OND",
            "domain": "image_classification",
            "detector_seed": "5678",
        }

        assumptions_path = os.path.join(os.path.dirname(__file__), "assumptions.json")
        with open(assumptions_path, "r") as f:
            contents = f.read()

        response = get(
            "/test/ids",
            files={
                "test_requirements": io.StringIO(json.dumps(payload)),
                "test_assumptions": io.StringIO(contents),
            },
        )

        _check_response(response)
        expected = f"OND.1.1.1234{os.linesep}"
        actual = response.content.decode("utf-8")
        self.assertEqual(expected, actual)

    # Session Request Tests
    def test_session_request_success(self):
        """Test session_request."""
        path = os.path.join(
            os.path.dirname(__file__), "data/OND/image_classification/test_ids.csv"
        )
        payload = {
            "protocol": "OND",
            "novelty_detector_version": "0.1.1",
        }

        with open(path, "r") as f:
            contents = f.read()

        response = post(
            "/session",
            files={
                "test_ids": io.StringIO(contents),
                "configuration": io.StringIO(json.dumps(payload)),
            },
        )

        _check_response(response)
        session_id = response.json()["session_id"]

        self.assertEqual(session_id, str(uuid.UUID(session_id)))
        self.assertTrue(os.path.exists(os.path.join(
            SERVER_RESULTS_DIR, f"{session_id}.json")))

    # Dataset Request Tests
    def test_dataset_request_success_with_round_id(self):
        """Test dataset request with rounds."""
        response = get(
            "/session/dataset",
            params={"session_id": "data_request",
                    "test_id": "OND.1.1.1234", "round_id": 0},
        )

        _check_response(response)
        expected = "n01484850_18013.JPEG\nn01484850_24624.JPEG\n"
        actual = response.content.decode("utf-8")
        self.assertEqual(expected, actual)

    def test_dataset_request_success_without_round_id(self):
        """Test dataset request without rounds."""
        response = get(
            "/session/dataset",
            params={"session_id": "data_request", "test_id": "OND.1.1.1234"},
        )

        _check_response(response)
        expected = f"n01484850_18013.JPEG{os.linesep}n01484850_24624.JPEG{os.linesep}n01484850_4515.JPEG{os.linesep}n01484850_45289.JPEG{os.linesep}"
        actual = response.content.decode("utf-8")
        self.assertEqual(expected, actual)

    # Get feedback Tests
    def test_get_feedback_failure_invalid_type(self):
        """Test get_feedback with invalid type for domain."""
        response = get(
            "/session/feedback",
            params={
                "feedback_ids": "|".join(["n01484850_4515.JPEG", "n01484850_45289.JPEG"]),
                "feedback_type": ProtocolConstants.DETECTION,
                "session_id": "get_feedback",
                "test_id": "OND.1.1.1234",
                "round_id": 0
            },
        )
        try:
            _check_response(response)
        except errors.ApiError as e:
            self.assertEqual('InvalidFeedbackType', e.reason)

    def test_get_feedback_success_multiple_types(self):
        """Test get_feedback with multiple types."""
        feedback_types = [ProtocolConstants.CLASSIFICATION,
                          ProtocolConstants.CHARACTERIZATION]
        response = get(
            "/session/feedback",
            params={
                "feedback_type": "|".join(feedback_types),
                "session_id": "get_feedback",
                "test_id": "OND.1.1.1234",
                "round_id": 1,
            },
        )

        _check_response(response)
        multipart_data = decoder.MultipartDecoder.from_response(response)
        result_dicts = []
        for i in range(len(feedback_types)):
            header = multipart_data.parts[i].headers[b"Content-Disposition"].decode(
                "utf-8")
            header_dict = {
                x[0].strip(): x[1].strip(" \"'")
                for x in [part.split("=") for part in header.split(";") if "=" in part]
            }
            result_dicts.append(header_dict)

        expected = ["nmi,1.0\n0,1.0\n2,1.0\n", "nmi,1.0\n0,1.0\n"]
        actual = []
        for i, part in enumerate(multipart_data.parts):
            actual = part.content.decode("utf-8")
            print('-----')
            print(actual)
            print('-----')
            self.assertEqual(expected[i], actual)

        for i, head in enumerate(result_dicts):
            self.assertEqual(feedback_types[i], head["name"])
            self.assertEqual(
                f"get_feedback.OND.1.1.1234.1_{feedback_types[i]}.csv", head["filename"])

    def test_get_feedback_success_classification(self):
        """Test get_feedback with classification."""
        feedback_types = [ProtocolConstants.CLASSIFICATION]
        response = get(
            "/session/feedback",
            params={
                "feedback_type": feedback_types[0],
                "session_id": "get_feedback",
                "test_id": "OND.1.1.1234",
                "round_id": 1,
            },
        )

        _check_response(response)
        expected = "n01484850_4515.JPEG,0\nn01484850_45289.JPEG,2\n"
        actual = response.content.decode("utf-8")
        self.assertEqual(expected, actual)

    def test_get_feedback_success_single_classification(self):
        """Test get_feedback with classification."""
        feedback_types = [ProtocolConstants.CLASSIFICATION]
        response = get(
            "/session/feedback",
            params={
                "feedback_type": feedback_types[0],
                "feedback_ids": ["n01484850_45289.JPEG"],
                "session_id": "get_feedback",
                "test_id": "OND.1.1.1234",
                "round_id": 1,
            },
        )

        _check_response(response)
        expected = "n01484850_45289.JPEG,2\n"
        actual = actual = response.content.decode("utf-8")
        self.assertEqual(expected, actual)

    def test_get_feedback_failure_no_round_id(self):
        """Test get_feedback fails with no round id."""
        response = get(
            "/session/feedback",
            params={
                "feedback_ids": "|".join(
                    ["n01484850_4515.JPEG", "n01484850_45289.JPEG"]),
                "feedback_type": ProtocolConstants.CLASSIFICATION,
                "session_id": "get_feedback",
                "test_id": "OND.1.1.1234",
            },
        )

        try:
            _check_response(response)
        except errors.ApiError as e:
            self.assertEqual('MissingParamsError', e.reason)

    def test_get_feedback_success_cluster(self):
        """Test get_feedback with type classification for cluster function."""
        response = get(
            "/session/feedback",
            params={
                "feedback_type": ProtocolConstants.CLASSIFICATION,
                "session_id": "get_feedback",
                "test_id": "OND.1.1.1234",
                "round_id": 1,
            },
        )

        _check_response(response)
        expected = "nmi,1.0\n0,1.0\n2,1.0\n"
        actual = response.content.decode("utf-8")
        self.assertEqual(expected, actual)

    def test_get_feedback_transcription(self):
        """Test get_feedback with type transcription for levenshtien function."""
        response = get(
            "/session/feedback",
            params={
                "feedback_type": ProtocolConstants.TRANSCRIPTION,
                "session_id": "get_feedback",
                "test_id": "OND.1.1.1234",
                "round_id": 1,
            },
        )

        _check_response(response)
        expected = "n01484850_4515.JPEG,3,0\nn01484850_45289.JPEG,0,0\n"
        actual = response.content.decode("utf-8")
        self.assertEqual(expected, actual)

    def test_get_feedback_only_max_ids(self):
        """Test get_feedback to only grab up to max ids. (Should not attempt to do anything with extra id)"""
        response = get(
            "/session/feedback",
            params={
                "feedback_ids": "|".join(
                    ["n01484850_4515.JPEG", "n01484850_45289.JPEG", "404_id.JPEG"]),
                "feedback_type": ProtocolConstants.TRANSCRIPTION,
                "session_id": "get_feedback",
                "test_id": "OND.1.1.1234",
                "round_id": 1,
            },
        )

        _check_response(response)
        expected = "n01484850_4515.JPEG,3,0\nn01484850_45289.JPEG,0,0\n"
        actual = response.content.decode("utf-8")
        self.assertEqual(expected, actual)

    def test_get_feedback_psuedo_classification(self):
        """Test getting psuedo labels for given ids"""
        response = get(
            "/session/feedback",
            params={
                "feedback_ids": "|".join(
                    ["n01484850_4515.JPEG", "n01484850_45289.JPEG"]),
                "feedback_type": ProtocolConstants.PSUEDO_CLASSIFICATION,
                "session_id": "get_feedback",
                "test_id": "OND.1.1.1234",
                "round_id": 1,
            },
        )

        _check_response(response)
        expected = "n01484850_4515.JPEG,2\nn01484850_45289.JPEG,1\n"
        actual = response.content.decode("utf-8")
        self.assertEqual(expected, actual)

    # Post Results Tests
    def test_post_results_success_with_round_id(self):
        """Test posting results with rounds."""
        result_files = {
            ProtocolConstants.CHARACTERIZATION: os.path.join(
                os.path.dirname(__file__), "test_results_OND.1.1.1234.csv"
            )
        }

        payload = {
            "session_id": "post_results",
            "test_id": "OND.1.1.1234",
            "round_id": 0,
            "result_types": "|".join(result_files.keys())
        }

        files = {"test_identification": io.StringIO(json.dumps(payload))}
        for r_type in result_files:
            with open(result_files[r_type], "r") as f:
                contents = f.read()
                files[f"{r_type}_file"] = io.StringIO(contents)

        response = post("/session/results", files=files)

        # payload["round_id"] = 1
        files = {"test_identification": io.StringIO(json.dumps(payload))}
        for r_type in result_files:
            with open(result_files[r_type], "r") as f:
                contents = f.read()
                files[f"{r_type}_file"] = io.StringIO(contents)

        response = post("/session/results", files=files)

        payload["round_id"] = 1
        files = {"test_identification": io.StringIO(json.dumps(payload))}
        for r_type in result_files:
            with open(result_files[r_type], "r") as f:
                contents = f.read()
                files[f"{r_type}_file"] = io.StringIO(contents)

        response = post("/session/results", files=files)

        _check_response(response)

    def test_post_results_failure_without_round_id(self):
        """Test posting results without rounds."""
        result_files = {
            ProtocolConstants.DETECTION: os.path.join(
                os.path.dirname(__file__), "test_results_OND.1.1.1234.csv"
            )
        }

        payload = {
            "session_id": "post_results",
            "test_id": "OND.1.1.1234",
            "result_types": "|".join(result_files.keys())
        }

        files = {"test_identification": io.StringIO(json.dumps(payload))}
        for r_type in result_files:
            with open(result_files[r_type], "r") as f:
                contents = f.read()
                files[f"{r_type}_file"] = io.StringIO(contents)

        response = post("/session/results", files=files)

        try:
            _check_response(response)
        except errors.ApiError as e:
            self.assertEqual('MissingParamsError', e.reason)

    def test_post_results_success_with_two_files(self):
        """Test posting results with two files."""
        result_files = {
            ProtocolConstants.DETECTION: os.path.join(
                os.path.dirname(__file__), "test_results_OND.1.1.1234.csv"
            ),
            ProtocolConstants.CHARACTERIZATION: os.path.join(
                os.path.dirname(__file__), "test_results_OND.1.1.1234.csv"
            ),
        }

        payload = {
            "session_id": "post_results",
            "test_id": "OND.1.1.1234",
            "round_id": 0,
            "result_types": "|".join(result_files.keys())
        }

        files = {"test_identification": io.StringIO(json.dumps(payload))}
        for r_type in result_files:
            with open(result_files[r_type], "r") as f:
                contents = f.read()
                files[f"{r_type}_file"] = io.StringIO(contents)

        response = post("/session/results", files=files)

        _check_response(response)

    # Evaluation Tests
    def test_evaluate_success_with_round_id(self):
        """Test evaluate with rounds."""
        response = get(
            "/session/evaluations",
            params={"session_id": "evaluation",
                    "test_id": "OND.1.1.1234", "round_id": 0},
        )

        _check_response(response)

        expected = "score, 0"
        actual = response.content.decode("utf-8")
        self.assertEqual(expected, actual)

    def test_evaluate_success_without_round_id(self):
        """Test evaluate without rounds."""
        response = get(
            "/session/evaluations",
            params={"session_id": "evaluation", "test_id": "OND.1.1.1234"},
        )

        _check_response(response)

        expected = "score, 0"
        actual = response.content.decode("utf-8")
        self.assertEqual(expected, actual)

    # Terminate Session Tests
    def test_terminate_session_success(self):
        """Test terminate_session."""
        response = delete("/session", params={"session_id": "termination"})

        _check_response(response)

    def test_get_metadata_success(self):
        response = get(
            "/test/metadata",
            params={"test_id": "OND.1.1.1234"},
        )

        _check_response(response)
        metadata = response.json()

        self.assertEqual("OND", metadata["protocol"])
        self.assertEqual(3, metadata["known_classes"])
        self.assertFalse("threshold" in metadata)
