"""Definition of the Provider interface."""

from abc import ABC, abstractmethod

from typing import List, Dict, Any, IO

FileResult = IO

"""
File name and indication of temporary requiring clenaup
"""


class Provider(ABC):
    """
    Abstract base class for serverside service provider functionality.

    Provider is an abstract class that is intended to be inherited, with custom
    implementations, and passed onto the API.

    Methods:
        get_test_metadata: Retrieves the metadata json for the specified test
            Arguments:
                -test_id: Unique identifier for the specific test
                -api_call: bool to tell the function whether its the api calling the
                 function or the file provider calling it locally
            Returns:
                metadata json: Json dictionary of the tests metadata
        test_ids_request: TA 2 Requests for Test Identifiers as part of a series of
          individual tests.
            Arguments:
            -protocol: Empirical protocol that the client would be evaluated against.
            -domain: Problem domain addressed by the novelty detector
            -detector_seed: The seed used to generate the test ids
            -test_assumptions: a json file with assumptions used for the
              tests. Includes:
                -Number of hold out classes
                -A Boolean indicator of presence or absence of novelty at each level.
                 For example,  [True, False,False,False]
            Returns:
                 Dictionary with three entries:
                    -test_ids: CSV file containing all of the appropriate test_ids
                        CSV File Naming Convention has four parts:
                        (Protocol)_(Domain)_(Detector Seed)_(Date/Time)
                    -detection_seed: The detection seed
                    -generator_seed: The generator seed

        new_session: Create a new session to evaluate the detector using an empirical
          protocol
            Arguments:
            -test_ids: CSV file containing the test_ids to be used in the session.
             The set Test IDs may be a partial set of all IDs received for a given
             protocol, domain, etc.
            -protocol: Empirical protocol that the client would be evaluated against.
            -novelty_detector_version: Version of the novelty detector being used
            Returns:
            -session_id: A unique identifier that the server associated with the client

        dataset_request: Request data for evaluation.
            Arguments:
            -session_id: Unique identifier for the client server
            -test_id: Unique identifier for the specific test
            -round_id: Unique identifier for the specific round within the test
            Returns:
            -Dictionary with two entries:
                -dataset_uris: A path to a CSV file of dataset URIs.
                 Each URI identifies the media to be used. It may be location
                 specific (e.g. S3) or independent, assuming a shared repository.
                -num_samples: Number of samples in the dataset

        get_labels: Get Labels from the server based provided one or more example ids
            Arguments:
            -example_ids: List of ids for which to get labels from the server
            -session_id: Unique identifier for the client server
            -test_id: Unique identifier for the specific test
            -round_id: Unique identifier for the specific round within the test
            Returns:
            -Dictionary: image_id -> label id

        post_results: Post client detector predictions for the dataset.
            Arguments:
            -session_id: Unique identifier for the client server
            -test_id: Unique identifier for the specific test
            -round_id: Unique identifier for the specific round within the test
            -detection_file (Optional): csv file of the detection results to be posted
            -characterization_file (Optional): characterization results CSV to be posted
            Note: Either detection_file, characterization_file, or both must be included
            Returns:
             Raises exception on error

        evaluate: Get results for test(s)
            Arguments:
            -session_id: Unique identifier for the client server
            -test_id: Unique identifier for the specific test
            -round_id: Unique identifier for the specific round within the test
            Returns:
           -filename: The filename for the evaluation file

        terminate_session: Terminate the session after evaluation is complete
            Arguments:
            -session_id: Unique identifier for the client server
            Returns:
              Raises exception on error
    """

    @abstractmethod
    def get_test_metadata(self, test_id: str, api_call: bool = True) -> Dict[str, Any]:
        """Get test metadata"""
        pass

    @abstractmethod
    def test_ids_request(
        self, protocol: str, domain: str, detector_seed: str, test_assumptions: str
    ) -> Dict[str, str]:
        """Request test IDs."""
        pass

    @abstractmethod
    def new_session(
        self, test_ids: List[str], protocol: str, novelty_detector_version: str
    ) -> str:
        """Create a new session."""
        pass

    @abstractmethod
    def dataset_request(self, session_id: str, test_id: str, round_id: int) -> FileResult:
        """Request a dataset."""
        pass

    """
        Dictionary of algorithms used for storing functions
        for the various types of feedback.
        All function implementations need to take as params the following:
        -Ground truth file csv reader: reader
        -Result file csv reader: reader
        -Feedback ids: List[str]
        -Metadata: Dict[str, Any]
    """
    feedback_algorithms = {}

    @abstractmethod
    def get_feedback(
        self,
        feedback_ids: List[str],
        feedback_type: str,
        session_id: str,
        test_id: str,
        round_id: int,
    ) -> Dict[str, Any]:
        """Get feedback."""
        pass

    @abstractmethod
    def post_results(
        self,
        session_id: str,
        test_id: str,
        round_id: int,
        result_files: Dict[str, str],
    ) -> None:
        """Post results."""
        pass

    @abstractmethod
    def evaluate(self, session_id: str, test_id: str, round_id: int) -> str:
        """Perform evaluation."""
        pass

    @abstractmethod
    def terminate_session(self, session_id: str) -> None:
        """Terminate a session."""
        pass
