"""This file contains a Flask REST API used for communicating between frameworks."""


from .errors import ServerError, ProtocolError, ApiError, RoundError
from flask import Flask, request, send_file
import json
import os
import logging
import sys
import traceback
import argparse
from flask import Response, Request
from .provider import Provider
from requests_toolbelt import MultipartEncoder

from typing import Union, Tuple, Optional, Dict, Any

app = Flask(__name__)
app.config.from_object(__name__)


class Binder:
    """Bind the provider to the api."""

    provider: Provider


def get_provider() -> Provider:
    """Return the current provider."""
    return Binder.provider


def set_provider(provider: Provider) -> None:
    """Set a provider."""
    logging.info(f"Provider set to {str(type(provider))}")
    Binder.provider = provider


def init(hostname: str, port: int) -> None:
    """Launch the application with a service provider."""
    # Note: Initializing the api will block the thread from continuing until it
    # is disabled. This should be the last call in the thread.
    if get_provider() is not None:
        app.run(host=hostname, port=port)
        logging.info(f"Api Server successfully started at {hostname}:{port}")
    else:
        raise ServerError(
            "NoProviderError", "Unable to start api because provider not set"
        )


def get_from_request(
    req: Request, item: str, default: Optional[str] = ""
) -> Optional[str]:
    """Retrieve an item from various places within a request."""
    if item in req.form:
        return req.form[item]
    elif item in req.files:
        return req.files[item].read().decode("utf-8")
    return default


@app.route("/test/metadata", methods=["GET"])
def get_test_metadata() -> Response:
    """
    Retrieves the metadata json for the specified test
    Arguments:
        -test_id
    Returns:
        metadata json
    """
    data = request.args
    try:
        session_id = data.get("session_id", None)
        test_id = data["test_id"]
        logging.info(f"Retrieved metadata for Session ID {session_id} Test Id {test_id}")
    except KeyError:
        raise ProtocolError(
            "MissingParamsError",
            "test/metadata requires test_id",
            traceback.format_exc(),
        )
    try:
        response = Binder.provider.get_test_metadata(session_id, test_id)
        logging.info(f"Returning metadata: {response}")
        return response
    except ServerError as e:
        raise e
    except ProtocolError as e:
        raise e
    except Exception as e:
        raise ServerError(str(type(e)), str(e), traceback.format_exc())


@app.route("/test/ids", methods=["GET"])
def test_ids_request() -> Response:
    """
    Request Test Identifiers as part of a series of individual tests.

    Arguments:
        -protocol
        -domain
        -detector_seed
        -test_assumptions
    Returns:
        -test_ids
        -detection_seed
    """

    try:
        val = get_from_request(request, "test_requirements", default=None)
        if val is None:
            protocol = get_from_request(request, "protocol", default=None)
            domain = get_from_request(request, "domain", default="image_classification")
            seed = get_from_request(request, "detector_seed")
        else:
            data = json.loads(val)
            protocol = data["protocol"]
            domain = data["domain"]
            seed = data["detector_seed"]

        if protocol is None:
            raise ProtocolError(
                "MissingRequestValue",
                "test_requirements or protocol missing from request object",
                traceback.format_exc(),
            )

        test_assumptions = get_from_request(request, "test_assumptions", default="{}")
        logging.info(
            f"TestIdsRequest called with protocol: {protocol} domain: {domain} detector_seed: {seed} test_assumptions: {test_assumptions}"  # noqa: E501
        )
    except KeyError:
        raise ProtocolError(
            "MissingParamsError",
            "TestIdsRequest requires protocol, domain, detector_seed, and test_assumptions",  # noqa: E501
            traceback.format_exc(),
        )

    try:
        if test_assumptions is None:
            raise ProtocolError(
                "MissingRequestValue",
                "test_assumptions missing from request object",
                traceback.format_exc(),
            )

        response = Binder.provider.test_ids_request(
            protocol, domain, seed, test_assumptions
        )
    except ServerError as e:
        raise e
    except ProtocolError as e:
        raise e
    except Exception as e:
        raise ServerError(str(type(e)), str(e), traceback.format_exc())

    # Ensures that the provider response has the correct parameters
    if "test_ids" in response and "generator_seed" in response:

        # Generates the file to be returned
        try:
            logging.info(f'Returning test_ids at file path: {response["test_ids"]}')
            return send_file(
                response["test_ids"],
                attachment_filename=f'{protocol}_{domain}_{response["generator_seed"]}.csv',
                as_attachment=True,
                mimetype="test/csv",
            )
        except Exception as e:
            raise ServerError(str(type(e)), str(e), traceback.format_exc())
    else:
        raise ServerError(
            "ProviderSetupError",
            "Provider not properly set up. Must return a csv file for test_ids, the detection_seed, and the generator_seed.",  # noqa: E501
        )


@app.route("/session", methods=["POST"])
def new_session() -> Dict[str, str]:
    """
    Create a new session to evaluate the detector using an empirical protocol.

    Arguments:
        -test_ids
        -protocol
        -novelty_detector_version
    Returns:
        -session_id
    """

    # Attempts to retrieve the proper variables from the API call body,
    # and passes them to the provider function
    try:
        val = get_from_request(request, "configuration", default="{}")
        if val is None:
            raise ProtocolError(
                "MissingRequestValue",
                "configuration missing from request object",
                traceback.format_exc(),
            )
        data = json.loads(val)
        protocol = data["protocol"]
        novelty_version = data["novelty_detector_version"]
        hints = data.get("hints", [])

        reader = request.files["test_ids"].read().decode("utf-8").split("\n")
        test_ids = [x.strip(" \"',") for x in filter(lambda x: x != "", reader)]
        logging.info(
            f"NewSession called with test_ids: {test_ids} protocol: {protocol} novelty_detector_version: {novelty_version}"  # noqa: E501
        )
    except KeyError:
        raise ProtocolError(
            "MissingParamsError",
            "NewSession requires test_ids, protocol, and novelty_detector_version",
            traceback.format_exc(),
        )

    if len(test_ids) == 0:
        raise ProtocolError("EmptyFile", "Test Ids file was empty")

    try:
        response = Binder.provider.new_session(test_ids, protocol, novelty_version, hints)

        logging.info(f"Returning session_id: {response}")
        return {"session_id": response}
    except ServerError as e:
        raise e
    except ProtocolError as e:
        raise e
    except Exception as e:
        raise ServerError(str(type(e)), str(e), traceback.format_exc())


@app.route("/session/dataset", methods=["GET"])
def dataset_request() -> Response:
    """
    Request data for evaluation.

    Arguments:
        -session_id
        -test_id
        -round_id
    Returns:
        -dataset_uris
        -num_samples
    """

    # Attempts to retrieve the proper variables from the API call body,
    # and passes them to the provider function
    data = request.args
    try:
        session_id = data["session_id"]
        test_id = data["test_id"]
        round_id = data.get("round_id")
        logging.info(
            f"DatasetRequest called with session_id: {session_id} test_id: {test_id} round_id: {round_id}"  # noqa: E501
        )
    except KeyError:
        raise ProtocolError(
            "MissingParamsError",
            "DatasetRequest requires session_id, test_id, and round_id",
            traceback.format_exc(),
        )

    try:
        file_name = Binder.provider.dataset_request(session_id, test_id, round_id)
    except ServerError as e:
        raise e
    except ProtocolError as e:
        raise e
    except Exception as e:
        raise ServerError(str(type(e)), str(e), traceback.format_exc())

    # returns the file
    try:
        logging.info(f"Returning dataset_uris at file path: {file_name}")
        return send_file(file_name, attachment_filename=f'{session_id}.{test_id}.{round_id}.csv', mimetype="test/csv")
    except Exception as e:
        raise ServerError(str(type(e)), str(e), traceback.format_exc())


@app.route("/session/feedback", methods=["GET"])
def get_feedback() -> Response:
    """
    Gets Feedback of the specified type from the server provided one or more label ids
    Arguments:
        -session_id
        -test_id
        -round_id
        -feedback_ids
        -feedback_type: detection, characterization, label
    Returns:
        -feedback dictionary
    """

    # Attempts to retrieve the proper variables from the API call body,
    # and passes them to the provider function
    data = request.args
    try:
        session_id = data["session_id"]
        test_id = data["test_id"]
        round_id = data["round_id"]
        feedback_types = data["feedback_type"].split("|")
        feedback_ids = data.get("feedback_ids", "")
        if feedback_ids == "":
            feedback_ids = None
        else:
            feedback_ids = feedback_ids.split("|")
        logging.info(
            f"GetFeedback called with session_id: {session_id} test_id: {test_id} round_id: {round_id} feedback_ids: {feedback_ids} feedback_types: {feedback_types}"  # noqa: E501
        )
    except KeyError:
        raise ProtocolError(
            "MissingParamsError",
            "GetFeedback requires feedback type(s), session_id, test_id, and round id",
            traceback.format_exc(),
        )

    try:
        responses = {}
        for f_type in feedback_types:
            responses[f_type] = Binder.provider.get_feedback(feedback_ids, f_type, session_id, test_id, round_id)
    except RoundError as e:
        raise e
    except ServerError as e:
        raise e
    except ProtocolError as e:
        raise e
    except Exception as e:
        raise ServerError(str(type(e)), str(e), traceback.format_exc())

    # returns the file(s)
    try:
        logging.info(f"Returning feedback at file path(s): {responses}")
        if len(feedback_types) > 1:
            m = MultipartEncoder({
                key: (
                    f"{session_id}.{test_id}.{round_id}_{key}.csv",
                    responses[key],
                    "text/csv",
                ) 
                for key in responses
            })
            
            return Response(m.to_string(), content_type=m.content_type, status=200)
        else:
            return send_file(responses[feedback_types[0]], attachment_filename=f'{session_id}.{test_id}.{round_id}_{feedback_types[0]}.csv', mimetype="test/csv")
    except Exception as e:
        raise ServerError(str(type(e)), str(e), traceback.format_exc())


@app.route("/session/results", methods=["POST"])
def post_results() -> str:
    """
    Post client detector predictions for the dataset.

    Arguments:
        -session_id
        -test_id
        -round_id
        -result_types
        -result_files
    Returns:
        -OK or error
    """

    # Attempts to retrieve the proper variables from the API call body,
    # and passes them to the provider function
    try:
        val = get_from_request(request, "test_identification", default="{}")
        if val is None:
            raise KeyError
        data = json.loads(val)
        session_id = data["session_id"]
        test_id = data["test_id"]
        round_id = data.get("round_id")
        result_types = data["result_types"].split("|")
        result_files = {}
        for r_type in result_types:
            result_files[r_type] = get_from_request(
                request, f"{r_type}_file", default=None
            )
            if result_files[r_type] is None:
                raise KeyError

        if len(result_files) == 0:
            raise KeyError

        logging.info(
            f"PostResults called with session_id: {session_id} test_id: {test_id} round_id: {round_id} result file types: {result_types}"  # noqa: E501
        )
    except KeyError:
        raise ProtocolError(
            "MissingParamsError",
            "PostResults requires session_id, test_id, round_id, and at least one result file with a matching result_type",  # noqa: E501
            traceback.format_exc(),
        )

    try:
        # Carries the call through to the provider and returns 'OK' if
        # successful
        Binder.provider.post_results(
            session_id, test_id, round_id, result_files
        )
        logging.info("Post Results returning 'OK'")
        return "OK"
    except ServerError as e:
        raise e
    except ProtocolError as e:
        raise e
    except Exception as e:
        raise ServerError(str(type(e)), str(e), traceback.format_exc())


@app.route("/session/evaluations", methods=["GET"])
def evaluate() -> Response:
    """
    Get results for test(s).

    Arguments:
        -session_id
        -test_id
        -round_id (Optional)
    Returns:
        -filename
    """

    # Attempts to retrieve the proper variables from the API call body,
    # and passes them to the provider function
    data = request.args
    try:
        session_id = data["session_id"]
        test_id = data["test_id"]
        round_id = data.get("round_id")
        logging.info(
            f"Evaluate called with session_id: {session_id} test_id: {test_id} round_id: {round_id}"  # noqa: E501
        )
    except KeyError:
        raise ProtocolError(
            "MissingParamsError",
            "Evaluate requires session_id, test_id, and round_id",
            traceback.format_exc(),
        )

    try:
        file_path = Binder.provider.evaluate(session_id, test_id, round_id)
        logging.info(f"Returning file at {file_path}")
        return send_file(
            file_path,
            attachment_filename="evaluation.{}.{}.csv".format(session_id, test_id),
            as_attachment=True,
            mimetype="test/csv",
        )
    except ServerError as e:
        raise e
    except ProtocolError as e:
        raise e
    except Exception as e:
        raise ServerError(str(type(e)), str(e), traceback.format_exc())


@app.route("/session", methods=["DELETE"])
def terminate_session() -> str:
    """
    Terminate the session  after the evaluation for the protocol is complete.

    Arguments:
        -session_id
    Returns:
        -OK or error
    """

    # Attempts to retrieve the proper variables from the API call body,
    # and passes them to the provider function
    data = request.args
    try:
        session_id = data["session_id"]
        logging.info(f"TerminateSession called with session_id: {session_id}")
    except KeyError:
        raise ProtocolError(
            "MissingParamsError",
            "TerminateSession requires session_id",
            traceback.format_exc(),
        )

    try:
        Binder.provider.terminate_session(session_id)
        logging.info("Terminate Session returning 'OK'")
        return "OK"
    except ServerError as e:
        raise e
    except ProtocolError as e:
        raise e
    except Exception as e:
        raise ServerError(str(type(e)), str(e), traceback.format_exc())


@app.route("/session/status", methods=["GET"])
def session_status() -> str:
    """
        Retrieve Session Names

        Arguments:
            -after date time iso formatted string lower bound
            -session_id specific session
            -include_tests provide all tests in session(s)
        Returns:
            CSV of session id and start date time and, termination date time stamp in iso format
          if include tests, then add a second column of tests, thus the format is:
           session_id, test_id, start date time, termination date time stamp
    """

    # Attempts to retrieve the proper variables from the API call body,
    # and passes them to the provider function
    data = request.args
    after = data["after"] if 'after' in data else None
    session_id = data["session_id"] if 'session_id' in data else None
    include_tests = data["include_tests"] if 'include_tests' in data else False
    include_tests = include_tests if type(include_tests) == bool else include_tests.lower() == 'true'

    try:
        return Binder.provider.session_status(after, session_id, include_tests).encode('utf-8')
    except ServerError as e:
        raise e
    except ProtocolError as e:
        raise e
    except Exception as e:
        raise ServerError(str(type(e)), str(e), traceback.format_exc())

@app.route("/session/zip", methods=["GET"])
def session_zip() -> str:
    """
    Produce zip file of test results for session.

    Arguments:
        -session_id date time stamp in iso format
    Returns:
        -zip file of session files
    """

    # Attempts to retrieve the proper variables from the API call body,
    # and passes them to the provider function
    data = request.args
    try:
        session_id = data["session_id"]
        logging.info(f"Zip Session called with session_id: {session_id}")
    except KeyError:
        raise ProtocolError(
            "MissingParamsError",
            "TerminateSession requires session_id",
            traceback.format_exc(),
        )

    try:
        file_path = Binder.provider.get_session_zip(session_id)
        return send_file(
            file_path,
            attachment_filename=f"{session_id}.zip",
            as_attachment=True,
            mimetype="application/zip",
        )
    except ServerError as e:
        raise e
    except ProtocolError as e:
        raise e
    except Exception as e:
        raise ServerError(str(type(e)), str(e), traceback.format_exc())

def main(args: argparse.Namespace) -> None:
    """Run the main application."""
    from . import FileProvider

    set_provider(
        FileProvider(
            os.path.abspath(args.data_directory),
            os.path.abspath(args.results_directory),
        )
    )
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        filename=args.log_file, filemode="w", level=args.log_level, format=log_format
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"Api server starting with provider set to FileProvider")
    init(*args.url.split(":"))


def command_line() -> None:
    """Run the `sail_on_server` command."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        help="(hostname:port) of the server",
        required=False,
        default="localhost:3306",
    )
    parser.add_argument("--log-file", help="File to save log", default="server.log")
    parser.add_argument("--log-level", default=logging.INFO, help="Logging levels")
    parser.add_argument(
        "--data-directory",
        help="test file structure",
        default=".",
        dest="data_directory",
    )
    parser.add_argument(
        "--results-directory",
        help="result file structure",
        default=".",
        dest="results_directory",
    )
    args = parser.parse_args()
    main(args)


@app.errorhandler(ApiError)
def handle_error(e: ApiError) -> Tuple[Dict[str, str], int]:
    """Handle all HTTP error conditions by sending back a canonical response."""
    logging.exception(f"{e.reason} : {e.msg}")
    return e.flask_response()


if __name__ == "__main__":
    command_line()
