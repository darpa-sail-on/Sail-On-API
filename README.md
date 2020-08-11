# Sail On framework
Framework for running a novelty detector using different protocols

## Quick Start
1. In a clean work directory, clone the
   [random_novelty_detector](https://gitlab.kitware.com/darpa-sail-on/random_novelty_detector)
   repository: `git clone https://gitlab.kitware.com/darpa-sail-on/random_novelty_detector`.
2. Next to it, clone this repository, and move into it:
   `git clone https://gitlab.kitware.com/darpa-sail-on/sail-on && cd sail-on`.
3. Ensure that you have Pipenv installed.
4. Install the dependencies: `pipenv install`.
5. Activate the virtual environment `pipenv shell`.
6. Install the random novelty detector in the environment: `pip install -e ../random_novelty_detector`.

You will now be inside a virtual environment with both sail-on and the random
novelty detector installed.

### Using EVM Based Novelty Detector
1. In the working directory from [quick_start](#Quick_Start) section, clone
   [evm_based_novelty_detector](https://gitlab.kitware.com/darpa-sail-on/evm_based_novelty_detector)
   repository: `git clone https://gitlab.kitware.com/darpa-sail-on/evm_based_novelty_detector.git
2. Ensure that the virtual env from the previous section is active using
    `pipenv shell`.
3. Go to the directory containing `evm_based_novelty_detector`:
    `cd evm_based_novelty_detector`
4. Install the dependencies for `evm_based_novelty_detector`:
    `pip install -e .`
5. Install the custom version of `timm `:
    `pip install -e timm`
6. There are two models required for `evm_based_novelty_detector`:
    1. Efficientnet-B3 is available [here](https://gitlab.kitware.com/darpa-sail-on/evm_based_novelty_detector/-/tree/master/evm_based_novelty_detector).
    2. EVM is a available [here](https://drive.google.com/open?id=1XrSWQWJsF-iPkvGM4AWkMNqvhFTb0yfk).

   Download the models and update models paths in `data/evm_nd.json`

You will now be inside a virtual environment with both sail-on and the TA2
agent installed.

### Running sail-on

The sail-on repository comes with three example configuration files, one for
running with the `DummyNoveltyDetector`, one with `RandomNoveltyDetector` and,
one with `EVMBasedNoveltyDetector`(TA2-agent). `DummyNoveltyDetector` doesn't
actually do anything, simply responding with whatever inputs it is giving.
`RandomNoveltyDetector` mimics a real novelty detector, responding with random
values between 0 and 1 whenever it is asked to make a prediction.

To run these examples, follow these steps:

1. Run the sail-on server from the `sail-on` directory:
   `sail_on_server --data-directory data/ --results-directory results`.
2. In another terminal, run the sail-on client:
   `sail_on_runner --log-file log.txt --config sail_on/data/sail_on_config.json`.
   This command will run the client with the dummy novelty detector.
3. Run the sail-on client again:
   `sail_on_runner --log-file log.txt --config sail_on/data/random_nd.json`.
   This time, the client runs using the random novelty detector.
4. Run the sail-on client one more time, to use the EVM based novelty detector:
   `sail_on_runner --log-file log.txt --config sail_on/data/evm_nd.json --data-root images`
5. Go to the results directory: `cd results/OSND/image_classification`.
6. List the files here by date: `ls -lstr`. The latest three files will be CSVs generated
   during each run, respectively. Note that the the dummy output contains just
   the input image filenames, while the random output also includes random
   values between 0 and 1.

## Local execution with a Stub
We provide `sail_on_runner` with the following parameters

```
-h, --help            show this help message and exit
--log-file LOG_FILE   file to save log (required)
--log-level LOG_LEVEL logging levels (optional)[defaults=log_info]
--data-root DATA_ROOT root directory where images are present
```

To run without any modification to sail_on or novelty_detector_impl and see
workflow through logs use

```python
sail_on_runner --log-file log.txt
```

If `NoveltyDetector` class in the stub has been not changed and the default
configuration is used, the output would along the lines

```
Making session request with [False, True, False], OSND, image_classification
session id: 11, metadata: {'num_tests': 2, 'test_id': [56, 20]}
session id: 11, test_id: 56: Constructing osnd tests
session id: 11, test_id: 56: Creating novelty detector
session id: 11, test id: 56: Inference method is called
session id: 11, test_id: 56: Requesting dataset
session id: 11, test_id: 56: Received a text file <sail_on_root>/sail-on/sail_on/data/image.txt
session id: 11, test id: 56: Inference method is called
session id: 11, test_id: 56: Received a text file <sail_on_root>/sail-on/sail_on/data/image.txt
session id: 11, test_id: 56: Submission [True]
session id: 11, test_id: 20: Constructing OSND tests
session id: 11, test_id: 20: Creating Novelty Detector
session id: 11, test id: 20: Inference method is called
session id: 11, test_id: 20: Requesting Dataset
session id: 11, test_id: 20: Received a text file <sail_on_root>/sail-on/sail_on/data/image.txt
session id: 11, test id: 20: Inference method is called
session id: 11, test_id: 20: Received a text file <sail_on_root>/sail-on/sail_on/data/image.txt
session id: 11, test_id: 20: Submission [True]
```

Currently we are using `sail_on/data/sail_on_config.json` in package root to provide default
configuration. These configurations include `novelty_levels`, `protocol`, `domain`.

To mock the communication we are using `sail_on/mock.py`,
`sail_on/data/image.txt` and `sail_on/data/image_with_labels.txt`. Currently
the files are empty.

## Client

### Command Line

#### Example Test Request

`curl -X POST http://localhost:3306/TestIdsRequest -F 'test_requirements={"protocol":"OSND", "domain":"images","detector_seed":1}' -F'test_assumptions={}'`

#### Running the algorithm

`sail_on_runner --log-file log.txt --config config.json`

The Configuration file is:

```json
   {
      "test_ids_filename" : "test_ids.csv",
      "protocol" : "OSND",
      "novelty_detector_version" : "EVT.0.0.1",
      "detector_config": {
          "foo": "bar"
      }
   }
```
Additional items can be optionally provided for the protocol in 'detector_config'.
All configuration items in detector_config key will be sent to the Protocol as keyed arguments list:
```OSND(session_id, test_id, foo="bar"```

## Data Generation/Evalutation Service

`sail_on_server --url localhost:3306 --data-directory ./tests/data --results-directory ./test/results`

### Provider

Implement Provider to provide the necessary services.
Implementing your own provider requires:
* extension of sail_on.api.Provider
* main routine, using main() and command_line() of sail_on/api/server.py as a basis.

We plan on having a way to register a different provider so that the same command (sail_on_server) can be used.

### FileProvider
This tools provides FileProvider (momentarily named MockProvider). The FileProvider serves all data generation function needs.
It provide a very basic scoring capability.  It is designed to be either
 (1) server as an example on how to provide the data generation or (2) to be basis for extension to its capability.
and evaluation services

The FileProvider assumes tests for protocols have been pre-constructed. This includes:
* List of Tests per Protocol and Domain
* Contents of Tests per Protocol and Domain
* Ground Truth for Tests per Protocol and Domain (i.e. Labels)

The FileProvider is given two directories: a location for the test data and location to store test results.
The file structure for test data is:
+ PROTOCOL LEVEL -> each folder is named after the PROTOCOL (e.g. OSND)
+ + DOMIN LEVEL -> each folder is named after a domain (e.g. images)
+ + + test_ids.csv -> a file summarizing all TEST ID files in the same folder
+ + + TEST DATA FILE: <PROTOCOL>.<TEST>.<NO>.<SEED>.csv files contain the list of images by URI or filename
+ + + TEST LABEL FILE:<PROTOCOL>.<TEST>.<NO>.<SEED>_labels.csv files contain the list of images by URI or filename along the label


## KEY COMPONENTS

* SEED is useful for distinguishing different seeds used for each test set.
* TEST is a name or number used to group files all designed to for one test to attain statistical signifiance.
* NO is the incremental test set number.
* URI or filename assumes that information is reachable by the client (system under test)
* Label CSV files have two mandatory columns: the image file URI or filename (as matched to the test data file).  All other columns are reserved for scoring.
These never shared with the client via the protocol.


## CAUTION

* At no time should the same test file contents change.  It is better to create a new file with a new name.
We want to allow the client system to pre-cache and pre-processing the test data given the URI or filename  (e.g. place in a pyTorch dataset).

## Scoring

The File Provider provides precision, accuracy, recall and F1 Score as a default.

## Building the Documentation
TBD

## Running tests
TBD
