## Protocol

The following section summarizes the set of experimental protocols.

**Open-set Novelty Detection, Discovery, Adaptation (OND+A)** is a blend of classification and novelty characterization. A TA2 agent starts with K+1-way classification and novelty detection (K known, D unknown). The TA1 harness will be able to handle the prediction space of K+1 classification -- K known, one group of all unknown activities.

Each trial is broken into two stages:

- Detection and Classification
- Characterization

In the Detection and Classification stage, the TA2 agent processes small batches of video clips over the course of multiple rounds. The TA2 determines the probability that an instance in a round represents the point of distribution shift into the novelty phase of the trial. TA2 also predicts the K+1 activity class in a round.**_The ‘+1’ label represents the group of all novel activity classes encountered during a trial._**

Once TA2 has indicated that the trial has entered into a novelty phase (via the detection result), TA2 can then request those forms of feedback as described earlier. This provides an opportunity for TA2 to increase the accuracy of the classifications in subsequent rounds within the same trial.

In the Characterization stage, the TA2 assigns activity labels to each of video instances using K+D labels where K represents known classes and D represents unknown classes. The maximum number of unknown classes per trial is 4.


## Trial Process

TA2 agent is evaluated over a set of trials, each trial with its own identifier. The list of trial identifiers is maintained in an ASCII text file.. The list is provided via a communication channel between TA1 and TA2 or by a REST call to a trial server.

The Trial process is managed by a client/server architecture. In this architecture a client interacts with a trial server on the TA2 agent’s behalf. The TA2 agent is invoked by and interacts with the client. The trial server is separate, allowing it to manage multiple ongoing trials simultaneously. Further, the client, with its TA2 component, can be run in resource specific hardware (e.g. GPUs) in a distributed fashion.

The client initializes a session with the server to execute one or more of the trials selected from the set of trial IDs. Not all trial IDs need to be executed in one session. For each trial in the session, the client initializes a TA2 agent. Trials can run in parallel. A single trial is a sequential process.


## Trial Identification and Construction

The number and make-up of trials created for each domain and protocol serve to determine statistical significance during evaluation of the trial results. A trial ID is associated with a specific set of data.

A series of trials is called a group. Many trials are grouped together in a concept called a trial group. TA1 uses trial groups to organize trials designed for statistical significance over specific dimensions of the novelty space.

Trials in the same group are composed of the same set of videos with different order. TA2 should ignore this fact and process each trial separately, with no information carried over from one trial to the next.

  


The trial ID is self describing, made up of the following parts separated by ‘.’.

- Protocol ID
- Group ID
- Run ID
- Seed

Example ID: OND.1.2.10631

The seed is selected at random to prevent leaking information. TA1 will maintain a list of all seeds and associated setup to be shared with other TA1/DARPA as needed.

Each trial will include aguaranteed Pre-Novelty set of batches. Trial meta-data indicates the size (number of batches) of the pre-novelty region. The specific batch and video instance in that batch representing the transition point to the novelty phase of trial is not disclosed.

  


The following meta-data is provided in JSON format to describe each trial:

- The number of activity classes (known classes) in the data set,

- The maximum possible novel activity classes (max novel classes)

  - This indicates the number of D columns in the classification file.

- The batch size (round size)

- The novelty detection point for given(non-system) detection trials (red light).

{

"protocol": "OND",

"known_classes":101,

"max_novel_classes":5,

"round_size":32,

"pre-novelty-batches": 10,

"red_light": "trial/trial_video_l3/014725.mp4"

}


### Detection Threshold

The TA1 system uses the threshold to determine when TA2 has signaled an entry into the novelty phase of the trial when inspecting the detection results on each and every round. The TA2 supplies the threshold to TA1 at session creation. This allows TA1 to run TA2 clients with different thresholds. The default threshold is 0.5. The section on detection results provides more details.


## Client/Server API

  


Trial results include novelty detections, video activity classifications and novelty characterizations.

With the exception of trial meta-data, all data is exchanged via a CSV format.

All videos presented in a trial are identified by TA1-provided UUID, serving to anonymize video.

Table 1. provides detail about different requests used for communication between client and server.

| **Request Type**  | **Definition**                                                                                            | **Request Data**                                                                                                                                                                                                               | **Response Data**                                                                                                                                                                                                                                                                                                   |
| ----------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Trial Request     | TA 2 Requests for trial Identifiers as part of a series of individual trials.                             | 1\. Protocol: Empirical protocol that the client would be evaluated against.2\. Domain: Problem domain addressed by the novelty detector3\. Detector Seed                                                                      | CSV FILE containing: trial IDs.CSV File Naming Convention has four parts:1. Protocol 2. Group 3. Run 4. Seed                                                                                                                                                                                                        |
| New Session       | Create a new session to evaluate the detector using an empirical protocol with a provided set of trials   | 1\. Trial IDs from the provided set (CSV File).2\. Protocol3\. Domain4\. Novelty Detector Version5\. Detection ThresholdThe set of trial IDs may be a partial set of all IDs received for a given protocol, domain, etc.       | Session ID: A unique identifier that the server associated with the client                                                                                                                                                                                                                                          |
| Get Meta Data     | Request Trial Metadata                                                                                    | 1\. Session Id.2\. Trial Id                                                                                                                                                                                                    | JSON                                                                                                                                                                                                                                                                                                                |
| Dataset Request   | Request data for evaluation.                                                                              | 1\. Session ID2\. Trial ID3\. Round Number                                                                                                                                                                                     | CSV of anonymized instance identifiers.The TA2 agent is assumed to have a mapping from the identifier to the physical location of the trial data.ERRORS:If all rounds are completed, HTTP Code 204 is returned.Other round errors (submitting for an older round or a skipping a round), HTTP Code 404 is returned. |
| Post Results      | Post client detector predictions for the dataset.                                                         | 1\. Session ID2\. Trial ID3\. Round ID4\. Result Files: (CSV)1. Classification 2. Characterization file types 3. Detection                                                                                                     | Result acknowledgement.                                                                                                                                                                                                                                                                                             |
| Get Feedback      | Get Feedback from the server based provided one or more video instance ids from the last completed round  | 1\. Session ID2\. Trial ID3\. Round Number4\. Feedback Types                                                                                                                                                                   | CSV file (see feedback section)ERRORS:If feedback is not permitted during this test, the content of the files are empty.An error message is not given.Thus, it is not harmful for a client to request feedback in pre-novelty. The only penalty is round-trip execution time.                                       |
| Terminate trial   | Signal completion of the trial by TA2. States when TA2 is completed sending all characterization results. | 1\. Session ID2\. Trial ID                                                                                                                                                                                                     | Acknowledgement of trial termination                                                                                                                                                                                                                                                                                |
| Terminate Session | Terminate the session after the evaluation for the protocol is complete                                   | 1\. Session ID2\. Logs for the session                                                                                                                                                                                         | Acknowledgement of session termination                                                                                                                                                                                                                                                                              |

Table 1. Requests and Responses for client-server communication

  


  


As shown in Figure 1,the client interacts with the server on the TA2 agent's behalf. The TA2 agent is initialized at the start of a trial with configuration data provided by the client. The agent informs the client its name and version for submission to the server. The client then commences with processing rounds of a trial. For each round, the TA2 is presented with list UUIDs for the anonymized video instances. The TA2 agent first detects if novelty has occurred. Next, the TA2 agent is provided the same list of video instances to which it answers classifications for each instance-- probability vectors across the activity label space (e.g. softmax). Client sends the detections and classifications to the server. The client then prompts the TA2 agent to request feedback. The TA2 answers a list of 0 to M video instance UUIDs for which it selects for feedback (M is the maximum number of instance IDs allowed per trial). The client requests for feedback from the server on the TA2 agent's behalf and sends the server’s answer to the TA2 agent.The TA2 then updates its internal state using the feedback and the next round begins.

At the end of all rounds, the client queries the TA2 agent for characterization of ALL the video instances seen in the trial. The TA2 answers those inquiries it is able and configured to answer.


## Pseudocode

At trial time, TA1 novelty generator does the following.

Video Instances produced at timet1...tn. Novelty is injected at some ratea started at time tinjected.

No novel video instances in \[t1,tinjected)

a% novel video in \[tinjected,tn ].

At trial time: sequence of video instances organized into batches (batchsize = b):

B0 : (x1,x2, ...xb), B1 : (xb+1, ...,x2b), … Bceil(N/b) : (xn-b, ...,xn )

The batches are submitted in order through k rounds where k = ceil(N/b)

For each batch Bk: // rounds

For each video in Bk, predict outputs of OND.

predict novel/not novel probability

predict K+1-probabilities

Submit predictions for Bk

Request feedback from trial server

Optionally update model

Submit characterization rules.

At trial time, what TA1 evaluator does:

- Collects predictions for all instancesxn .

Take the first instance with prediction = novel (or pnovel≥ established session threshold (ie. 0.5) ) and consider that time-step as the point where TA2 predicts that novelty has been introduced ( tdetected ) and compute the required program metrics and additional TA1/TA2 agreed upon metrics.  


# Trial Data Examples

The next sections demonstrate the data exchanged between client and server.


## Trial IDs

Trial IDs are provided by TA1 via a set of one or more text files.

Example:

OND.1.0001.1720

OND.2.0404.1369

OND.3.0003.1614

Upon execution, the client chooses a subset of the trials to run in each session on behalf of the TA2 agent.


## Trial Request

Each trial ID and each round, the client requests activity instances id. Trial Requests are responded with a text file of file UUIDs. Prior to running a set of trials, TA1 provides TA2 a set of anonymized video files. This reduces the exchange of large files. The same video file may be reused in several trials. See the Architecture Section for more details.

Example:

b3b28ca7-d55e-4ade-ab94-ad59af03e60f.avi

cff908d5-728b-4aa6-912a-7a2e4690d1d1.avi

203293e9-4132-4a64-81f8-8f8586b701ad.avi

5fbfcf64-a414-4cd6-a1ab-9ecd86d66807.avi

8a6805b2-8ce3-41f3-8eca-7a567c87e3a1.avi

e8a65311-142d-4ebb-badc-1640c9c3ffe5.avi

2eaeed87-d169-4df0-8892-98ef84c6a01f.avi

aa21b0cf-a325-49de-8754-b7ad88e69d1d.avi





## Detection, Classification and Characterization Posted Results

For each round, the TA2 agent responds with detection and classification results. This section describes the format of the TA2 posted results.

Column 0 of all CSV files is the Image ID (URL) as provided in the trial data set file(s).

**For detection,** the subsequent column 1 is the novel phase prediction N(_i_) at given instance_i_ over all instances seen in the trial.

Example:

e8a65311-142d-4ebb-badc-1640c9c3ffe5.avi,0

2eaeed87-d169-4df0-8892-98ef84c6a01f.avi,0.1

aa21b0cf-a325-49de-8754-b7ad88e69d1d.avi,0.11

TA2 should submit results as weighted for each video instance as probabilities of encountering the change detection point (CDP). The detection file will have an initial probability near 0. As the test progresses, TA2 is looking for distribution changes. It is expected in the ‘guaranteed pre-novelty’ of the test, as defined in the meta-data, the detection stays near 0. Since distribution changes are not detected instantaneously, the detection curve should not move above threshold until some reasonable evidence or confidence of novel-world detection has been achieved.

0,0,0.....,0,0.1,0.1,0.25,0.25,..,0.4,...,0.4,0.6,0.6,...0.8,0.8,....,0.95,0.95,...,1,1,1,1

TA2 need not use the curve concept. Another approach would be to have TA2 manage thresholds themselves, sending 0 and 1. TA2 can instead use a single class model approach of presence or absence (1 and 0). When TA2 believes the trial is in the pre-novelty phase, all entries are 0. When in the novelty phase, all entries in the file are 1. The change from 0 to 1 is the change detection point (CDP). This approach does not support TA1 doing some thresholding analysis.

NOTE: If TA2 would like to record per-instance novelty predictions (pnovel(xi)) for a separate confusion matrix as a method to assess the accuracy of instance novelty detections, TA2 can submit a second numeric column with the per instance detection probability as shown in this example:

8a65311-142d-4ebb-badc-1640c9c3ffe5.avi,0,0

2eaeed87-d169-4df0-8892-98ef84c6a01f.avi,0.1,0.1

aa21b0cf-a325-49de-8754-b7ad88e69d1d.avi,0.11,0.13

.

.

.

a920e7ac-82bd-4409-bf2a-f39451a8ab24.avi,0.8,0.84

2cec0cd1-9e2d-4c22-8d23-331b5713fec4.avi,0.81,0.3

049e069a-2710-4a11-b0f1-02e8e3c52567.avi,0.82,0.88

8343b9fb-c382-4e66-b2e6-b5ed7f206f82.avi,0.81,0.80

69eae693-d68b-4552-add7-80aa7ac686b9.avi,0.75,0.9

  


**For classification,** the subsequent K+1 columns provide a prediction of the K activity class

labels and the represented single unknown activity class labels:\[p1,p2, ...,pk,punknown]

Example (with K=5):

e8a65311-142d-4ebb-badc-1640c9c3ffe5.avi,0.1,0.2,0.70,0.00,0.0,0.0

2eaeed87-d169-4df0-8892-98ef84c6a01f.avi,0.1,0.1,0.00,0.00,0.8,0.0

aa21b0cf-a325-49de-8754-b7ad88e69d1d.avi,0.8,0.1,0.05,0.05,0.0,0.0

For video activity recognition based on the UCF 101 and Ontology, K=88 . The assigned activity numeric identifiers from the Ontology are numbered 0 through 87. These identifiers map to the activity columns of the activity classification file. The first column of the activity classification file is the video identifier, the second column maps to activity 0, the third column maps to activity 1, and so on. Thus, classification file columns 2 through 89 are associated with known activities. Column 90 of the classification file corresponds to a novel activity.

Real Example:

8fcc5b2-bb20-482b-bf65-25f2edb59606.avi,0.4017,0.0025,0.0077,0.0155,0.0110,0.0146,0.0040,0.0028,0.0055,0.0048,0.0015,0.0070,0.0007,0.0039,0.0064,0.0018,0.0082,0.0017,0.0065,0.0073,0.0017,0.0090,0.0041,0.0064,0.0063,0.0080,0.0090,0.0043,0.0037,0.0211,0.0047,0.0072,0.0127,0.0068,0.0063,0.0050,0.0030,0.0003,0.0026,0.0062,0.0043,0.0009,0.0028,0.0039,0.0069,0.0139,0.0056,0.0064,0.0025,0.0028,0.0004,0.0008,0.0110,0.0056,0.0037,0.0213,0.0031,0.0108,0.0127,0.0093,0.0081,0.0045,0.0050,0.0156,0.0024,0.0004,0.0036,0.0265,0.0053,0.0104,0.0049,0.0395,0.0004,0.0033,0.0025,0.0058,0.0095,0.0023,0.0020,0.0123,0.0123,0.0003,0.0230,0.0049,0.0073,0.0031,0.0047,0.0078,0.0000

**For Characterization**, we wish to cluster by novelty type. We will define three sets of clusters for classification, submitted via a separate CSV file. The CSV file contains N rows for each of N video instances in a trial. Each row predicts the probability of the instance occuring in each cluster. Each column in the CSV file is associated with a cluster. Thus each cell represents the probability that the cell’s instance belongs in the cell’s cluster.

Classification characterization is a separate submission by TA2 from the K+1 classification file. In this submission file,there are D+1 activity columns. D bounds the maximum number of novel activity types (i.e. 2 in M18). Each column represents a single cluster of video instances of the same or similar activity type. The ‘+1’ cluster represents the known classes.

The purpose of characterization is to organize the videos in clusters similar to each other in terms of activity type.

Each column should reflect the probability that a video instance belongs in the column’s associated cluster.

![](https://lh6.googleusercontent.com/GniPux3uVciJjZppUfPP1uoNM89k_dIdGxzI9pCtcYUcovCCmKEheLdZCSYLYitSMCL-mZLoK0VbNzEUhDibqj1qHYzgtiU35eX45mqUh0jU2I-dFDJVUC7oHWwZo8ECIAa_6BO-0DB5NSY7w68l)

Example Classification Characterization:

Suppose a round in a trial has six examples: A, B, C, D, E, and F. Suppose the trial has only two activity types presented--one known and one unknown. Known swimming instances are B, E and F. Instances A, C and D are sampled from the unknown class batting/baseball. The classification characterization file will have the following entries, as probabilities of occurring in each cluster (assuming TA2 elects that column 1 is batting and column 2 is swimming):

A,0.90,0.10,0

B,0.25,0.75,0

C,0.33,0.67,0

D,0.99,0.01,0

E,0.10,0.90,0

F,0.15,0.85,0

In this example, assume D = 2.


## Feedback


### Instance Feedback

This feedback is availablewithout penalty. The feedback is given on a budget of N% the batch size. For example, if N = 10 and the batch size for video activity recognition is 32, each batch is permitted to request 4 instance labels.

_Per-Instance Feedback can be only applied to batch instances associated with the current round, not prior rounds._

  


For instance based feedback, TA2 requests feedback for video instances (IDs). For each TA2 provided ID, the TA1 server responds with five closets KINETICS 600 classes:

  


e8a65311-142d-4ebb-badc-1640c9c3ffe5.avi,,HighJump,LongJump,PoleVault,


### Detection Feedback

This feedback is availablewithout penalty. The feedback is given on a budget of N% the batch size. For example, if N = 10 and the batch size for video activity recognition is 32, each batch is permitted to request 4 instance labels.

_Per-Instance Feedback can be only applied to batch instances associated with the current round, not prior rounds._

For instance based feedback, TA2 requests feedback for video instances (IDs). For each TA2 provided ID, the TA1 server responds with a novel indicator: 0 = not novel, 1 = novel

e8a65311-142d-4ebb-badc-1640c9c3ffe5.avi,1

dca65x11-13dd-8exb-befc-2030c9c3f301.avi,0


### Classification Accuracy Feedback

This feedback is available without penalty after the agent has indicated the novelty phase has been entered (via the detection file).

For accuracy feedback, TA2 will also be provided a single accuracy of the trial as of the current round. The reason for providing the accuracy for the entire trial is to include some basis in terms of pre-novelty accuracy to judge future performance. For example, if the current round accuracy is 0.5 and the trial accuracy is 0.9, without combining the two values, the TA2 agent cannot assess if 0.5 is due to novelty or due to poor performance across the entire trial. Armed with information, the TA2 can see performance trends in each subsequent post-novelty round. Recall, TA2 agent knows the round size and the number of rounds processed at each feedback request point.

Labelsliwhere_i_ is an instance number as seen since the beginning of the trial. In each round _r_,Nr instances are seen by the agent. After _R_ rounds,  0i &lt;RNr . The ground truth activity label for an example video_i_ is g_i_. An error_e_is defined as:

ei =mini (li,gi)

(x, y)= 0 if x==y, else 1.

Accuracy is defined as1/(RNr)iNr(1-ei) with Nr= the number of examples per round, R=number of rounds completed at the time of the feedback request.

The TA1 client sends the CSV accuracy result to the TA2 agent as follows:

accuracy,0.78
