# How Do Higher-Order Interactions Persist in Real-World Hypergraphs?

This repository contains the source code for the paper [How Do Higher-Order Interactions Persist in Real-World Hypergraphs?](https://) by [Hyunjin Choo](https://github.com/jin-choo/) and [Kijung Shin](https://kijungs.github.io/), to be presented at [](https://).

In this work, we empirically investigate the persistence of higher-order interactions (HOIs) in 13 real-world hypergraphs from 6 domains.
We define the measure of the persistence of HOIs, and using the measure, we closely examine the persistence at 3 different levels (hypergraphs, groups, and nodes), with a focus on patterns, predictability, and predictors.
* Patterns: We reveal power-laws in the persistence and examine how they vary depending on the size of HOIs. Then, we explore relations between the persistence and 16 group- or node-level structural features, and we find some (e.g., entropy in the sizes of hyperedges including them) closely related to the persistence.
* Predictibility: Based on the 16 structural features, we assess the predictability of the future persistence of HOIs. Additionally, we examine how the predictability varies depending on the sizes of HOIs and how long we observe HOIs.
* Predictors: We find strong group- and node-level predictors of the persistence of HOIs, through Gini importance-based feature selection. The strongest predictors are (a) the number of hyperedges containing the HOI and (b) the average (weighted) degree of the neighbors of each node in the HOIs.

## Supplementary Document

Please see [supplementary](./doc/supplementary.pdf).

## Datasets
All parsed datasets are available at this [link](https://www.dropbox.com/sh/lha0oevqos6jxn9/AAAz3Xkql2aKwcnKmX3kt357a?dl=0).
The source of each dataset is listed below.
| Name          | Structure                                       | Size                   | # Non-zeros | Source   |
| ------------- |:-----------------------------------------------:| :---------------------:| :----------:| :-------:|
| Divvy Bikes   | sources x destinations x time (minutes)         | 673 x 673 x 525594     | 3.82M       | [Link](https://www.divvybikes.com/system-data) |
| Chicago Crime | communities x crime types x time (hours)        | 77 x 32 x 148464       | 5.33M       | [Link](http://frostt.io/) |
| New York Taxi | sources x destinations x time (seconds)         | 265 x 265 x 5184000    | 84.39M      | [Link](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |
| Ride Austin   | sources x destinations x colors x time (minutes)| 219 x 219 x 24 x 285136| 0.89M       | [Link](https://data.world/andytryba/rideaustin) |

## Requirements

* C/C++17 compiler
* CMake
* Git

## Used Libraries

* Eigen (<https://gitlab.com/libeigen/eigen>)
* yaml-cpp (<https://github.com/jbeder/yaml-cpp>)

## Download

```bash
git clone --recursive https://github.com/DMLab-Tensor/SliceNStitch.git
```

## Generate Build System

To generate the build system, run the following command on your terminal:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
```

After that, you can build the program using the build automation software (e.g. Make, Ninja).

## Execution

To test the algorithms in the paper, run the following command on your terminal:

```bash
./SliceNStitch [config_file]
```

## Config File

### Example config file

```yaml
# test.yaml
data:
    filePath: "nyt_2019.csv"  # Path of the data stream file
    nonTempDim: [265, 265]  # Non-temporal dimension
    tempDim: 5184000  # Temporal length of data stream

tensor:
    unitNum: 10  # The number of indices in the time mode (W)
    unitSize: 3600  # Period (T)
    rank: 20  # Rank of CPD (R)

algorithm:
    name: "SNS_RND+"
    settings:  # Details are described in the next section
        numSample: 20
        clipping: 1000.0

test:
    outputPath: "out.txt"  # Path of the output file
    startTime: 0  # Starting time of the input data to be processed
    numEpoch: 180000  # The number of epochs
    checkPeriod: 1  # Period of printing the error
    updatePeriod: 1  # Period of updating the tensor
    random: true  # Randomness of the initial points
```

### Examples of Possible Algorithms

```yaml
algorithm:
    name: "ALS"
    settings:
```

```yaml
algorithm:
    name: "SNS_MAT"
    settings:
        numIter: 1
```

```yaml
algorithm:
    name: "SNS_VEC"
    settings:
```

```yaml
algorithm:
    name: "SNS_VEC+"
    settings:
        clipping: 1000.0  # Clipping value (η)
```

```yaml
algorithm:
    name: "SNS_RND"
    settings:
        numSample: 20  # Threshold (θ)
```

```yaml
algorithm:
    name: "SNS_RND+"
    settings:
        numSample: 20  # Threshold (θ)
        clipping: 1000.0  # Clipping value (η)
```

## Input & Output Format

Input (data:filePath in a config file) must be a CSV file that consists of a multi-aspect data stream.
Each row of the file is a single event and the file should be formatted as follows.

For a CSV file with N columns,

* First (N-2) columns represent the non-temporal indices of events
* The (N-1)th column represents the time indices of events
* The last column represents the values of events

The output (test:outputPath) of the code will be:
```
-----Test Results-----
RMSE	Fitness
0.33438	0.675436
0.329659 0.639608
...
0.37203	0.692263

-----The Total Number of Updates-----
4030721

-----Total Elapse Time-----
330.706 (sec)

-----Elapse Time per each Update-----
8.20463e-05 (sec)
```

## Reference

If you use this code as part of any research, please cite the following paper.

```
@inproceedings{kwon2021slicenstitch,
  title={Slicenstitch: Continuous cp decomposition of sparse tensor streams},
  author={Kwon, Taehyung and Park, Inkyu and Lee, Dongjin and Shin, Kijung},
  booktitle={2021 IEEE 37th International Conference on Data Engineering (ICDE)},
  pages={816--827},
  year={2021},
  organization={IEEE}
}
```
{"mode":"full","isActive":false}
