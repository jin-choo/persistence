# On the Persistence of Higher-Order Interactions in Real-World Hypergraphs

This repository contains the source code for the paper [On the Persistence of Higher-Order Interactions in Real-World Hypergraphs](https://).

In this work, we empirically investigate the persistence of higher-order interactions (HOIs) in 13 real-world hypergraphs from 6 domains.
We define the measure of the persistence of HOIs, and using the measure, we closely examine the persistence at 3 different levels (hypergraphs, groups, and nodes), with a focus on patterns, predictability, and predictors.
* **Patterns**: We reveal power-laws in the persistence and examine how they vary depending on the size of HOIs. Then, we explore relations between the persistence and 16 group- or node-level structural features, and we find some (e.g., entropy in the sizes of hyperedges including them) closely related to the persistence.
* **Predictibility**: Based on the 16 structural features, we assess the predictability of the future persistence of HOIs. Additionally, we examine how the predictability varies depending on the sizes of HOIs and how long we observe HOIs.
* **Predictors**: We find strong group- and node-level predictors of the persistence of HOIs, through Gini importance-based feature selection. The strongest predictors are (a) the number of hyperedges containing the HOI and (b) the average (weighted) degree of the neighbors of each node in the HOIs.

## Supplementary Document

Please see [supplementary](./supplementary.pdf).

## Datasets

All datasets are available at this [link](https://www.cs.cornell.edu/~arb/data/).

| Domain       | Dataset    |   # Nodes  | # Hyperedges | Time Range | Time Unit |
|--------------|------------|:----------:|:------------:|:----------:|:---------:|
| Coatuhorship | DBLP       | 1,924,991  |  3,700,067   |     83     |   1 Year  |
|              | Geology    | 1,256,385  |  1,590,335   |    219     |   1 Year  |
|              | History    | 1,014,734  |  1,812,511   |    219     |   1 Year  |
| Contact      | High       |    327     |   172,035    |     84     |   1 Day   |
|              | Primary    |    242     |   106,879    |    649     |  6 Hours  |
| Email        | Enron      |    143     |    10,883    |     43     |  1 Month  |
|              | Eu         |    998     |   234,760    |     75     |  2 Weeks  |
| NDC          | Classes    |   1,161    |    49,724    |    118     |  2 Years  |
|              | Substances |   5,311    |   112,405    |    118     |  2 Years  |
| Tags         | math.sx    |   1,629    |   822,059    |     89     |  1 Month  |
|              | ubuntu     |   3,029    |   271,233    |    104     |  1 Month  |
| Threads      | math.sx    |  176,445   |   719,792    |     85     |  1 Month  |
|              | ubuntu     |  125,602   |   192,947    |     92     |  1 Month  |

## Requirements

To install requirements, run the following command on your terminal:
```setup
pip install -r requirements.txt
```

## Measurement of the persistence of HOIs

To measure the persistence of HOIs, run these commands:

```
./compile.sh
./run.sh $dataset [interval|interval_front] $number_of_time_units $size_of_hois $maximum_size_of_he [0:others|1:coauthorship] $number_of_observed_time_units $number_of_observed_time_units_for_features
```

We provide the running example codes for measureing the persistence of HOIs, by running this command.

```
./run_batch.sh
```


## Preprocessing

To preprocess before analyzing the persistence of HOIs, run these commands:

```
./count_time_units.sh
./graph.sh
```

## Global Analysis: Persistence vs. Frequency

To obtain the global patterns in the persistence of HOIs, run these commands:

```
python3 main.py -f $dataset -d global_analysis -m $number_of_time_units -i $number_of_observed_time_units
python3 main.py -d global_stat
```

We provide the running example codes for global analysis, by running this command.

```
./global_analysis.sh
```

## Local Analysis: Features vs. Persistence

To assess the local patterns in the persistence of HOIs, run these commands:

```
python3 main.py -f $dataset -d [local_analysis|local_group_group|local_node_group|local_node_node] -m $number_of_time_units -i $number_of_observed_time_units -t $number_of_observed_time_units_for_features
python3 main.py -d local_stat
```

We provide the running example codes for local analysis, by running this command.

```
./local_analysis.sh
```

## Predictability and Predictors

To examine the predictability of the persistence of HOIs, run these commands:

```
python3 main.py -f $dataset -d [pred_1|pred_1_past|pred_2|pred_2_past] -m $number_of_time_units -i $number_of_observed_time_units -t $number_of_observed_time_units_for_features
python3 main.py -d pred_feature_selection -i $number_of_observed_time_units -t $number_of_observed_time_units_for_features
python3 main.py -d predictability -i $number_of_observed_time_units
```

We provide the running example codes for local analysis, by running this command.

```
./pred.sh
```

## Reference

This code is free and open source for only academic/research purposes (non-commercial). If you use this code as part of any published research, please acknowledge the following paper.
```

```
