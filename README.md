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



## Reference

This code is free and open source for only academic/research purposes (non-commercial). If you use this code as part of any published research, please acknowledge the following paper.
```

```
