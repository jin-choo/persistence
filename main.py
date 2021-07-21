import argparse
from algorithms import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filepath",
        action="store",
        default="email-Enron",
        type=str,
        help="Select the file path",
    )
    parser.add_argument(
        "-d",
        "--do",
        action="store",
        default="count_time_units",
        type=str,
        help="Select what to do",
    )
    parser.add_argument(
        "-m",
        "--m",
        action="store",
        default="43",
        type=int,
        help="Select the number of time units",
    )
    parser.add_argument(
        "-k",
        "--k-tuples",
        action="store",
        default="4",
        type=int,
        help="Select the size of hois",
    )
    parser.add_argument(
        "-b",
        "--basket-max",
        action="store",
        default="25",
        type=int,
        help="Select the maximum size of a hyperedge",
    )
    parser.add_argument(
        "-i",
        "--interval",
        action="store",
        default="10",
        type=int,
        help="Select the number of observed time units",
    )
    parser.add_argument(
        "-t",
        "--time-units",
        action="store",
        default="5",
        type=int,
        help="Select the of observed time units for measuring features",
    )
    parser.add_argument(
        "-u",
        "--unit",
        action="store",
        default="1",
        type=int,
        help="Select the unit",
    )
    parser.add_argument(
        "-r",
        "--read",
        action="store",
        default="local_group_group",
        type=str,
        help="Select what to read",
    )

    args = parser.parse_args()
    if args.do == 'count_time_units':
        count_time_units(args.filepath, args.unit)
    elif args.do == 'graph':
        graph(args.filepath)
    elif args.do == 'graph_bi':
        graph_bi(args.filepath)
    elif args.do == 'global_analysis':
        global_analysis(args.filepath, args.m, args.k_tuples, args.basket_max, args.interval)
    elif args.do in ['local_analysis_past', 'local_analysis']:
        local_analysis_save(args.filepath, args.m, args.k_tuples, args.basket_max, args.time_units, args.interval, args.do)
    elif args.do in ['local_group_group_past', 'local_group_group']:
        local_group_group(args.filepath, args.m, args.k_tuples, args.basket_max, args.time_units, args.interval, args.do)
    elif args.do in ['local_node_group_past', 'local_node_group']:
        local_node_group(args.filepath, args.m, args.k_tuples, args.basket_max, args.time_units, args.interval, args.do)
    elif args.do in ['local_node_node_past', 'local_node_node']:
        local_node_node(args.filepath, args.m, args.k_tuples, args.basket_max, args.time_units, args.interval, args.do)
    elif args.do == 'global_stat':
        global_stat(args.basket_max)
    elif args.do == 'local_stat':
        local_stat(args.basket_max, args.time_units, args.interval, args.do, args.read)
    elif args.do in ['pred_1_past', 'pred_1']:
        pred_1(args.filepath, args.m, args.k_tuples, args.basket_max, args.time_units, args.interval, args.do)
    elif args.do in ['pred_2_past', 'pred_2']:
        pred_2(args.filepath, args.m, args.k_tuples, args.basket_max, args.time_units, args.interval, args.do)
    elif args.do in ['pred_feature_selection_past', 'pred_feature_selection']:
        pred_feature_selection(args.k_tuples, args.basket_max, args.time_units, args.interval, args.do)
    elif args.do == 'predictability':
        predictability(args.basket_max, args.interval)