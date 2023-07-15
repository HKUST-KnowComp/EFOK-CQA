import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=150)
parser.add_argument("--end", type=int, default=250)
parser.add_argument("--dataset", type=str, default='FB15k')
parser.add_argument("--each", type=int, default=1)
parser.add_argument("--num_positive", type=int, default='800')
parser.add_argument("--num_negative", type=int, default='400')
parser.add_argument("--s_each", type=int, default=10)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    dataset = args.dataset
    datafolder = 'data/' + dataset + '-EFOX-filtered'
    output_folder = 'data/' + dataset + '-EFOX-filtered'
    each_num = args.each
    for start in range(args.start, args.end, each_num):
        command = ("nohup python sample_query.py "
                   f"--data_folder {datafolder} "
                   f"--num_positive {args.num_positive} "
                   f"--num_negative {args.num_negative} "
                   f"--store_each {args.s_each} "
                   f"--output_folder {output_folder} "
                   f"--start_index {start} "
                   f"--end_index {start + each_num - 1} > sample_EFOX_{dataset}_{start}_{start + each_num - 1}.log 2>&1 &")
        print(command, '\n')
        os.system(command)
        print("launched")

