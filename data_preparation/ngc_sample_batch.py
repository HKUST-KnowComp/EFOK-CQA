import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=5)
parser.add_argument("--dataset", type=str, default='NELL')
parser.add_argument("--each", type=int, default=5)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    dataset = args.dataset
    datafolder = 'data/' + dataset + '-EFOX'
    output_folder = 'data/' + dataset + '-EFOX'
    each_num = args.each
    for start in range(args.start, args.end, each_num):
        command = (
                   "ngc batch run "
                   f"--name ml-model.Truth_value_{dataset}_{start} "
                   "--priority NORMAL "
                   "--preempt RUNONCE "
                   "--ace nv-us-west-2 "
                   "--instance cpu.x86.tiny "
                   f'--commandline "cd /mount/kgtvr/Truth-Value-Reasoning-on-Knowledge-Graphs && pip install python-constraint && python3 sample_query.py --data_folder {datafolder} --output_folder {output_folder} --start_index {start} --end_index {start + each_num - 1}" '
                   "--result /results "
                   "--image nvidia/pytorch:22.04-py3 "
                   "--org nvidian "
                   "--team sae "
                   "--workspace 8sr-vg7_STK0qaux2KusFA:/mount/kgtvr:RW "
                   "--order 50")
        print(command, '\n')
        os.system(command)
        print("launched")
