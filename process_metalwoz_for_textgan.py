import json
import random
import os
from argparse import ArgumentParser

random.seed(271)

TRAINSET_RATIO = 0.9


def flatten(in_data):
    turns = []
    for dialog in in_data:
        for turn in dialog['turns']:
            turns.append(turn['text'].lower())
    return turns


def split_traintest(in_data):
    train_datapoints = int(len(in_data) * TRAINSET_RATIO)
    random.shuffle(in_data)
    return in_data[:train_datapoints], in_data[train_datapoints:]


def write_textgan_dataset(in_dataset_name, in_train, in_test):
    result_folder = 'dataset'

    if not(os.path.exists(result_folder)):
        os.makedirs(result_folder)
    target_folder = os.path.join(result_folder, in_dataset_name)
    if not(os.path.exists(target_folder)):
        os.makedirs(target_folder)

    with open(os.path.join(target_folder, 'train.txt'), 'w') as train_out:
        train_out.write('\n'.join(in_train))
    with open(os.path.join(target_folder, 'test.txt'), 'w') as test_out:
        test_out.write('\n'.join(in_test))


def main(in_src_file, in_dataset_name):
    with open(in_src_file) as data_in:
        data = json.load(data_in)
    data_flat = flatten(data)
    train, test = split_traintest(data_flat)
    write_textgan_dataset(in_dataset_name, train, test)


def init_argument_parser():
    parser = ArgumentParser()
    parser.add_argument('source_file')
    parser.add_argument('--dataset_name', default='metalwoz')
    return parser

if __name__ == '__main__':
    parser = init_argument_parser()
    args = parser.parse_args()
    main(args.source_file, args.dataset_name)
