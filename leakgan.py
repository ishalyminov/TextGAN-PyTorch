from __future__ import print_function

import argparse

import config as cfg
from instructor.real_data.leakgan_instructor import LeakGANInstructor


def program_config(parser):
    # Program
    parser.add_argument('--if_test', default=False, type=int)
    parser.add_argument('--run_model', default='leakgan', type=str)
    parser.add_argument('--dataset', default='metalwoz', type=str)
    parser.add_argument('--model_type', default='vanilla', type=str)
    parser.add_argument('--loss_type', default='JS', type=str)
    parser.add_argument('--if_real_data', default=True, type=int)
    parser.add_argument('--cuda', default=True, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--shuffle', default=False, type=int)
    parser.add_argument('--use_truncated_normal', default=True, type=int)

    # Basic Train
    parser.add_argument('--samples_num', default=10000, type=int)
    parser.add_argument('--vocab_size', default=10000, type=int)
    parser.add_argument('--mle_epoch', default=8, type=int)
    parser.add_argument('--adv_epoch', default=200, type=int)
    parser.add_argument('--inter_epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_seq_len', default=20, type=int)
    parser.add_argument('--start_letter', default=1, type=int)
    parser.add_argument('--padding_idx', default=0, type=int)
    parser.add_argument('--gen_lr', default=1e-3, type=float)
    parser.add_argument('--gen_adv_lr', default=1e-4, type=float)
    parser.add_argument('--dis_lr', default=1e-4, type=float)
    parser.add_argument('--clip_norm', default=5.0, type=float)
    parser.add_argument('--pre_log_step', default=10, type=int)
    parser.add_argument('--adv_log_step', default=20, type=int)
    parser.add_argument('--train_data', default='dataset/train.txt', type=str)
    parser.add_argument('--test_data', default='dataset/test.txt', type=str)
    parser.add_argument('--temp_adpt', default='exp', type=str)
    parser.add_argument('--temperature', default=2, type=int)
    parser.add_argument('--ora_pretrain', default=True, type=int)
    parser.add_argument('--gen_pretrain', default=True, type=int)
    parser.add_argument('--dis_pretrain', default=True, type=int)

    # Generator
    parser.add_argument('--adv_g_step', default=1, type=int)
    parser.add_argument('--rollout_num', default=4, type=int)
    parser.add_argument('--gen_embed_dim', default=32, type=int)
    parser.add_argument('--gen_hidden_dim', default=32, type=int)
    parser.add_argument('--goal_size', default=16, type=int)
    parser.add_argument('--step_size', default=4, type=int)
    parser.add_argument('--mem_slots', default=cfg.mem_slots, type=int)
    parser.add_argument('--num_heads', default=cfg.num_heads, type=int)
    parser.add_argument('--head_size', default=cfg.head_size, type=int)

    # Discriminator
    parser.add_argument('--d_step', default=5, type=int)
    parser.add_argument('--d_epoch', default=3, type=int)
    parser.add_argument('--adv_d_step', default=5, type=int)
    parser.add_argument('--adv_d_epoch', default=3, type=int)
    parser.add_argument('--dis_embed_dim', default=64, type=int)
    parser.add_argument('--dis_hidden_dim', default=64, type=int)
    parser.add_argument('--num_rep', default=cfg.num_rep, type=int)

    # Log
    parser.add_argument('--log_file', default=cfg.log_filename, type=str)
    parser.add_argument('--save_root', default=cfg.save_root, type=str)
    parser.add_argument('--signal_file', default=cfg.signal_file, type=str)
    parser.add_argument('--tips', default='vanilla LeakGAN', type=str)

    return parser


# MAIN
if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser = program_config(parser)
    opt = parser.parse_args()
    cfg.init_param(opt)

    inst = LeakGANInstructor(opt)
    if not cfg.if_test:
        inst._run()
    else:
        inst._test()
