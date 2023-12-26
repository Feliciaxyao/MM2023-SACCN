import time
import torch
import argparse


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir_base", type=str, default="./results")
    parser.add_argument("--feat_dir", type=str, default="./dataset/violin")
    parser.add_argument("--bert_dir", type=str, default="./roberta.base")
    parser.add_argument("--model", type=str, default="SACCNModel", choices=['ViolinBase', 'VlepBase', 'SACCNModel'])
    parser.add_argument("--data", type=str, default="ViolinDataset", choices=['VlepDataset', 'ViolinDataset'])
    parser.add_argument("--lr1", type=float, default=5e-6, help="learning rate")
    parser.add_argument("--lr2", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="weight decay")
    parser.add_argument("--first_n_epoch", type=int, default=30, help="number of epochs to run")
    parser.add_argument("--second_n_epoch", type=int, default=3, help="number of epochs to run")
    parser.add_argument("--batch_size", type=int, default=16, help="mini-batch size")
    parser.add_argument("--test_batch_size", type=int, default=16, help="mini-batch size for testing")

    parser.add_argument("--feat", type=str, nargs="+", default=['c3d'], choices=['2d', '3d', 'resnet', 'c3d'])
    parser.add_argument("--vid_feat_size", type=int, default=2048, help="visual feature dimension")
    parser.add_argument("--input_streams", type=str, nargs="+", choices=["vid", "sub", "none"], default=['vid' 'sub'],
                        help="input streams for the model, or use a single option 'none'")

    parser.add_argument("--dim", type=int, default=768, help="input size for the encoderLayer")
    parser.add_argument("--hidden_size", type=int, default=2048, help="hidden size for the encoderLayer")
    parser.add_argument('--lstm_hidden_size', default=100, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)

    parser.add_argument('--a_con', default=0.5, type=float)
    parser.add_argument("--ccl", action='store_true')
    parser.add_argument("--ccl_loss", action='store_true')
    parser.add_argument("--consensus_weight", type=int, default=1)


    parser.add_argument("--self_attention", action='store_true')
        
    parser.add_argument("--ssl", action='store_true')
    parser.add_argument("--self_loss_weight", type=float, default=3)
    parser.add_argument('--pretrain_epoches', type=int, default=5, help='number of epochs for normal training')

    parser.add_argument("--test", action='store_true')
    parser.add_argument("--model_path", type=str, default="./VIOLIN_SACCN_checkpoint.pth")
    parser.add_argument("--frame", type=str, default="", choices=['first', 'last', 'mid', ''],help="testing with only one frame")

    opt = parser.parse_args()
    opt.results_dir = opt.results_dir_base + time.strftime("_%Y_%m_%d_%H_%M_%S") + '_' + opt.model
    opt.results_dir += '_' + str(opt.lr1)
    if opt.ccl:
        opt.results_dir += '_ccl'
    if opt.ccl_loss:
        opt.results_dir += '_' + str(opt.consensus_weight)
    if opt.pretrain_epoches:
        opt.results_dir += '_pretrain' + str(opt.pretrain_epoches)


    #opt.vid_feat_size = 2048 if opt.feat == 'resnet' else 4096

    return opt
