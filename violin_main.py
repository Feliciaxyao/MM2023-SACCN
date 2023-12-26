import os
import math


os.environ["CUDA_VISIBLE_DEVICES"]= "1"
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
import numpy as np
from tqdm import tqdm
import json
import torch
torch.set_num_threads(1)
import torch.nn as nn
from torch.utils.data import DataLoader

from violin_dataset import ViolinDataset, violin_pad_collate
from config import get_argparse
from model.SACCNModel import SACCNModel

from collections.abc import Iterable
from fairseq.models.roberta import RobertaModel
#from apex import amp
#from transformers import get_linear_schedule_with_warmup
#from loss import ContrastiveLoss

def check_param(model):
    grad_lst = []
    for name, param in model.named_parameters():
        grad_lst.append(torch.norm(param.grad.data.view(-1)).item())
    return grad_lst

def get_data_loader(opt, dset, batch_size, if_shuffle):
    return DataLoader(dset, batch_size=batch_size, shuffle=if_shuffle, num_workers=0, drop_last=True, collate_fn=violin_pad_collate)

def loss_fn_entropy(output, target):
    return nn.CrossEntropyLoss()(output, target)

def loss_fn_bce(input, target):
    return nn.BCEWithLogitsLoss()(input, target)
    

def loss_fn_mse(input, target):
    return nn.MSELoss()(input, target)

def loss_fn_cos(input, target):
    return torch.cosine_similarity(input, target, dim=1)

def loss_modality(input, target):
    return nn.BCEWithLogitsLoss()(input, target)


def validate(model, valid_loader):
    model.eval()
    with torch.no_grad():
        valid_loss = []
        valid_corrects = []
        clip_ids = []

        # all_acc = []
        # all_corrects = []

        for batch_idx, batch in enumerate(tqdm(valid_loader)):

            clip_ids, padded_vid_feat, sub_input, state_input, modality_text, modality_video = batch
            state_input = list(map(list, zip(*state_input)))

           
            real_state_output = model(padded_vid_feat, sub_input, state_input[0])
            fake_state_output = model(padded_vid_feat, sub_input, state_input[1])

            real_state_output, a_video_real, a_text_real, average1 = real_state_output
            fake_state_output, a_video_fake, a_text_fake, average1 = fake_state_output
            real_state_output = real_state_output.squeeze()
            fake_state_output = fake_state_output.squeeze()


            threshold = 0.5
            loss = torch.mean(-torch.log(1.0-fake_state_output)-torch.log(real_state_output), dim=0)
            loss_sum = torch.sum(-torch.log(1.0 - fake_state_output) - torch.log(real_state_output), dim=0)

            valid_corrects += (real_state_output>=threshold).to(torch.device('cpu')).tolist()
            valid_corrects += (fake_state_output<threshold).to(torch.device('cpu')).tolist()


            a_video1_real_ = a_video_real.detach()
            a_text1_real_ = a_text_real.detach()
            a = 0.5
            loss_video_real = loss_fn_mse(a_video_real, a_text1_real_) #
            loss_text_real = loss_fn_mse(a_text_real, a_video1_real_) #
            loss_consensus_real = a * loss_video_real + (1 - a) * loss_text_real

            a_video1_fake_ = a_video_fake.detach()
            a_text1_fake_ = a_text_fake.detach()
            a = 0.5
            loss_video_fake = loss_fn_mse(a_video_fake, a_text1_fake_)
            loss_text_fake = loss_fn_mse(a_text_fake, a_video1_fake_)
            
            loss_consensus_fake = a * loss_video_fake + (1 - a) * loss_text_fake

            b = 24
            
            loss = loss + b * (loss_consensus_real + loss_consensus_fake)

            valid_loss.append(loss_sum.item())

        valid_acc = sum(valid_corrects) / float(len(valid_corrects))
        valid_loss = sum(valid_loss) / float(len(valid_corrects))
        

    return valid_loss, valid_acc




if __name__ == '__main__':
    random_seed = 12412412
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)

    opt = get_argparse()

    roberta = RobertaModel.from_pretrained(opt.bert_dir, checkpoint_file='model.pt')
    # for p in roberta.parameters():
      #   p.requires_grad = False
    roberta.cuda()
    roberta.eval()

    #freeze_by_idxs(roberta.model, 0)

    DSET = eval(opt.data)


    if not opt.test:
        os.makedirs(opt.results_dir)
        trn_dset = DSET(opt, 'train')
        # val_dset = DSET(opt, bert_tokenizer, 'validate')
        # val_dset = DSET(opt, 'dev')
        val_dset = DSET(opt, 'validate')
        tst_dset = DSET(opt, 'test')
    else:
        tst_dset = DSET(opt, 'test')

    model = eval(opt.model)(roberta, opt)
    print(model)

    if opt.test:
        model.load_state_dict(torch.load(opt.model_path))
        
    model.cuda()

    if opt.test:
        # model = amp.initialize(model, opt_level="O1")
        test_loader = get_data_loader(opt, tst_dset, opt.test_batch_size, False)
        test_loss, test_acc = validate(model, test_loader)
        print("Test loss %.4f acc %.4f\n"
            % (test_loss, test_acc))
        with open(opt.model_path+'_test.res','w') as ftst:
            ftst.write("Test loss %.4f acc %.4f\n"
            % (test_loss, test_acc))
    