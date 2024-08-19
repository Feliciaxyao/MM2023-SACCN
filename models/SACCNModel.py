
from pyparsing import Opt
import torch
from torch import nn
import torch.nn.functional as F
from fairseq.data.data_utils import collate_tokens
from model.CCL import CCL                
from util.utils import l2norm
import random
import math
from model.gcn import AdjLearner, GCN ###add
#from block import fusions  #pytorch >= 1.1.0
#from gcn_conv import GATv2Conv
import  sklearn
from sklearn.preprocessing import StandardScaler

from .rnn import RNNEncoder


class SACCNModel(nn.Module):
    def __init__(self, roberta, opt):
        super(SACCNModel, self).__init__()

        self.opt = opt
        self.roberta = roberta
        self.input_streams = opt.input_streams
        self.video_fc = nn.Sequential(
            nn.Linear(opt.vid_feat_size, opt.dim),
            nn.ReLU(),
            nn.Dropout(opt.dropout),
        )

        self.final_fc = nn.Sequential(
            nn.Linear(opt.vid_feat_size, opt.dim),
            nn.ReLU(),
            nn.Dropout(opt.dropout),
            nn.Linear(768, 1),
            nn.Sigmoid()
        )
        
        self.text_trans = torch.nn.TransformerEncoderLayer(opt.dim, 8, opt.hidden_size, dropout=opt.dropout)
        self.vid_trans = torch.nn.TransformerEncoderLayer(opt.dim, 8, opt.hidden_size, dropout=opt.dropout)
        self.multi_trans = torch.nn.TransformerEncoderLayer(opt.dim, 8, opt.hidden_size, dropout=opt.dropout)

        self.linear_vision = nn.Sequential(nn.Linear(768, 1024), nn.ReLU(), nn.Linear(1024, 300))
        self.linear_300 = nn.Sequential(nn.Linear(768, 1024), nn.ReLU(), nn.Linear(1024, 300))

        #add
        hidden_size = 768
        hidden_size1 = 512
        input_dropout_p = 0.3
        self.adj_learner = AdjLearner(
            hidden_size, hidden_size, dropout=input_dropout_p)

        self.gcn = GCN(
            hidden_size,
            hidden_size,
            hidden_size,
            num_layers=2,
            dropout=input_dropout_p)



        self.gcn_atten_pool = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=-1))  # change to dim=-2 for attention-pooling otherwise sum-pooling
        #add

        self.lstm_raw = RNNEncoder(opt.dim, opt.dim // 2, bidirectional=True, dropout_p=0, n_layers=1, rnn_type='lstm')
        # self.aggregation_LSTM = nn.LSTM(
        #     input_size=opt.dim,
        #     hidden_size=opt.dim//2,
        #     num_layers=1,
        #     bidirectional=True,
        #     batch_first=True
        # )
        if opt.ccl:
            self.CCL = CCL(opt)

        if opt.self_attention:
            self.fc_virtual_video = nn.Linear(opt.dim, opt.virtual_class)
            self.fc_virtual_text = nn.Linear(opt.dim, opt.virtual_class)
            self.fc_virtual_konwledge = nn.Linear(opt.dim, opt.virtual_class)
            self.attention_self_video = torch.nn.MultiheadAttention(opt.virtual_class, 8)
            self.attention_self_text = torch.nn.MultiheadAttention(opt.virtual_class, 8)
            self.attention_cross_video = torch.nn.MultiheadAttention(opt.virtual_class, 8)
            self.attention_cross_text = torch.nn.MultiheadAttention(opt.virtual_class, 8)    

        else:

            self.MLP = nn.Sequential(
                # nn.Linear(opt.dim*3, opt.dim),
                # nn.Dropout(opt.dropout),
                nn.Linear(768, 1),
                ##nn.Linear(768, 2),
                # nn.Linear(768, 384),
                # nn.ReLU(),
                # nn.Linear(opt.dim * 8 * 2, opt.dim * 8),
                # nn.Dropout(0.3),
                # nn.Linear(opt.dim, 2),
            )
            def init_weights(m):
                if type(self.MLP) == nn.Linear:
                    nn.init.normal_(m.weight, std=0.01)

            self.MLP.apply(init_weights)

        self.softmax = nn.Softmax(dim=1)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, vid_input, sub_input, state_input, self_sup=True, test=False):
    #def forward(self, vid_input, sub_input, state_input, bert_features, self_sup=True, test=False):#[VLEP]
        if test:
            tokens1 = collate_tokens(
                [self.roberta.encode(pair[0], pair[1]) for pair in zip(sub_input, state_input[0])], pad_idx=1
            )

            tokens2 = collate_tokens(
                [self.roberta.encode(pair[0], pair[1]) for pair in zip(sub_input, state_input[1])], pad_idx=1
            )

            last_layer_features1 = self.roberta.extract_features(tokens1)
            last_layer_features2 = self.roberta.extract_features(tokens2)

            last_layer_features1 = self.li(last_layer_features1)
            last_layer_features2 = self.li(last_layer_features2)

            text_output1 = self.text_trans(last_layer_features1)
            text_output2 = self.text_trans(last_layer_features2)
            # text_output1 = self.trans(last_layer_features1)
            # text_output2 = self.trans(last_layer_features2)

            vid_output = None

            if 'vid' in self.input_streams:
                vid_feat, vid_lens = vid_input
                # vid_feat = vid_feat.cuda()
                vid_projected = self.video_fc(vid_feat)
                vid_output = self.vid_trans(vid_projected)
                # vid_output = self.trans(vid_projected)

            seq_len = text_output1.shape[1]
            concat_data = torch.cat([text_output1, text_output2, vid_output], dim=1)

            multi_output = self.multi_trans(concat_data)
            # multi_output = self.trans(concat_data)

            cls1 = multi_output[:, 0,:]
            cls2 = multi_output[:, seq_len,:]

            concat_cls = torch.cat([cls1, cls2], dim=1)

            output = self.MLP(concat_cls)

            return output

        else:
            
            batch_size = len(sub_input)
            encode = [self.roberta.encode(pair[0], pair[1]) for pair in zip(sub_input, state_input)] 
            #encode = [self.roberta.encode(pair[0], pair[1], pair[2]) for pair in zip(sub_input, state_input[0], state_input[1])] #VLEP
            tokens = collate_tokens(
                encode, pad_idx=1, pad_to_length=128
            )
            last_layer_features = self.roberta.extract_features(tokens)
            text_embed = self.text_trans(last_layer_features.transpose(0, 1)).transpose(0, 1)
            if 'vid' in self.input_streams:
                vid_feat, vid_lens = vid_input
                ##
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                vid_feat = vid_feat.to(device)

                vid_projected = self.video_fc(vid_feat)
                video_embed = self.vid_trans(vid_projected.transpose(0, 1)).transpose(0, 1)
            seq_len = video_embed.shape[1]
            concat_data = torch.cat([video_embed, text_embed], dim=1)
            hybrid_video_text = self.multi_trans(concat_data.transpose(0, 1)).transpose(0, 1)
            hybrid_video = hybrid_video_text[:, 0:seq_len, :]
            hybrid_text = hybrid_video_text[:, seq_len:, :]
            

            a_video = None
            a_text = None
            norm = 0
            # consensus
            if self.opt.ccl:
                hybrid_video, hybrid_text, a_video, a_text, average = self.CCL(hybrid_video, hybrid_text)
                #print(hybrid_text.shape)
                #print(hybrid_video.shape)
                a_video_norm = torch.norm(a_video, p=1, dim=1).sum()
                a_text_norm = torch.norm(a_text, p=1, dim=1).sum()
                average_norm = torch.norm(average, p=1, dim=1).sum()
                norm = (a_video_norm + a_text_norm + average_norm) / 3
                hybrid = torch.cat([hybrid_video, hybrid_text], dim=1).mean(dim=1) #

                ### add-GCN
                adj = self.adj_learner(hybrid_text, hybrid_video)
                ###adj1 = self.adj_learner(hybrid_text1, hybrid_video1)
                # q_v_inputs of shape (batch_size, q_v_len, hidden_size)
                q_v_inputs = torch.cat((hybrid_text, hybrid_video), dim=1)
                ###q_v_inputs1 = torch.cat((hybrid_text1, hybrid_video1), dim=1)
                # q_v_output of shape (batch_size, q_v_len, hidden_size)
                q_v_output = self.gcn(q_v_inputs, adj)
                #q_v_output2 = self.gat(q_v_inputs, adj)
                ###q_v_output1 = self.gcn(q_v_inputs1, adj1)
                #q_v_output1 = self.gcn_conv(q_v_inputs, adj)

                q_v_output_ = q_v_output.mean(dim=1)
                ##q_v_output_1 = q_v_inputs1.mean(dim=1)

                ## attention pool
                local_attn = self.gcn_atten_pool(q_v_output)
                # print(local_attn)
                local_out = torch.sum(q_v_output * local_attn, dim=1)
             
                #MLP_output1 = self.MLP(q_v_output_)
                MLP_output1 = self.MLP(local_out)
                ###MLP_output1 = self.MLP(q_v_output_1)
                MLP_output2 = self.sig(MLP_output1)
                return MLP_output1, a_video, a_text, average_norm
                # a_video = self.softmax(a_video)
                # a_text = self.softmax(a_text)





            if self.opt.self_attention:
                video_virtual = self.relu(self.fc_virtual_video(hybrid_video))
                text_virtual = self.relu(self.fc_virtual_text(hybrid_text))
                bert_features = self.fc_virtual_konwledge(bert_features)
                video_self = self.attention_self_video(video_virtual, video_virtual, video_virtual)[0]
                video_cross = self.attention_cross_video(video_virtual, text_virtual, text_virtual)[0]
                text_self = self.attention_self_text(text_virtual, text_virtual, text_virtual)[0]
                text_cross = self.attention_cross_text(text_virtual, video_virtual, video_virtual)[0]
                hybrid_video = video_virtual + video_self + video_cross
                hybrid_text = text_virtual + text_self + text_cross







