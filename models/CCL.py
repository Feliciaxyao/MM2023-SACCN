import torch
from torch import nn
import sys
sys.path.append("")
from config import get_argparse


class CCL(nn.Module):
    def __init__(self, opt):
        super(CCL, self).__init__()

        self.opt = opt
        self.input_streams = opt.input_streams
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.globa_cov = nn.Conv1d(opt.dim, opt.dim, 3, 1, 1)
        self.cross_cov = nn.Conv1d(opt.dim, opt.dim, 3, 1, 1)
        self.sig = nn.Sigmoid()

        self.conv_length = nn.Conv2d(1, 1, (3, opt.dim), 1, (1, 0))

        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, 3, 1, 1),
            nn.Conv1d(8, 8, 3, 1, 1),
            nn.Conv1d(8, 1, 3, 1, 1),
        )

    def forward(self, video, text):
            
        video_len = video.shape[1]
        text_len = text.shape[1]
        if video_len < text_len:
            video = self.cat(video, text_len)
        else:
            text = self.cat(text, video_len)

        video_tmp = video
        text_tmp = text

        video_global_context = self.global_pool(video.permute(0, 2, 1)).permute(0, 2, 1)
        text_global_context = self.global_pool(text.permute(0, 2, 1)).permute(0, 2, 1)
        # global_context = self.globa_cov(x)
        # cross_modal = self.cross_cov(text)
        x = video_global_context * text
        ##x_local = video * text
        ###x_local = video * text_global_context
        video_M = self.sig(x)
        ##video_M = self.sig(x_local)

        ##text_global_context = self.global_pool(text.permute(0, 2, 1)).permute(0, 2, 1)
        # text_global_context = self.globa_cov(y)
        # cross_modal = self.cross_cov(video)
        y = text_global_context * video
        ##y_local = text * video
        ###y_local = text * video_global_context
        text_M = self.sig(y)
        ##text_M = self.sig(y_local)

        video_output = video_tmp * video_M
        text_output = text_tmp * text_M

        video_tmp = video_output.unsqueeze(1)
        text_tmp = text_output.unsqueeze(1)

        video_tmp = self.conv_length(video_tmp).squeeze(3)
        text_tmp = self.conv_length(text_tmp).squeeze(3)

        a_video = self.conv(video_tmp).squeeze(1)
        a_text = self.conv(text_tmp).squeeze(1)

        video_tmp = a_video.unsqueeze(2)
        text_tmp = a_text.unsqueeze(2)

        video_output = video_output * video_tmp
        text_output = text_output * text_tmp
        
        average = (a_video + a_text) / 2
        return video_output, text_output, a_video, a_text, average

    def cat(self, input, length):
        original_len = input.shape[1]
        time = length // original_len
        input = input.repeat(1, time, 1)

        input = torch.cat((input,input[:, 0:length%original_len, :]), dim=1)
        return input

if __name__ == '__main__':
    opt = get_argparse()
    video = torch.randn(16, 120, 512)
    text = torch.randn(16, 50, 512)
    model = CCL(opt)


    video_output, text_output, average = model(video, text)

