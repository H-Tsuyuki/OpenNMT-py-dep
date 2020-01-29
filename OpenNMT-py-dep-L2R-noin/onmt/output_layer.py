import torch
import torch.nn as nn
import torch.nn.functional as F
#from onmt.decoders.transformer import TransformerDecoderLayer

class Generator(nn.Module):
    def __init__(self, opt, dicts, word_emb, label_emb):
        super(Generator, self).__init__()
        self.fc_word = nn.Linear(opt.dec_rnn_size, opt.dec_rnn_size)
        self.word_proj = nn.Linear(opt.dec_rnn_size, len(dicts["tgt"]))
        self.logsm = nn.LogSoftmax(dim=-1)

    def forward(self, dec_out):
        dec_out = dec_out.transpose(0,1)
        
        out = self.fc_word(dec_out) 
        word_p = self.word_proj(out)
        
        word_logprob = self.logsm(word_p)
        return word_logprob
