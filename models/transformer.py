import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn



class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x = x + self.pe[:x.size(0), :]
        x = x + self.pe[:, :x.size(1)]
        return x


################## The following modules are obsolete and updated versions can be found in poseformer.py ##################


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, intermediate=False, mdn=False, encoding=True, pred_dropout=0.0):

        super().__init__()

        self.encoding = encoding
        if encoding:
            self.position_enc = PositionalEncoding(n_src)
        self.dropout = nn.Dropout(p=dropout)
        self.intermediate = intermediate
        self.mdn = mdn
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        if intermediate:
            if mdn:
                # TODO: Add prediction dropout here as well?
                self.intermediate_pred = nn.ModuleList([nn.Linear(16 * d_model, 16 * 3 * 5) for _ in range(n_layers)])
                self.intermediate_enc = nn.ModuleList([nn.Linear(16 * 3 * 5, 16 * d_model) for _ in range(n_layers)])
                self.intermediate_sigma = nn.ModuleList([nn.Linear(16 * d_model, 5) for _ in range(n_layers)])
                self.intermediate_alpha = nn.ModuleList([nn.Linear(16 * d_model, 5) for _ in range(n_layers)])
            else:  
                self.intermediate_pred = nn.ModuleList([nn.Sequential(nn.Dropout(p=pred_dropout), nn.Linear(16 * d_model, 16 * 3)) for _ in range(n_layers)])
                # self.intermediate_pred = nn.ModuleList([nn.Sequential(nn.Linear(16 * d_model, 16 * d_model), nn.ReLU(inplace=True), nn.Linear(16 * d_model, 16 * 3)) for _ in range(n_layers)])
                self.intermediate_enc = nn.ModuleList([nn.Linear(16 * 3, 16 * d_model) for _ in range(n_layers)])
                # self.intermediate_enc = nn.ModuleList([nn.Conv1d(3, d_model, 1) for _ in range(n_layers)])

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []
        intermediate_list = []

        if self.mdn:
            mu_list = []
            sigma_list = []
            alpha_list = []

        # -- Forward
        
        if self.encoding:
            src_seq = self.position_enc(src_seq)
        enc_output = self.dropout(src_seq)
        enc_output = self.layer_norm(enc_output)
        b = src_seq.size(0)

        i = 0
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)

            if self.intermediate:
                if self.mdn:
                    enc = enc_output.clone().view(b, -1)
                    mu = self.intermediate_pred[i](enc)
                    res = self.intermediate_enc[i](mu).view(b, 16, -1)
                    enc_output += res
                    sigma = self.intermediate_sigma[i](enc)
                    sigma = nn.functional.elu(sigma) + 1
                    alpha = self.intermediate_alpha[i](enc)
                    alpha = nn.functional.softmax(alpha, dim=1)
                    mu_list += [mu]
                    sigma_list += [sigma]
                    alpha_list += [alpha]
                else:
                    pred = self.intermediate_pred[i](enc_output.clone().view(b, -1))
                    res = self.intermediate_enc[i](pred).view(b, 16, -1)
                    # res = self.intermediate_enc[i](pred.view(b, 16, -1).permute(0, 2, 1)).permute(0, 2, 1)
                    enc_output += res
                    intermediate_list += [pred] if self.intermediate else []

            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
            # intermediate_list += [enc_output] if self.intermediate else []
            
            i += 1

        if return_attns:
            if self.intermediate:
                return intermediate_list, enc_slf_attn_list
            return enc_output, enc_slf_attn_list
        if self.intermediate:
            if self.mdn:
                return mu_list, sigma_list, alpha_list
            else:
                return intermediate_list
        return enc_output,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src,
            d_word_vec=256, d_model=256, d_inner=512,
            n_layers=6, n_head=8, dropout=0.2, d_k=64, d_v=64, intermediate=False, mdn=False, encoding=True, pred_dropout=0.0):

        super().__init__()

        self.intermediate = intermediate
        self.mdn = mdn

        self.encoder = Encoder(
            n_src=n_src,
            d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout, intermediate=intermediate, mdn=mdn, encoding=encoding, pred_dropout=pred_dropout)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'
        if mdn:
            assert intermediate, 'Only allows mdn for intermediate supervision.'

    # (MATT) for liftformer architecture we only use the encoder layers
    def forward(self, src_seq, src_mask=None):
        if self.intermediate:
            if self.mdn:
                mu_list, sigma_list, alpha_list = self.encoder(src_seq, src_mask)
                return mu_list, sigma_list, alpha_list
            else:
                intermediate_list = self.encoder(src_seq, src_mask)
                return intermediate_list
        else:
            enc_output, *_ = self.encoder(src_seq, src_mask)

            return enc_output


class LiftFormer(nn.Module):

    def __init__(self, num_joints_in, j_nfeatures, 
                 c_filter_width = 3, t_nhead = 12, t_nhid = 512, 
                 t_nlayers=6, out_channels=512, dropout=0.0, intermediate=False, mdn=False, spatial_encoding=False, 
                 conv_enc=False, conv_dec=False, attn_mask=None, pred_dropout=0.0, use_images=False):
        super(LiftFormer, self).__init__()

        """
        Initialize the model.
        
        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        j_nfeatures -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        r_field -- receptive field
        c_filter_width -- 1D convolution kernel width
        t_nhead -- number of multi-attention heads in transformer encoder layer
        t_nhid -- number of hidden layers in transformer encoder layer
        t_nlayers -- number of encoder layers
        out_channels -- number of convolution channels 
        """

        self.num_joints_in = num_joints_in
        self.j_nfeatures = j_nfeatures
        self.t_nhid = t_nhid
        self.intermediate = intermediate
        self.mdn = mdn
        self.spatial_encoding = spatial_encoding
        self.conv_enc = conv_enc
        self.conv_dec = conv_dec
        self.attn_mask = attn_mask
        self.use_images = use_images

        # in channels, out channels, stride
        # c_padding = (c_filter_width-1) // 2
        # self.expand_conv = nn.Conv1d(j_nfeatures, out_channels, 1)

        if use_images:
            feature_expansion = t_nhid // 2
            self.image_expand = nn.Sequential(
                nn.Conv2d(3, feature_expansion//4, kernel_size=5, padding=0),
                nn.ReLU(),
                nn.Conv2d(feature_expansion//4, feature_expansion//2, kernel_size=5, padding=0),
                nn.ReLU(),
                nn.Conv2d(feature_expansion//2, feature_expansion, kernel_size=5, padding=0),
                nn.ReLU(),
                nn.Conv2d(feature_expansion, feature_expansion, kernel_size=3, padding=0),
                nn.ReLU()
            )
        else:
            feature_expansion = t_nhid

        if conv_enc:
            self.expand_conv = nn.Conv1d(j_nfeatures, feature_expansion, 1)
        else:
            self.expand_conv = nn.Linear(j_nfeatures*num_joints_in, num_joints_in * feature_expansion)
        # self.expand_conv = nn.ModuleList([nn.Linear(j_nfeatures, t_nhid) for i in range(num_joints_in)])

        if spatial_encoding:
            self.spatial_embedding = nn.Parameter(torch.zeros(1, num_joints_in, t_nhid))

        # Encoder layers only! LiftFormer does not require decoder layers
        self.transformer_encoder = Transformer(t_nhid, d_word_vec=t_nhid, d_model=t_nhid, n_layers=t_nlayers, dropout=dropout, intermediate=intermediate, mdn=mdn and intermediate, encoding= not spatial_encoding, pred_dropout=pred_dropout)

        # decoder takes middle channel from tformer as current frame for pred 
        # self.decoder = nn.Conv1d(t_nhid, 3, 1, bias=False)
        if not self.intermediate:
            # TODO: Add prediction dropout here as well?
            if self.mdn:
                self.decoder_mu = nn.Linear(num_joints_in * t_nhid, num_joints_in * 3 * 5)
                self.decoder_sigma = nn.Linear(num_joints_in * t_nhid, 5)
                self.decoder_alpha = nn.Linear(num_joints_in * t_nhid, 5)
            else:
                if self.conv_dec:
                    self.decoder = nn.Conv1d(t_nhid, 3, 1)
                else:
                    self.decoder = nn.Sequential(
                        nn.Dropout(p=pred_dropout),
                        nn.Linear(num_joints_in * t_nhid, num_joints_in * 3)
                    )
        # self.decoder = nn.ModuleList([nn.Linear(t_nhid, 3) for i in range(num_joints_in)])

        # self.lift_bn = nn.BatchNorm1d(1, momentum=0.1)

        # self._weight_fuckery()

        # self.expand_conv.register_forward_pre_hook(self.forward_fuckery_enc)
        # self.decoder.register_forward_pre_hook(self.forward_fuckery_dec)

    def _weight_fuckery(self):
        enc = self.expand_conv.weight.data
        dec = self.decoder.weight.data
        for i in range(self.num_joints_in):
            enc[i*self.t_nhid:i*self.t_nhid+self.t_nhid, :i*self.j_nfeatures] = 0
            enc[i*self.t_nhid:i*self.t_nhid+self.t_nhid, i*self.j_nfeatures+self.j_nfeatures:] = 0

            dec[:i*3, i*self.t_nhid:i*self.t_nhid+self.t_nhid] = 0
            dec[i*3+3:, i*self.t_nhid:i*self.t_nhid+self.t_nhid] = 0

        self.expand_conv.weight.data = enc
        self.decoder.weight.data = dec

    def forward_fuckery_enc(self, module, input):
        enc = module.weight.data
        for i in range(self.num_joints_in):
            enc[i*self.t_nhid:i*self.t_nhid+self.t_nhid, :i*self.j_nfeatures] = 0
            enc[i*self.t_nhid:i*self.t_nhid+self.t_nhid, i*self.j_nfeatures+self.j_nfeatures:] = 0
        self.expand_conv.weight.data = enc

    def forward_fuckery_dec(self, module, input):
        dec = module.weight.data
        for i in range(self.num_joints_in):
            dec[:i*3, i*self.t_nhid:i*self.t_nhid+self.t_nhid] = 0
            dec[i*3+3:, i*self.t_nhid:i*self.t_nhid+self.t_nhid] = 0
        self.decoder.weight.data = dec

    def forward(self, src, image_features=None):
        # src = src.permute(0, 2, 1)
        # src = self.expand_conv(src)
        # src = src.permute(0,2,1)
        # output = self.transformer_encoder(src)
        # output = output.permute(0,2,1)
        # output = self.decoder(output)
        # output = output.permute(0,2,1)

        b, j, c = src.shape

        if self.conv_enc:
            src = self.expand_conv(src.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            src = self.expand_conv(src.view(b, -1)).view(b, j, -1)

        if self.use_images:
            image_features = self.image_expand(image_features.view(-1, 3, 15, 15)).view(b, 16, -1)
            src = torch.cat([src, image_features], dim=2)

        if self.spatial_encoding:
            src += self.spatial_embedding
        if self.intermediate:
            # intermediate = self.transformer_encoder(src)
            # out = [self.decoder[i](out.view(b, -1)).view(b, j, -1) for i, out in enumerate(intermediate)]
            if self.mdn:
                intermediate_mu, intermediate_sigma, intermediate_alpha = self.transformer_encoder(src, self.attn_mask)
                intermediate_mu = [out.view(b, j*3, 5) for out in intermediate_mu]

                out = [torch.cat([mu, sigma.unsqueeze(1), alpha.unsqueeze(1)], dim=1) for mu, sigma, alpha in zip(intermediate_mu, intermediate_sigma, intermediate_alpha)]
            else:
                intermediate = self.transformer_encoder(src, self.attn_mask)
                out = [out.view(b, j, -1) for out in intermediate]
        else:
            out = self.transformer_encoder(src, self.attn_mask)
            if self.mdn:
                mu = self.decoder_mu(out.view(b, -1)).view(b, j*3, 5)
                sigma = self.decoder_sigma(out.view(b, -1)).view(b, 5)
                sigma = nn.functional.elu(sigma) + 1
                alpha = self.decoder_alpha(out.view(b, -1)).view(b, 5)
                alpha = nn.functional.softmax(alpha, dim=1)

                out = torch.cat([mu, sigma.unsqueeze(1), alpha.unsqueeze(1)], dim=1)
            else:
                if self.conv_dec:
                    out = self.decoder(out.permute(0, 2, 1)).permute(0, 2, 1)
                else:
                    out = self.decoder(out.view(b, -1)).view(b, j, -1)

        # src = torch.cat([self.expand_conv[i](src[:, i, :]).unsqueeze(1) for i in range(self.num_joints_in)], dim=1)
        # out = self.transformer_encoder(src)
        # out = torch.cat([self.decoder[i](out[:, i, :]).unsqueeze(1) for i in range(self.num_joints_in)], dim=1)

        return out