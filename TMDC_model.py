import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import *

class Denoising(nn.Module):
    def __init__(self, args, adim, tdim, vdim, D_e, n_classes, depth=4, num_heads=4, mlp_ratio=1, drop_rate=0, attn_drop_rate=0, no_cuda=False):
        super(Denoising, self).__init__()
        self.n_classes = n_classes
        self.D_e = D_e
        self.num_heads = num_heads
        D = 3 * D_e
        self.device = args.device
        self.no_cuda = no_cuda
        self.adim, self.tdim, self.vdim = adim, tdim, vdim
        self.out_dropout = args.drop_rate
        self.depth = depth
        self.test_condition = args.test_condition
        
        self.as_in_proj = nn.Conv1d(self.adim, D_e, kernel_size=1, padding=0, bias=False)
        self.ts_in_proj = nn.Conv1d(self.tdim, D_e, kernel_size=1, padding=0, bias=False)
        self.vs_in_proj = nn.Conv1d(self.vdim, D_e, kernel_size=1, padding=0, bias=False)
        
        self.dropout_a = nn.Dropout(args.drop_rate)
        self.dropout_t = nn.Dropout(args.drop_rate)
        self.dropout_v = nn.Dropout(args.drop_rate)

        self.intra_attn_as = Attention(dim=self.D_e,num_heads=self.num_heads,mlp_ratio=mlp_ratio,attn_drop=attn_drop_rate)
        self.intra_attn_ts = Attention(dim=self.D_e,num_heads=self.num_heads,mlp_ratio=mlp_ratio,attn_drop=attn_drop_rate)
        self.intra_attn_vs = Attention(dim=self.D_e,num_heads=self.num_heads,mlp_ratio=mlp_ratio,attn_drop=attn_drop_rate)
        
        self.inter_attn_common = Attention(dim=self.D_e,num_heads=self.num_heads,mlp_ratio=mlp_ratio,attn_drop=attn_drop_rate)

        self.as_vib = VIB(dim=self.D_e, encoding_dim=self.D_e, output_dim=n_classes)
        self.ts_vib = VIB(dim=self.D_e, encoding_dim=self.D_e, output_dim=n_classes)
        self.vs_vib = VIB(dim=self.D_e, encoding_dim=self.D_e, output_dim=n_classes)

        self.as_ts_vs_common_vib = VIB(dim=self.D_e, encoding_dim=self.D_e, output_dim=n_classes)
        
        self.proj1 = nn.Linear(D, D)
        self.proj2 = nn.Linear(D, D)
        self.nlp_head = nn.Linear(D, n_classes)
        self.classifier_a = nn.Linear(D_e, n_classes)
        self.classifier_t = nn.Linear(D_e, n_classes)
        self.classifier_v = nn.Linear(D_e, n_classes)

        self.classifier_a_common = nn.Linear(D_e, n_classes)
        self.classifier_t_common = nn.Linear(D_e, n_classes)
        self.classifier_v_common = nn.Linear(D_e, n_classes)
            
    def forward(self, inputfeats, input_features_mask=None, umask=None, first_stage=False):
        """
        inputfeats -> ?*[seqlen, batch, dim]
        qmask -> [batch, seqlen]
        umask -> [batch, seqlen]
        seq_lengths -> each conversation lens
        input_features_mask -> ?*[seqlen, batch, 3]
        """
        audio, text, video = inputfeats[:, :, :self.adim], inputfeats[:, :, self.adim:self.adim + self.tdim], \
        inputfeats[:, :, self.adim + self.tdim:]
        seq_len, B, C = audio.shape
        
        # --> [batch, seqlen, dim]
        # audio, text, video = audio.permute(1, 0, 2), text.permute(1, 0, 2), video.permute(1, 0, 2)
        # --> [batch, dim, seqlen]
        audio, text, video = audio.permute(1, 2, 0), text.permute(1, 2, 0), video.permute(1, 2, 0)
        proj_a = self.dropout_a(self.as_in_proj(audio))
        proj_t = self.dropout_t(self.ts_in_proj(text))
        proj_v = self.dropout_v(self.vs_in_proj(video))
        

        # --> [batch, seqlen, 3]
        input_mask = torch.clone(input_features_mask.permute(1, 0, 2))
        input_mask[umask == 0] = 0
        # --> [batch, 3, seqlen] -> [batch, 3*seqlen]
        attn_mask = input_mask.transpose(1, 2).reshape(B, -1)

        # change to --> [batch, seqlen, dim]
        proj_a, proj_t, proj_v = proj_a.permute(0, 2, 1), proj_t.permute(0, 2, 1), proj_v.permute(0, 2, 1)
        # --> [batch, 3*seqlen, dim]

        # Remove the noises
        (mu_a, std_a), denoise_a, out_a, a_kls = self.as_vib(proj_a)
        (mu_t, std_t), denoise_t, out_t, t_kls = self.ts_vib(proj_t)
        (mu_v, std_v), denoise_v, out_v, v_kls = self.vs_vib(proj_v)

        (mu_a_common, std_a_common), denoise_a_common, out_a_common, a_kls_common = self.as_ts_vs_common_vib(proj_a)
        (mu_t_common, std_t_common), denoise_t_common, out_t_common, t_kls_common = self.as_ts_vs_common_vib(proj_t)
        (mu_v_common, std_v_common), denoise_v_common, out_v_common, v_kls_common = self.as_ts_vs_common_vib(proj_v)
        
        # intra-modality interaction
        intra_as = self.intra_attn_as(denoise_a, denoise_a)
        intra_ts = self.intra_attn_ts(denoise_t, denoise_t)
        intra_vs = self.intra_attn_vs(denoise_v, denoise_v)

        inter_as = self.inter_attn_common(denoise_a_common, denoise_a_common)
        inter_ts = self.inter_attn_common(denoise_t_common, denoise_t_common)
        inter_vs = self.inter_attn_common(denoise_v_common, denoise_v_common)

        outputs_intra_a = self.classifier_a(intra_as)
        outputs_intra_t = self.classifier_t(intra_ts)
        outputs_intra_v = self.classifier_v(intra_vs)

        outputs_inter_a = self.classifier_a_common(inter_as)
        outputs_inter_t = self.classifier_t_common(inter_ts)
        outputs_inter_v = self.classifier_v_common(inter_vs)
        
        if first_stage:
            x = torch.cat([denoise_a, denoise_t, denoise_v], dim=1)# [32, 189, 256]
        else:
            # meaningless
            out_a, out_t, out_v, out_a_common, out_t_common, out_v_common = torch.rand((B, seq_len, self.n_classes)), torch.rand((B, seq_len, self.n_classes)), torch.rand((B, seq_len, self.n_classes)), torch.rand((B, seq_len, self.n_classes)), torch.rand((B, seq_len, self.n_classes)), torch.rand((B, seq_len, self.n_classes))
            a_kls, t_kls, v_kls, a_kls_common, t_kls_common, v_kls_common = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
            outputs_intra_a, outputs_intra_t, outputs_intra_v, outputs_inter_a, outputs_inter_t, outputs_inter_v = torch.rand((B, seq_len, self.n_classes)), torch.rand((B, seq_len, self.n_classes)), torch.rand((B, seq_len, self.n_classes)), torch.rand((B, seq_len, self.n_classes)), torch.rand((B, seq_len, self.n_classes)), torch.rand((B, seq_len, self.n_classes))

            # assert self.test_condition in ['a', 't', 'v', 'at', 'av', 'tv', 'atv']
            if self.test_condition == 't':
                fusd_spe_ts = self.intra_attn_ts(denoise_t, denoise_t_common)
                fusd_common_as = self.inter_attn_common(denoise_t_common, denoise_t)
                fusd_common_vs = self.inter_attn_common(denoise_t_common, denoise_t)
                all_fused = torch.cat([fusd_common_as, fusd_spe_ts, fusd_common_vs], dim=1)
                x = all_fused
                
            elif self.test_condition == 'a':
                fusd_spe_as = self.intra_attn_as(denoise_a, denoise_a_common)
                fusd_common_ts = self.inter_attn_common(denoise_a_common, denoise_a)
                fusd_common_vs = self.inter_attn_common(denoise_a_common, denoise_a)
                all_fused = torch.cat([fusd_spe_as, fusd_common_ts, fusd_common_vs], dim=1)
                x = all_fused
            elif self.test_condition == 'v':
                fusd_spe_vs = self.intra_attn_vs(denoise_v, denoise_v_common)
                fusd_common_as = self.inter_attn_common(denoise_v_common, denoise_v)
                fusd_common_ts = self.inter_attn_common(denoise_v_common, denoise_v)
                all_fused = torch.cat([fusd_common_as, fusd_common_ts, fusd_spe_vs], dim=1)
                x = all_fused
            elif self.test_condition == 'at':
                fusd_spe_as = self.intra_attn_as(denoise_a, denoise_a_common)
                fusd_spe_ts = self.intra_attn_ts(denoise_t, denoise_t_common)
                fusd_common_vs = self.inter_attn_common(denoise_t_common, denoise_a) + self.inter_attn_common(denoise_a_common, denoise_t)
                all_fused = torch.cat([fusd_spe_as, fusd_spe_ts, fusd_common_vs], dim=1)
                x = all_fused
            elif self.test_condition == 'av':
                fusd_spe_as = self.intra_attn_as(denoise_a, denoise_a_common)
                fusd_common_ts = self.inter_attn_common(denoise_v_common, denoise_a) + self.inter_attn_common(denoise_a_common, denoise_v)
                fusd_spe_vs = self.intra_attn_vs(denoise_v, denoise_v_common)
                all_fused = torch.cat([fusd_spe_as, fusd_common_ts, fusd_spe_vs], dim=1)
                x = all_fused
            elif self.test_condition == 'tv':
                fusd_common_as = self.inter_attn_common(denoise_t_common, denoise_v) + self.inter_attn_common(denoise_v_common, denoise_t)
                fusd_spe_ts = self.intra_attn_ts(denoise_t, denoise_t_common)
                fusd_spe_vs = self.intra_attn_vs(denoise_v, denoise_v_common)
                all_fused = torch.cat([fusd_common_as, fusd_spe_ts, fusd_spe_vs], dim=1)
                x = all_fused
            elif self.test_condition == 'atv':
                fusd_spe_ts = self.intra_attn_ts(denoise_t, denoise_t_common)
                fusd_spe_as = self.intra_attn_as(denoise_a, denoise_a_common)
                fusd_spe_vs = self.intra_attn_vs(denoise_v, denoise_v_common)
                all_fused = torch.cat([fusd_spe_as, fusd_spe_ts, fusd_spe_vs], dim=1)
                x = all_fused


        x_a, x_t, x_v = x[:, :seq_len, :], x[:, seq_len:2*seq_len, :], x[:, 2*seq_len:, :]
        x_joint = torch.cat([x_a, x_t, x_v], dim=-1)
        res = x_joint
        u = self.proj2(F.dropout(F.relu(self.proj1(res)), p=self.out_dropout, training=self.training))
        hidden = u + res
        out = self.nlp_head(hidden)
  
        return hidden, out, [outputs_intra_a, outputs_intra_t, outputs_intra_v], [outputs_inter_a, outputs_inter_t, outputs_inter_v], [out_a, out_t, out_v, out_a_common, out_t_common, out_v_common], [a_kls, t_kls, v_kls, a_kls_common, t_kls_common, v_kls_common]
         


if __name__ == '__main__':
    input = [torch.randn(61, 32, 300)]
    model = Denoising(100, 100, 100, 128, 1)
    anchor = torch.randn(32, 61, 128)
    hidden, out, _ = model(input)
