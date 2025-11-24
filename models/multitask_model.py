import torch
import torch.nn as nn
import torch.nn.functional as F

from clear_multi.unet_parts import *


def compute_scaling_features(magnitude):
    w = -0.6892 + 0.2893 * magnitude
    l = -2.1621 + 0.5493 * magnitude
    s = -2.8512 + 0.8386 * magnitude
    da = -4.3611 + 0.6238 * magnitude
    dm = -3.7393 + 0.6151 * magnitude
    az = -1.3350 + 0.3033 * magnitude
    ax = -2.4664 + 0.5113 * magnitude
    m = magnitude - 5.0
    return torch.stack([w, l, s, da, dm, az, ax, m], dim=1)


class MagnitudePredictor(nn.Module):
    def __init__(self, n_channels=3, pay_bilinear=False):
        super().__init__()
        self.bilinear = pay_bilinear
        self.enc_disp = FiLMEncoderLast(n_channels, bilinear=self.bilinear)
        self.enc_vel = FiLMEncoderLast(n_channels, bilinear=self.bilinear)
        self.fusion = GatingFusion(256)
        self.decoder = GlobalFeatureDecoder(
            in_channels=256,
            hidden_dims=[256, 128, 64],
            use_layernorm=[False, False, True],
        )

    def forward(self, x_disp_row, x_vel_row, loc_disp, loc_vel):
        x5_disp = self.enc_disp(x_disp_row, loc_disp)
        x5_vel = self.enc_vel(x_vel_row, loc_vel)
        x5 = self.fusion(x5_disp, x5_vel)
        magnitude_pred = self.decoder(x5).squeeze(-1)
        return magnitude_pred


class MaxPredictor(nn.Module):
    def __init__(self, n_channels=3, pay_bilinear=False):
        super().__init__()
        self.bilinear = pay_bilinear
        self.enc_disp = FiLMEncoderLast(n_channels, bilinear=self.bilinear)
        self.enc_vel = FiLMEncoderLast(n_channels, bilinear=self.bilinear)
        self.fusion = GatingFusion(256)
        self.tab_mlp = TabularMLP(input_dim=8, output_dim=32)
        self.tab_fusion = SimpleFusion(256, 32)
        self.decoder = GlobalFeatureDecoder(
            in_channels=256,
            hidden_dims=[256, 128, 64, 32],
            use_layernorm=[False, False, True, True],
        )

    def forward(self, x_disp_row, x_vel_row, loc_disp, loc_vel, scaling_features):
        x5_disp = self.enc_disp(x_disp_row, loc_disp)
        x5_vel = self.enc_vel(x_vel_row, loc_vel)
        x5 = self.fusion(x5_disp, x5_vel)
        tab_feat = self.tab_mlp(scaling_features)
        fused_feat = self.tab_fusion(x5, tab_feat)
        max_slip = self.decoder(fused_feat)
        return max_slip
    
    def freeze_all_except_decoder(self):
        for name, param in self.named_parameters():
            if name.startswith("decoder"):
                param.requires_grad = True
            else:
                param.requires_grad = False



class Zone_SlipPredictor(nn.Module):
    def __init__(
        self,
        n_channels=3,
        final_size=(30, 40),
        task_mode="all",
        pay_bilinear=False,
    ):
        super().__init__()
        self.bilinear = pay_bilinear
        self.task_mode = task_mode
        self.enc_disp = FiLMEncoderFull(n_channels, bilinear=self.bilinear)
        self.enc_vel = FiLMEncoderFull(n_channels, bilinear=self.bilinear)
        self.fusion = GatingFusion(256)
        self.cat_4 = CatConvFuse(in_channels=128, out_channels=128)
        self.cat_3 = CatConvFuse(in_channels=64, out_channels=64)
        self.cat_2 = CatConvFuse(in_channels=32, out_channels=32)
        self.cat_1 = CatConvFuse(in_channels=16, out_channels=16)
        self.tab_mlp = TabularMLP(input_dim=8, output_dim=32)
        self.tab_fusion_area = SimpleFusion(256, 32)
        self.tab_fusion_slip = SimpleFusion(256, 32)
        self.decoder_core_area = UNetDecoderCore(256, bilinear=self.bilinear)
        self.decoder_core_slip = UNetDecoderCore(256, bilinear=self.bilinear)
        self.area_head = LearnedUpsample(
            in_channels=16, out_channels=2, final_size=final_size
        )
        self.slip_head = LearnedUpsample(
            in_channels=16, out_channels=1, final_size=final_size
        )
        self.final_size = final_size

    def forward(self, x_disp, x_vel, loc_disp, loc_vel, scaling_features):
        x5_d, x4_d, x3_d, x2_d, x1_d = self.enc_disp(x_disp, loc_disp)
        x5_v, x4_v, x3_v, x2_v, x1_v = self.enc_vel(x_vel, loc_vel)
        x5 = self.fusion(x5_d, x5_v)
        x4 = self.cat_4(x4_d, x4_v)
        x3 = self.cat_3(x3_d, x3_v)
        x2 = self.cat_2(x2_d, x2_v)
        x1 = self.cat_1(x1_d, x1_v)
        tab_feat = self.tab_mlp(scaling_features)
        output = {}
        if self.task_mode in ["slip_area", "all", "slip"]:
            fused_feat_area = self.tab_fusion_area(x5, tab_feat)
            dec_area = self.decoder_core_area(fused_feat_area, x4, x3, x2, x1)
            area_logits = self.area_head(dec_area)
            output["slip_area"] = area_logits
        if self.task_mode in ["slip", "all"]:
            fused_feat_slip = self.tab_fusion_slip(x5, tab_feat)
            dec_slip = self.decoder_core_slip(fused_feat_slip, x4, x3, x2, x1)
            slip_norm = self.slip_head(dec_slip)
            slip_norm = F.relu(slip_norm)
            output["slip_norm"] = slip_norm
        return output


class BuildModel(nn.Module):
    def __init__(
        self,
        params,
        task_mode="all",
        Freeze_m=False,
        Freeze_max=False,
        Freeze_slip=False,
    ):
        super().__init__()
        self.task_mode = task_mode
        self.predict_magnitude = True
        self.predict_max = task_mode in ["max_slip", "all"]
        self.predict_slip = task_mode in ["slip", "slip_area", "all"]
        self.magnitude_predictor = MagnitudePredictor()
        if Freeze_m:
            for param in self.magnitude_predictor.parameters():
                param.requires_grad = False
        if self.predict_max:
            self.max_predictor = MaxPredictor()
            if Freeze_max:
                self.max_predictor.freeze_all_except_decoder()
        if self.predict_slip:
            self.zone_slippredictor = Zone_SlipPredictor(
                n_channels=3,
                final_size=params.get("final_size", (30, 40)),
                task_mode=task_mode,
            )
            if Freeze_slip:
                for param in self.zone_slippredictor.parameters():
                    param.requires_grad = False

    def forward(self,x_disp,x_vel,m_true,max_disp,max_vel,loc_disp,loc_vel,):
        x_disp_row = x_disp * (max_disp.unsqueeze(1).unsqueeze(1))
        x_vel_row = x_vel * (max_vel.unsqueeze(1).unsqueeze(1))
        magnitude_pred = self.magnitude_predictor(
            x_disp_row, x_vel_row, loc_disp, loc_vel
        )
        scaling_features = compute_scaling_features(magnitude_pred)
        output = {}
        if self.predict_magnitude:
            output["magnitude"] = magnitude_pred
        if self.predict_max:
            max_slip = self.max_predictor(
                x_disp_row, x_vel_row, loc_disp, loc_vel, scaling_features
            )
            output["max_slip"] = max_slip
        if self.predict_slip:
            slip_outputs = self.zone_slippredictor(
                x_disp, x_vel, loc_disp, loc_vel, scaling_features
            )
            output.update(slip_outputs)
        if "slip_norm" in output and "max_slip" in output:
            max_val = torch.exp(output["max_slip"].unsqueeze(-1).unsqueeze(-1)) - 1
            output["slip_final"] = output["slip_norm"] * max_val
        return output
