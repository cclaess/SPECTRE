import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR


def UPA(hr_volume, lr_volume, device="cuda", use_amp=True):
    """
    hr_volume: numpy or torch [C,Hh,Wh,Dh]
    lr_volume: torch [1,C,Hl,Wl,Dl]
    """

    hr = torch.as_tensor(hr_volume).unsqueeze(0).float().to(device)
    lr = lr_volume.to(device)

    _, _, Hh, Wh, Dh = hr.shape
    _, _, Hl, Wl, Dl = lr.shape
    scale = Hh // Hl
    assert Wh // Wl == scale and Dh // Dl == scale, "Inconsistent scale factors"

    model = LearnablePixelwiseAnisoJBU3D(
        Hl, Wl, Dl, scale=scale
    ).to(device)

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-1)
    max_steps = 1000
    gamma = (1e-9 / 1e-1) ** (1.0 / max_steps)
    scheduler = LambdaLR(opt, lr_lambda=lambda step: gamma ** step)
    scaler = torch.amp.GradScaler(device=device, enabled=use_amp)

    for step in range(max_steps):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device, enabled=use_amp):
            pred = model(lr, hr)
            loss = F.l1_loss(pred, hr)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        if step % 50 == 0:
            print(f"step {step}: loss={loss.item():.5f}")

    model.eval()
    with torch.inference_mode(), \
        torch.amp.autocast(device_type=device, enabled=use_amp, dtype=torch.float16):
        out = model(lr, hr)
 
    return out


@torch.no_grad()
def _build_offsets_3d(R_max: int, device):
    offs = torch.arange(-R_max, R_max + 1, device=device)
    dX, dY, dZ = torch.meshgrid(offs, offs, offs, indexing="ij")
    return (
        dX.reshape(-1),
        dY.reshape(-1),
        dZ.reshape(-1),
    )  # [K]


def gather_lr_scalar_3d(map_lr, Ui, Vi, Wi):
    """
    map_lr: [1,1,Hl,Wl,Dl] or [Hl,Wl,Dl]
    Ui,Vi,Wi: [Bn,Hh,Wh,Dh]
    """
    Hl, Wl, Dl = map_lr.shape[-3:]
    flat = Hl * Wl * Dl
    idx = (Ui * Wl * Dl + Vi * Dl + Wi).reshape(-1)
    t = map_lr.view(flat)
    vals = t.index_select(0, idx)
    return vals.view(Ui.shape)


def gs_jbu_aniso_noparent_3d(
    feat_lr,          # [1,C,Hl,Wl,Dl]
    guide_hr,         # [1,G,Hh,Wh,Dh]
    scale,
    sigma_x_map,
    sigma_y_map,
    sigma_z_map,
    sigma_r_map,
    R_max=3,
    alpha_dyn=2.0,
    C_chunk=64,
    Nn_chunk=125,
):
    _, C, Hl, Wl, Dl = feat_lr.shape
    _, _, Hh, Wh, Dh = guide_hr.shape
    device = feat_lr.device
    dtype_feat = feat_lr.dtype

    # HR grid
    x = torch.arange(Hh, device=device, dtype=torch.float32)
    y = torch.arange(Wh, device=device, dtype=torch.float32)
    z = torch.arange(Dh, device=device, dtype=torch.float32)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")

    u = (X + 0.5) / scale - 0.5
    v = (Y + 0.5) / scale - 0.5
    w = (Z + 0.5) / scale - 0.5

    uc = torch.round(u).clamp(0, Hl - 1).long()
    vc = torch.round(v).clamp(0, Wl - 1).long()
    wc = torch.round(w).clamp(0, Dl - 1).long()

    # Dynamic radius
    sigma_eff = torch.maximum(
        sigma_x_map,
        torch.maximum(sigma_y_map, sigma_z_map),
    )
    sigma_eff_hr = F.interpolate(
        sigma_eff, (Hh, Wh, Dh), mode="trilinear", align_corners=False
    )
    # sigma_eff_hr = sigma_eff_hr.squeeze(0).squeeze(0)
    R_map = torch.ceil(alpha_dyn * sigma_eff_hr).clamp(1, R_max).long()

    dX_all, dY_all, dZ_all = _build_offsets_3d(R_max, device)

    num = torch.zeros(C, Hh, Wh, Dh, device=device, dtype=torch.float32)
    den = torch.zeros(Hh, Wh, Dh, device=device, dtype=torch.float32)
    m = torch.full((Hh, Wh, Dh), -1e9, device=device, dtype=torch.float32)

    feat_flat = feat_lr[0].permute(1, 2, 3, 0).reshape(-1, C)
    guide_lr = F.interpolate(
        guide_hr, (Hl, Wl, Dl), mode="trilinear", align_corners=False
    )

    for n0 in range(0, len(dX_all), Nn_chunk):
        dX = dX_all[n0:n0+Nn_chunk][:, None, None, None]
        dY = dY_all[n0:n0+Nn_chunk][:, None, None, None]
        dZ = dZ_all[n0:n0+Nn_chunk][:, None, None, None]

        Ui = torch.clamp(uc.unsqueeze(0) + dX, 0, Hl - 1)
        Vi = torch.clamp(vc.unsqueeze(0) + dY, 0, Wl - 1)
        Wi = torch.clamp(wc.unsqueeze(0) + dZ, 0, Dl - 1)

        # mask = (dX**2 + dY**2 + dZ**2 <= R_map[None, ...] ** 2)
        mask = (dX**2 + dY**2 + dZ**2 <= R_map**2).squeeze(0).squeeze(0)

        cx = (Ui.float() + 0.5) * scale - 0.5
        cy = (Vi.float() + 0.5) * scale - 0.5
        cz = (Wi.float() + 0.5) * scale - 0.5

        dx = X.unsqueeze(0) - cx
        dy = Y.unsqueeze(0) - cy
        dz = Z.unsqueeze(0) - cz

        sx = gather_lr_scalar_3d(sigma_x_map, Ui, Vi, Wi).clamp_min(1e-6)
        sy = gather_lr_scalar_3d(sigma_y_map, Ui, Vi, Wi).clamp_min(1e-6)
        sz = gather_lr_scalar_3d(sigma_z_map, Ui, Vi, Wi).clamp_min(1e-6)
        sr = gather_lr_scalar_3d(sigma_r_map, Ui, Vi, Wi).clamp_min(1e-6)

        log_ws = (
            -(dx**2)/(2*sx**2)
            -(dy**2)/(2*sy**2)
            -(dz**2)/(2*sz**2)
        )

        diff2 = 0.0
        for g in range(guide_hr.shape[1]):
            g0 = gather_lr_scalar_3d(guide_lr[0, g], Ui, Vi, Wi)
            diff2 += (guide_hr[0, g] - g0) ** 2

        log_wr = -diff2 / (2 * sr**2 + 1e-8)
        log_w = torch.where(mask, log_ws + log_wr, -1e9)

        m_chunk = log_w.max(dim=0).values
        m_new = torch.maximum(m, m_chunk)

        scale_old = torch.exp(m - m_new)
        num *= scale_old
        den *= scale_old

        w = torch.exp(log_w - m_new)
        den += w.sum(0)

        idx_flat = (Ui * Wl * Dl + Vi * Dl + Wi).reshape(-1)

        for c0 in range(0, C, C_chunk):
            c1 = min(c0 + C_chunk, C)
            f = feat_flat.index_select(0, idx_flat)[:, c0:c1]
            f = f.view(w.shape + (c1 - c0,))
            num[c0:c1] += (f * w[..., None]).sum(0).permute(3, 0, 1, 2)

        m = m_new

    out = (num / den.clamp_min(1e-8)).unsqueeze(0)
    return out.to(dtype_feat)


class LearnablePixelwiseAnisoJBU3D(nn.Module):
    def __init__(
        self,
        Hl,
        Wl,
        Dl,
        scale,
        init_sigma=1.5,
        init_sigma_r=0.1,
        R_max=3,
        alpha_dyn=2.0,
    ):
        super().__init__()
        self.scale = scale
        self.R_max = R_max
        self.alpha_dyn = alpha_dyn

        self.sx_raw = nn.Parameter(torch.full((1,1,Hl,Wl,Dl), math.log(init_sigma)))
        self.sy_raw = nn.Parameter(torch.full((1,1,Hl,Wl,Dl), math.log(init_sigma)))
        self.sz_raw = nn.Parameter(torch.full((1,1,Hl,Wl,Dl), math.log(init_sigma)))
        self.sr_raw = nn.Parameter(torch.full((1,1,Hl,Wl,Dl), math.log(init_sigma_r)))

    def forward(self, feat_lr, guide_hr):
        return gs_jbu_aniso_noparent_3d(
            feat_lr,
            guide_hr,
            self.scale,
            torch.exp(self.sx_raw),
            torch.exp(self.sy_raw),
            torch.exp(self.sz_raw),
            torch.exp(self.sr_raw),
            R_max=self.R_max,
            alpha_dyn=self.alpha_dyn,
        )


if __name__ == "__main__":
    import argparse

    import numpy as np
    import nibabel as nib
    import monai.transforms as transforms

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_amp", action="store_true")
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.LoadImaged(keys=("image", "mask")),
        transforms.EnsureChannelFirstd(keys=("image", "mask"), channel_dim="no_channel"),
        transforms.ScaleIntensityRanged(
            keys=("image",),
            a_min=-1000,
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        transforms.Orientationd(keys=("image", "mask"), axcodes="RAS"),
        transforms.RandWeightedCropd(
            keys=("image", "mask"), 
            w_key="mask", 
            spatial_size=(128, 128, 64), 
            num_samples=1,
        ),
        transforms.CopyItemsd(keys=("mask"), times=1, names=("mask_low_res")),
        transforms.Resized(keys=("mask_low_res"), spatial_size=(16, 16, 8), mode="nearest")
    ])
    sample = transform({
        "image": args.image_path,
        "mask": args.mask_path,
    })[0]

    mask_out = UPA(
        sample["image"],
        sample["mask_low_res"].unsqueeze(0),
        device=args.device,
        use_amp=args.use_amp,
    )

    nib.save(
        nib.Nifti1Image(
            (sample["image"] * 255).squeeze(0).cpu().numpy().astype(np.uint8),
            affine=np.eye(4),
        ),
        "image.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(
            sample["mask"].squeeze(0).cpu().numpy().astype(np.uint8),
            affine=np.eye(4),
        ),
        "mask.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(
            mask_out.squeeze(0).cpu().numpy().astype(np.uint8),
            affine=np.eye(4),
        ),
        "upsampled_mask.nii.gz",
    )
    
