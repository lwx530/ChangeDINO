import torch
import torch.nn as nn
import torch.nn.functional as F
class SFHM(nn.Module):
    """
    NSDSAM 中的空频混合模块 (Spatial-Frequency Hybrid Module)
    结合空间域多尺度卷积与频域 FFT，大幅提升缺陷边缘的锐度并抑制噪声。
    """

    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim

        # 预处理：降维降低计算量
        self.reduce = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, kernel_size=1, bias=False),
            nn.Conv2d(in_dim // 2, in_dim, kernel_size=3, padding=1, bias=False)
        )

        # === 空间域分支 (Spatial Branch) ===
        # 将特征分为两组，分别用 3x3 和 5x5 提取多尺度空间特征
        self.sp_conv3 = nn.Sequential(
            nn.Conv2d(in_dim // 2, in_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.sp_conv5 = nn.Sequential(
            nn.Conv2d(in_dim // 2, in_dim // 2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        self.sp_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        # === 频率域分支 (Frequency Branch) ===
        self.freq_enhance = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )

        # 最终融合
        self.final_proj = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # 1. 预处理
        feat = self.reduce(x)

        # 2. 空间域分支
        # Split channel-wise
        f_sp1, f_sp2 = torch.split(feat, self.in_dim // 2, dim=1)
        f_sp1 = self.sp_conv3(f_sp1)
        f_sp2 = self.sp_conv5(f_sp2)
        f_spatial = self.sp_proj(torch.cat([f_sp1, f_sp2], dim=1))

        # 3. 频率域分支 (2D FFT)
        # 转入频域 (实数转复数)
        f_freq = torch.fft.fft2(feat, dim=(-2, -1), norm="ortho")
        # 提取幅度谱进行卷积增强 (频域卷积通常只作用于幅度，也可直接对实部虚部分别处理，这里用幅度+相位恢复)
        amplitude = torch.abs(f_freq)
        phase = torch.angle(f_freq)

        # 增强幅度谱 (强化高频，抑制低频)
        amp_enhanced = self.freq_enhance(amplitude)

        # 恢复复数形式并逆变换回空间域
        f_freq_complex = amp_enhanced * torch.exp(1j * phase)
        f_freq_spatial = torch.fft.ifft2(f_freq_complex, dim=(-2, -1), norm="ortho").real

        # 4. 融合空域与频域特征
        fused = f_spatial + f_freq_spatial
        out = self.final_proj(fused)

        return out + x  # 残差连接