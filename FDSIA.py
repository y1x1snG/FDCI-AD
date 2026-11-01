import torch
import torch.nn as nn


class FDSIA_Attention(nn.Module):
    """Full-Dimensional Sequential Interaction Attention (FDSIA, Section 3.1).

    Realizes the BOTM (Block Operation on Tensor in Matrix style) / BOTV
    (Block Operation on Tensor in Vector style) formalism of GAM through the
    equivalent permute / reshape / MLP operations, so behaviour stays
    numerically identical to the released GAM-style implementation.

    Pipeline (paper notation):
        X  in R^{B*C*H*W} : input
        B  in R^{B*C*H*W} : channel-refined feature after the MLP (Eq. 1-8)
        M1 in R^{B*C*H*W} : channel attention weight            (Eq. 9-13)
        F  = M1 (⊙) X                                            (Eq. 14)
        G  = S(F) (⊗) F,  S = spatial attention sub-module        (Eq. 15)
    """

    def __init__(self, in_channels, rate=4):
        super().__init__()
        # Channel attention: Z = MLP(A') -- two-layer MLP (encoder-decoder, ratio r).  (Eq. 7)
        self.channel_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // rate),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // rate, in_channels),
        )
        # Spatial attention sub-module S (Eq. 15). Keeps the 7x7 conv block of the
        # released code so the reported metrics are preserved.
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // rate, kernel_size=7, padding=3),
            nn.BatchNorm2d(in_channels // rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // rate, in_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(in_channels),
        )

    @staticmethod
    def _botm(x):
        """BOTM (matrix style): (B,C,H,W) -> (B, H*W, C).  Eq. (1)/(11)."""
        b, c, h, w = x.shape
        return x.permute(0, 2, 3, 1).reshape(b, h * w, c)

    def forward(self, x):
        b, c, h, w = x.shape

        # --- Channel branch: X -> B -> M1  (Eq. 1-13) ---
        # Eq. (1)-(6): D1=BOTM(X); D=Perm(D1); D_r=BOTV(D); Y=Perm(D_r);
        #              Y_i=BOTV(Y); A=Reshape(Y_i)  ->  A = _botm(X)
        a = self._botm(x)                       # A, (B, H*W, C)
        # Eq. (7): Z = MLP(A')
        z = self.channel_mlp(a)                # Z, (B, H*W, C)
        # Eq. (8): B = Reshape(Z')  -> (B, C, H, W)
        bb = z.permute(0, 2, 1).reshape(b, c, h, w)   # B
        # Eq. (9)-(12): B1=BOTV(B); K=Perm(B1); K1=BOTM(K); E=Perm(K1) -> E=_botm(B)
        e = self._botm(bb)                      # E, (B, H*W, C)
        # Eq. (13): M1 = Sigmoid(E)
        m1 = e.permute(0, 2, 1).reshape(b, c, h, w).sigmoid()   # M1

        # Eq. (14): F = M1 (⊙) X
        f = x * m1

        # --- Spatial branch: G = S(F) (⊗) F  (Eq. 15) ---
        s = self.spatial_attention(f).sigmoid()  # S(F)
        g = f * s                                # G
        return g
