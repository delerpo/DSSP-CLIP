import torch
import torch.nn as nn
import torch.nn.functional as F


class IDSR(nn.Module):
    """
    IDSR module (paper Eqs. 5–9).

    Uses K projection networks to map the global visual feature into diverse
    queries, then retrieves the most relevant abnormal semantics from a
    candidate text-feature bank via cross-attention.
    """

    def __init__(self, visual_dim: int, text_dim: int, num_queries: int = 4):
        super().__init__()
        self.num_queries = num_queries

        # K projection networks φ_k (Eqs. 5–6)
        self.query_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(visual_dim, visual_dim),
                nn.GELU(),
                nn.Linear(visual_dim, text_dim),
            )
            for _ in range(num_queries)
        ])

        # Key and value projections for candidate text features (Eqs. 7–8)
        self.W_K = nn.Linear(text_dim, text_dim, bias=False)
        self.W_V = nn.Linear(text_dim, text_dim, bias=False)

    def forward(self, F_image: torch.Tensor, T_A: torch.Tensor):
        """
        Args:
            F_image: [B, C]        global visual feature (CLS token)
            T_A:     [B, N_a, C]   candidate abnormal text features
        Returns:
            T_A_s:   [B, K, C]     instance-aware abnormal text features (Eq.9)
        """
        B = F_image.size(0)

        # K projection nets -> K queries  (Eqs. 5–6)
        Q = torch.stack([proj(F_image) for proj in self.query_projs], dim=1)  # [B, K, C]

        # Project candidate text features  (Eqs. 7–8)
        K = self.W_K(T_A)  # [B, N_a, C]
        V = self.W_V(T_A)  # [B, N_a, C]

        # Cross-attention  (Eq.9)
        d_k = Q.size(-1)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, K, N_a]
        T_A_s = torch.matmul(attn_weights, V)           # [B, K, C]

        return T_A_s


def idsr_diversity_loss(T_A_s: torch.Tensor):
    """
    Orthogonal diversity constraint (paper Eq.10).

    Discourages redundant semantics among the K generated text features.
    """
    K = T_A_s.size(1)
    loss = 0.0
    count = 0
    for i in range(K):
        for j in range(K):
            if i != j:
                sim = F.cosine_similarity(T_A_s[:, i], T_A_s[:, j], dim=-1)
                loss += (sim ** 2).sum()
                count += 1
    return loss / max(count, 1)


def idsr_similarity_loss(T_A_s: torch.Tensor, F_image: torch.Tensor, T_A: torch.Tensor):
    """
    Semantic consistency constraint (paper Eq.11).

    Aligns the generated text features with both the current image content
    and the candidate abnormal semantic space.
    """
    K = T_A_s.size(1)
    B = T_A_s.size(0)

    # First term: align generated features with visual content
    loss_vis = 0.0
    for k in range(K):
        sim = F.cosine_similarity(T_A_s[:, k], F_image, dim=-1)
        loss_vis += (sim ** 2).sum()
    loss_vis = loss_vis / (K * B)

    # Second term: regularize within candidate abnormal semantic space
    loss_tex = 0.0
    for k in range(K):
        # For each generated feature, find the most similar candidate
        sim_matrix = F.cosine_similarity(
            T_A_s[:, k:k + 1], T_A, dim=-1
        )  # [B, N_a]
        loss_tex += sim_matrix.max(dim=-1)[0].pow(2).sum()
    loss_tex = loss_tex / (K * B)

    return -(loss_vis + loss_tex)
