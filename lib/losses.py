import torch
from torch.nn import functional as F


@torch.jit.script
def sim(u: torch.Tensor, v: torch.Tensor, temperature: float):
    return F.cosine_similarity(u.unsqueeze(1), v.unsqueeze(0), dim=-1) / temperature


class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, u: torch.Tensor, v: torch.Tensor):
        # u: shape (B, D) where M is the batch size and D is the feature dimension
        # v: shape (B, D) where M is the batch size and D is the feature dimension

        # Compute cosine similarity between all pairs
        similarity_matrix = sim(u, v, self.temperature)

        # Create labels for cross entropy loss
        labels = torch.arange(u.shape[0], device=u.device)

        # Compute the loss
        loss = self.cross_entropy_loss(similarity_matrix, labels)
        return loss


class DCL_paper(torch.nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, u: torch.Tensor, v: torch.Tensor):
        # u: shape (B, D) where B is the batch size and D is the feature dimension
        # v: shape (B, D) where B is the batch size and D is the feature dimension

        B = u.size(0)
        sim_uv = sim(u, v, self.temperature)
        sim_uu = sim(u, u, self.temperature)

        pos_mask = torch.eye(B, device=u.device, dtype=torch.bool)
        pos_loss = sim_uv.masked_select(pos_mask)

        neg_mask = ~pos_mask
        neg_sim_uv = sim_uv.masked_select(neg_mask).view(B, -1)
        neg_sim_uu = sim_uu.masked_select(neg_mask).view(B, -1)
        negative_sim = torch.cat((neg_sim_uv, neg_sim_uu), dim=1)
        neg_loss = torch.logsumexp(negative_sim, dim=-1)

        return (-pos_loss + neg_loss).mean()


class DHEL(torch.nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, u: torch.Tensor, v: torch.Tensor):
        # u: shape (B, D) where B is the batch size and D is the feature dimension
        # v: shape (B, D) where B is the batch size and D is the feature dimension

        B = u.size(0)
        sim_uv = sim(u, v, self.temperature)
        sim_uu = sim(u, u, self.temperature)

        pos_mask = torch.eye(B, device=u.device, dtype=torch.bool)
        pos_loss = sim_uv.masked_select(pos_mask)

        # In DHEL, the denominator only contains the negative samples of the anchors
        neg_sim_uu = sim_uu[~pos_mask].view(B, -1)
        neg_loss = torch.logsumexp(neg_sim_uu, dim=-1)

        return (-pos_loss + neg_loss).mean()


class NT_xent(torch.nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, u: torch.Tensor, v: torch.Tensor):
        z = torch.cat((u, v), dim=0)
        N = z.size(0)
        sim_zz = sim(z, z, self.temperature)

        diag_mask = torch.eye(N, device=u.device, dtype=torch.bool)
        pos_mask = diag_mask.roll(shifts=N // 2, dims=0) | diag_mask.roll(
            shifts=-N // 2, dims=0
        )
        pos_loss = sim_zz.masked_select(pos_mask)

        # In NT_xent, the denominator contains both positive and negative samples.
        neg_mask = ~diag_mask
        neg_sim_zz = sim_zz.masked_select(neg_mask).view(N, -1)
        neg_loss = torch.logsumexp(neg_sim_zz, dim=-1)

        return (-pos_loss + neg_loss).mean()


class DCL(torch.nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, u: torch.Tensor, v: torch.Tensor):
        z = torch.cat((u, v), dim=0)
        N = z.size(0)
        sim_zz = sim(z, z, self.temperature)

        diag_mask = torch.eye(N, device=u.device, dtype=torch.bool)
        pos_mask = diag_mask.roll(shifts=N // 2, dims=0) | diag_mask.roll(
            shifts=-N // 2, dims=0
        )
        pos_loss = sim_zz.masked_select(pos_mask)

        # As per the DCL paper, the loss func is same as NT-Xent but the denominator only contains the negative samples.
        neg_mask = ~pos_mask & ~diag_mask
        neg_sim_zz = sim_zz.masked_select(neg_mask).view(N, -1)
        neg_loss = torch.logsumexp(neg_sim_zz, dim=-1)

        return (-pos_loss + neg_loss).mean()


if __name__ == "__main__":

    torch.random.manual_seed(42)

    # Example usage
    M, d = 100, 128  # Example dimensions
    u = torch.randn(M, d)
    v = torch.randn(M, d)
    temperature = 0.1

    infonce = InfoNCELoss(temperature)
    infonce_loss = infonce(u, v)
    print("InfoNCE Loss:", infonce_loss.item())

    dhel = DHEL(temperature)
    dhel_loss = dhel(u, v)
    print("DHEL Loss:", dhel_loss.item())

    nt_xent = NT_xent(temperature)
    nt_xent_loss = nt_xent(u, v)
    print("NT-Xent Loss:", nt_xent_loss.item())

    dcl = DCL(temperature)
    dcl_loss = dcl(u, v)
    print("DCL Loss:", dcl_loss.item())

    dclp = DCL_paper(temperature)
    dclp_loss = dclp(u, v)
    print("DCL Imlp Paper Loss:", dclp_loss.item())
