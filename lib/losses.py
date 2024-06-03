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


class DCL(torch.nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, u: torch.Tensor, v: torch.Tensor):
        # u: shape (B, D) where B is the batch size and D is the feature dimension
        # v: shape (B, D) where B is the batch size and D is the feature dimension

        B, _ = u.shape
        sim_uv = sim(u, v, self.temperature)
        sim_uu = sim(u, u, self.temperature)

        pos_mask = torch.eye(B, device=u.device, dtype=torch.bool)
        pos_loss = -sim_uv.masked_select(pos_mask)

        neg_mask = ~pos_mask
        neg_sim_uv = sim_uv.masked_select(neg_mask).view(B, -1)
        neg_sim_uu = sim_uu.masked_select(neg_mask).view(B, -1)
        negative_sim = torch.cat((neg_sim_uv, neg_sim_uu), dim=1)
        neg_loss = torch.logsumexp(negative_sim, dim=-1)

        return (pos_loss + neg_loss).mean()


class DHEL(torch.nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, u: torch.Tensor, v: torch.Tensor):
        # u: shape (B, D) where B is the batch size and D is the feature dimension
        # v: shape (B, D) where B is the batch size and D is the feature dimension

        B, _ = u.shape
        sim_uv = sim(u, v, self.temperature)
        sim_uu = sim(u, u, self.temperature)

        pos_mask = torch.eye(B, device=u.device, dtype=torch.bool)
        pos_loss = -sim_uv.masked_select(pos_mask)

        neg_sim_uu = -sim_uu[~pos_mask].view(B, -1)
        neg_loss = torch.logsumexp(neg_sim_uu, dim=-1)

        return (pos_loss + neg_loss).mean()


if __name__ == "__main__":

    torch.random.manual_seed(42)

    # Example usage
    M, d = 100, 128  # Example dimensions
    u = torch.randn(M, d)
    v = torch.randn(M, d)
    temperature = 0.1

    dcl = DCL(temperature)
    dcl_loss = dcl(u, v)
    print("DCL Loss:", dcl_loss.item())

    infonce = InfoNCELoss(temperature)
    infonce_loss = infonce(u, v)
    print("InfoNCE Loss:", infonce_loss.item())

    dhel = DHEL(temperature)
    dhel_loss = dhel(u, v)
    print("DHEL Loss:", dhel_loss.item())
