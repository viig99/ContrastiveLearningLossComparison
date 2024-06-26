import torch
from torch.nn import functional as F
from typing import Tuple


@torch.jit.script
def sim(u: torch.Tensor, v: torch.Tensor, temperature: float):
    return F.cosine_similarity(u.unsqueeze(1), v.unsqueeze(0), dim=-1) / temperature


class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature: float):
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
    def __init__(self, temperature: float):
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
        # Positives are uv pairs, so we only need the diagonal elements.

        neg_mask = ~pos_mask
        neg_sim_uv = sim_uv.masked_select(neg_mask).view(B, -1)
        neg_sim_uu = sim_uu.masked_select(neg_mask).view(B, -1)

        # In DCL, the denominator contains only the negative samples from the (anchor, anchor) & (anchor, positive) pairs.
        negative_sim = torch.cat((neg_sim_uv, neg_sim_uu), dim=1)
        neg_loss = torch.logsumexp(negative_sim, dim=-1)

        # Key identity: log(exp(x) = x, log(a/b) = log(a) - log(b)
        # -log ( e^(sim_uv) / (e^(neg_sim_uv) + sum(e^(neg_sim_uu))) = -sim_uv + logsumexp(concat[neg_sim_uv, neg_sim_uu])
        return (-pos_loss + neg_loss).mean()


class DHEL(torch.nn.Module):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(self, u: torch.Tensor, v: torch.Tensor):
        # u: shape (B, D) where B is the batch size and D is the feature dimension
        # v: shape (B, D) where B is the batch size and D is the feature dimension

        B = u.size(0)

        norm_u = F.normalize(u, p=2, dim=-1)
        norm_v = F.normalize(v, p=2, dim=-1)
        sim_uu = torch.mm(norm_u, norm_u.t()) / self.temperature
        # We can also use `sim` here, but since we have already normalized the vectors, it more efficient to use mm instead.

        pos_loss = torch.div(
            (norm_u * norm_v).sum(dim=1), self.temperature
        )  # Since only the positive diagonal elements are required

        # In DHEL, the denominator only contains the negative samples of the (anchor, anchor) pairs.
        neg_mask = ~torch.eye(B, device=u.device, dtype=torch.bool)
        neg_sim_uu = sim_uu.masked_select(neg_mask).view(B, -1)
        neg_loss = torch.logsumexp(neg_sim_uu, dim=-1)

        return (-pos_loss + neg_loss).mean()


class NT_xent(torch.nn.Module):
    def __init__(self, temperature: float):
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

        # In NT_xent, the denominator contains both positive and negative samples from both (anchor, anchor) & (anchor, positive) pairs,
        # only the diagonal elements are excluded.
        neg_mask = ~diag_mask
        neg_sim_zz = sim_zz.masked_select(neg_mask).view(N, -1)
        neg_loss = torch.logsumexp(neg_sim_zz, dim=-1)

        return (-pos_loss + neg_loss).mean()


class DCL_symmetric(torch.nn.Module):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(self, u: torch.Tensor, v: torch.Tensor):
        z = torch.cat((u, v), dim=0)
        N = z.size(0)
        sim_zz = sim(z, z, self.temperature)

        diag_mask = torch.eye(N, device=u.device, dtype=torch.bool)
        # (u, v) and (v, u) along their respective diagonals are considered as positive pairs.
        pos_mask = diag_mask.roll(shifts=N // 2, dims=0) | diag_mask.roll(
            shifts=-N // 2, dims=0
        )
        pos_loss = sim_zz.masked_select(pos_mask)

        # As per the DCL paper, the loss func is same as NT-Xent but the denominator only contains the negative samples.
        # Symmetric version of DCL, so it also contains non-diagonal (v, u), (v, v) pairs in the denominator.
        neg_mask = ~pos_mask & ~diag_mask
        neg_sim_zz = sim_zz.masked_select(neg_mask).view(N, -1)
        neg_loss = torch.logsumexp(neg_sim_zz, dim=-1)

        return (-pos_loss + neg_loss).mean()


class VICReg(torch.nn.Module):
    def __init__(
        self,
        temperature: float = 0,
        sim_loss_weight: float = 25.0,
        var_loss_weight: float = 25.0,
        cov_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.sim_loss_weight = sim_loss_weight
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight

    def forward(self, u: torch.Tensor, v: torch.Tensor):
        sim_loss = self._similarity_loss(u, v)
        var_loss = self._variance_loss(u, v)
        cov_loss = self._covariance_loss(u, v)

        return (
            (self.sim_loss_weight * sim_loss)
            + (self.var_loss_weight * var_loss)
            + (self.cov_loss_weight * cov_loss)
        )

    def _similarity_loss(self, u: torch.Tensor, v: torch.Tensor):
        return F.mse_loss(u, v)

    def _variance_loss(self, u: torch.Tensor, v: torch.Tensor):
        eps = 1e-4
        std_u = torch.sqrt(u.var(dim=0) + eps)
        std_v = torch.sqrt(v.var(dim=0) + eps)
        return torch.mean(F.relu(1 - std_u)) + torch.mean(F.relu(1 - std_v))

    def _covariance_loss(self, u: torch.Tensor, v: torch.Tensor):
        N, D = u.size()
        u = u - u.mean(dim=0)
        v = v - v.mean(dim=0)
        cov_u = (u.T @ u) / (N - 1)
        cov_v = (v.T @ v) / (N - 1)
        mask = ~torch.eye(D, device=u.device, dtype=torch.bool)
        return (cov_u[mask].pow_(2).sum() + cov_v[mask].pow_(2).sum()) / D


class InfoNCELoss_angle(torch.nn.Module):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature
        self.margin = 10 * torch.pi / 180

    def forward(self, u: torch.Tensor, v: torch.Tensor):
        B = u.size(0)
        norm_u = F.normalize(u, p=2, dim=-1)
        norm_v = F.normalize(v, p=2, dim=-1)

        similarity_uv = torch.mm(norm_u, norm_v.t()).clamp(-1, 1)
        theta_uv = torch.acos(similarity_uv).to(dtype=similarity_uv.dtype)
        pos_mask = torch.eye(B, device=u.device, dtype=torch.bool)

        # As per the angle InfoNCE loss, the margin is only added to the positive pairs.
        pos = torch.cos(theta_uv[pos_mask] + self.margin) / self.temperature
        sim_uv_temp = similarity_uv / self.temperature

        # since denominator contains the margin added positives & non margin negatives.
        sim_uv_temp[pos_mask] = pos
        return (-pos + torch.logsumexp(sim_uv_temp, dim=-1)).mean()


def benchmark(
    loss_func, input1, input2, times=10, warmup=5
) -> Tuple[float, torch.Tensor]:
    loss = torch.Tensor([0])
    for _ in range(warmup):
        loss_func(input1, input2)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()  # type: ignore
    for _ in range(times):
        loss = loss_func(input1, input2)
    end_time.record()  # type: ignore

    torch.cuda.synchronize()
    return start_time.elapsed_time(end_time) / times, loss


if __name__ == "__main__":

    torch.random.manual_seed(42)

    # Example usage
    M, d = 100, 128  # Example dimensions
    u = torch.randn(M, d).cuda()
    v = torch.randn(M, d).cuda()
    temperature = 0.05

    infonce = InfoNCELoss(temperature)
    infonce_time, infonce_loss = benchmark(infonce, u, v)
    print(f"InfoNCE Loss: {infonce_loss.item():.4f} in {infonce_time:.2f}ms")

    dhel = DHEL(temperature)
    dhel_time, dhel_loss = benchmark(dhel, u, v)
    print(f"DHEL Loss: {dhel_loss.item():.4f} in {dhel_time:.2f}ms")

    nt_xent = NT_xent(temperature)
    nt_xent_time, nt_xent_loss = benchmark(nt_xent, u, v)
    print(f"NT-Xent Loss: {nt_xent_loss.item():.4f} in {nt_xent_time:.2f}ms")

    dcl = DCL(temperature)
    dcl_time, dcl_loss = benchmark(dcl, u, v)
    print(f"DCL Loss: {dcl_loss.item():.4f} in {dcl_time:.2f}ms")

    dcls = DCL_symmetric(temperature)
    dcls_time, dcls_loss = benchmark(dcls, u, v)
    print(f"DCL Symmetric Loss: {dcls_loss.item():.4f} in {dcls_time:.2f}ms")

    vicreg = VICReg()
    vicreg_time, vicreg_loss = benchmark(vicreg, u, v)
    print(f"VICReg Loss: {vicreg_loss.item():.4f} in {vicreg_time:.2f}ms")

    infonce_angle = InfoNCELoss_angle(temperature)
    infonce_angle_time, infonce_angle_loss = benchmark(infonce_angle, u, v)
    print(
        f"InfoNCE Angle Loss: {infonce_angle_loss.item():.4f} in {infonce_angle_time:.2f}ms"
    )
