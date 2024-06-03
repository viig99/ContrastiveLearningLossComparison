import torch


class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, anchors: torch.Tensor, positives: torch.Tensor):
        # anchors: shape (B, D) where M is the batch size and D is the feature dimension
        # positives: shape (B, D) where M is the batch size and D is the feature dimension

        # Compute cosine similarity between all pairs
        similarity_matrix = (
            self.cosine_similarity(anchors.unsqueeze(1), positives.unsqueeze(0))
            / self.temperature
        )

        # Create labels for cross entropy loss
        labels = torch.arange(anchors.shape[0], device=anchors.device)

        # Compute the loss
        loss = self.cross_entropy_loss(similarity_matrix, labels)

        return loss
