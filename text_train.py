from lib.text_data import MSMarcoDataset
from torch.utils.data import DataLoader
from sentence_transformers import losses, models, SentenceTransformer

from torch import nn, Tensor
from typing import Dict, Any, Iterable
from lib.losses import DCL, DHEL, NT_xent, InfoNCELoss
import torch
import argparse


class RankingLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, loss_func_cls) -> None:
        super().__init__()
        self.model = model
        self.loss_fct = loss_func_cls(temperature=0.05)

    def forward(
        self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor
    ) -> Tensor:
        reps = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])

        return self.loss_fct(embeddings_a, embeddings_b)

    def get_config_dict(self) -> Dict[str, Any]:
        return {}

    def get_name(self) -> str:
        return self.loss_fct.__class__.__name__


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="nreimers/MiniLM-L6-H384-uncased"
    )
    parser.add_argument("--loss", type=str, default="MNRL")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--warmup_epoch", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    model_name = args.model_name
    # Now we create a SentenceTransformer model from scratch
    word_emb = models.Transformer(model_name)
    pooling = models.Pooling(word_emb.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_emb, pooling])

    # For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
    train_dataset = MSMarcoDataset()
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=True,
    )
    train_loss_mnrl = losses.MultipleNegativesRankingLoss(model=model)
    train_loss_dcl = RankingLoss(model=model, loss_func_cls=DCL)
    train_loss_dhel = RankingLoss(model=model, loss_func_cls=DHEL)
    train_loss_ntxent = RankingLoss(model=model, loss_func_cls=NT_xent)
    train_loss_info_nce = RankingLoss(model=model, loss_func_cls=InfoNCELoss)

    if args.loss == "MNRL":
        train_loss = train_loss_mnrl
    elif args.loss == "DCL":
        train_loss = train_loss_dcl
    elif args.loss == "DHEL":
        train_loss = train_loss_dhel
    elif args.loss == "NT_XENT":
        train_loss = train_loss_ntxent
    elif args.loss == "INFO_NCE":
        train_loss = train_loss_info_nce
    else:
        raise ValueError("Invalid loss function")

    # print(f"Train {model_name} with {train_loss.get_name()} loss")
    warmup_steps = int(len(train_dataloader) * args.num_epochs * args.warmup_epoch)

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.num_epochs,
        warmup_steps=warmup_steps,
        use_amp=True,
        optimizer_params={"lr": args.lr},
    )

    dev_dataset = MSMarcoDataset(data_type="dev")
    evaluator = dev_dataset.get_evaluator(args.loss)

    # Save the model
    evaluator(model, output_path=f"text_results")
