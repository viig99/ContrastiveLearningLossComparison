from collections import defaultdict
import random
import os
import tarfile
from sentence_transformers import util, InputExample
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import Dataset


class MSMarcoDataset(Dataset):
    def __init__(
        self,
        data_folder="./datasets",
        data_type: str = "train",
    ):
        super().__init__()
        self.data_folder = data_folder
        self.data_type = data_type
        self.corpus = {}
        self.queries = {}
        self.train_queries = defaultdict(list)
        self._load_dataset()
        self._cleanup_extra()
        self.queries_ids = list(self.train_queries.keys())

    def _load_dataset(self):
        collection_filepath = os.path.join(self.data_folder, "collection.tsv")
        queries_filepath = os.path.join(
            self.data_folder, f"queries.{self.data_type}.tsv"
        )
        qrel_path = os.path.join(self.data_folder, f"qrels.{self.data_type}.tsv")

        if (
            not os.path.exists(collection_filepath)
            or not os.path.exists(queries_filepath)
            or not os.path.exists(qrel_path)
        ):
            tar_filepath = os.path.join(self.data_folder, "collectionandqueries.tar.gz")
            if not os.path.exists(tar_filepath):
                print("Download collection, queries and qrels")
                util.http_get(
                    "https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz",
                    tar_filepath,
                )

                util.http_get(
                    "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv",
                    os.path.join(self.data_folder, "qrels.dev.tsv"),
                )

            with tarfile.open(tar_filepath, "r:gz") as tar:
                tar.extractall(path=self.data_folder)

            os.remove(tar_filepath)

        print(f"Reading qrels: qrels.{self.data_type}.tsv")
        with open(qrel_path, "r", encoding="utf8") as fIn:
            for line in fIn:
                qid, _, pid, _ = line.strip().split("\t")
                self.train_queries[qid].append(pid)

        qids_with_relevant_passages = set(self.train_queries.keys())

        pids_with_relevant_queries = set()
        for qid in qids_with_relevant_passages:
            pids_with_relevant_queries.update(self.train_queries[qid])

        print("Reading corpus: collection.tsv")
        with open(collection_filepath, "r", encoding="utf8") as fIn:
            for line in fIn:
                pid, passage = line.strip().split("\t")
                if not passage or pid not in pids_with_relevant_queries:
                    continue
                self.corpus[pid] = passage

        print(f"Reading queries: queries.{self.data_type}.tsv")
        with open(queries_filepath, "r", encoding="utf8") as fIn:
            for line in fIn:
                qid, query = line.strip().split("\t")
                if not query or qid not in qids_with_relevant_passages:
                    continue
                self.queries[qid] = query

    def _cleanup_extra(self):
        qids_present = set(self.queries.keys())
        train_qids = set(self.train_queries.keys())
        qids_to_remove = train_qids.difference(qids_present)

        for qid in qids_to_remove:
            del self.train_queries[qid]

    def __getitem__(self, item):
        q_id = self.queries_ids[item]
        query_text = self.queries[q_id]

        rel_corpus_items = self.train_queries[q_id]
        pos_id = random.choice(rel_corpus_items)
        pos_text = self.corpus[pos_id]
        return InputExample(texts=[query_text, pos_text])

    def __len__(self):
        return len(self.train_queries)

    def get_evaluator(self, name="msmarco-dev"):
        relevant_docs = {}
        for qid in self.train_queries:
            relevant_docs[qid] = set(self.train_queries[qid])

        return InformationRetrievalEvaluator(
            self.queries,
            self.corpus,
            relevant_docs,
            show_progress_bar=True,
            corpus_chunk_size=10000,
            precision_recall_at_k=[10, 100],
            name=name,
        )


if __name__ == "__main__":

    from tqdm.auto import tqdm

    msmarco_dataset = MSMarcoDataset(data_type="train")
    print("Number of queries:", len(msmarco_dataset))

    for data in tqdm(msmarco_dataset):
        continue
