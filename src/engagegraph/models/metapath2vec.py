"""MetaPath2Vec on the user--app engagement graph (Dong et al., KDD 2017).

The graph is a weighted bipartite network collapsed to binary edges for random
walks: a user is connected to every app they used in January.  The metapath
``app -> user -> app`` captures co-usage context; a symmetric path is used in
the sensitivity analysis.  This is the canonical metapath2vec objective, not a
re-implementation of the original notebook's (buggy) cosine read-out.

The optional dependency on PyTorch Geometric is imported lazily so that the
core package and API remain lightweight.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from scipy import sparse

from engagegraph.data.preprocess import Dataset
from engagegraph.models.base import BaseRecommender, Explanation


class MetaPath2VecRecommender(BaseRecommender):
    name = "metapath2vec"

    def __init__(
        self,
        embedding_dim: int = 64,
        walk_length: int = 20,
        context_size: int = 5,
        walks_per_node: int = 5,
        num_negative_samples: int = 5,
        epochs: int = 3,
        batch_size: int = 128,
        learning_rate: float = 0.01,
        metapath: str = "app-user-app",
        scoring: str = "profile",
        seed: int = 42,
    ) -> None:
        if embedding_dim <= 0 or walk_length <= 1 or epochs <= 0:
            raise ValueError("invalid metapath2vec hyper-parameters")
        if metapath not in {"app-user-app", "user-app-user"}:
            raise ValueError("metapath must be 'app-user-app' or 'user-app-user'")
        if scoring not in {"profile", "dot"}:
            raise ValueError("scoring must be 'profile' or 'dot'")
        super().__init__(
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            metapath=metapath,
            scoring=scoring,
            seed=seed,
        )
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.metapath = metapath
        self.scoring = scoring
        self.seed = seed
        self.user_embeddings = np.zeros((0, embedding_dim), dtype=np.float32)
        self.item_embeddings = np.zeros((0, embedding_dim), dtype=np.float32)
        self.training_history: list[dict[str, float]] = []

    def _edge_types(
        self,
    ) -> tuple[tuple[str, str, str], tuple[str, str, str], list[tuple[str, str, str]]]:
        if self.metapath == "app-user-app":
            forward = ("user", "uses", "app")
            backward = ("app", "uses", "user")
            path = [backward, forward]
        else:
            forward = ("user", "uses", "app")
            backward = ("app", "uses", "user")
            path = [forward, backward]
        return forward, backward, path

    def fit(self, dataset: Dataset) -> MetaPath2VecRecommender:
        try:
            import torch
            from torch_geometric.data import HeteroData
            from torch_geometric.nn import MetaPath2Vec
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError(
                "torch and torch-geometric are required for metapath2vec: "
                "pip install 'engagegraph[graph]'"
            ) from exc

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        # Random walks are CPU-bound in PyG; forcing CPU keeps results stable
        # across CUDA/MPS machines (the embedding table is small anyway).
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        device = torch.device("cpu")

        self.n_users, self.n_items = dataset.train.shape
        self.item_popularity = np.asarray(dataset.train.getnnz(axis=0)).ravel().astype(np.float32)

        coo = dataset.train.tocoo()
        user_edges = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
        item_edges = torch.as_tensor(np.vstack([coo.col, coo.row]), dtype=torch.long)

        data = HeteroData()
        data["user"].num_nodes = self.n_users
        data["app"].num_nodes = self.n_items
        data["user", "uses", "app"].edge_index = user_edges
        data["app", "uses", "user"].edge_index = item_edges

        path = self._edge_types()[2]
        model = MetaPath2Vec(
            data.edge_index_dict,
            embedding_dim=self.embedding_dim,
            metapath=path,
            walk_length=self.walk_length,
            context_size=self.context_size,
            walks_per_node=self.walks_per_node,
            num_negative_samples=self.num_negative_samples,
            num_nodes_dict={"user": self.n_users, "app": self.n_items},
            sparse=False,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loader = model.loader(batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.training_history = []
        model.train()
        for epoch in range(1, self.epochs + 1):
            total = 0.0
            batches = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total += float(loss.detach().cpu())
                batches += 1
            mean_loss = total / max(batches, 1)
            self.training_history.append({"epoch": float(epoch), "loss": mean_loss})

        self.user_embeddings = model("user").detach().cpu().numpy().astype(np.float32)
        self.item_embeddings = model("app").detach().cpu().numpy().astype(np.float32)
        return self

    def score_users(
        self,
        user_indices: np.ndarray,
        candidate_items: np.ndarray | None = None,
        train: sparse.csr_matrix | None = None,
    ) -> np.ndarray:
        if self.scoring == "dot":
            scores = self.user_embeddings[user_indices] @ self.item_embeddings.T
        else:
            if train is None:
                raise ValueError("profile scoring requires the training matrix")
            # A user profile is the engagement-weighted mean of the metapath2vec
            # item embeddings in their January history.  This is consistently
            # stronger than the jointly learned user vector for top-K ranking,
            # and it degrades gracefully for users with short histories.
            items = self.item_embeddings.astype(np.float32, copy=False)
            item_norms = np.linalg.norm(items, axis=1, keepdims=True)
            item_norms[item_norms == 0] = 1.0
            normalized = items / item_norms
            profiles = train[user_indices] @ normalized
            profile_norms = np.linalg.norm(profiles, axis=1, keepdims=True)
            profile_norms[profile_norms == 0] = 1.0
            scores = (profiles / profile_norms) @ normalized.T
        return self._as_candidate_scores(np.asarray(scores, dtype=np.float32), candidate_items)

    def similar_items(self, item_index: int, k: int = 10) -> list[tuple[int, float]]:
        if self.item_embeddings.size == 0:
            return []
        norms = np.linalg.norm(self.item_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = self.item_embeddings / norms
        scores = normalized @ normalized[item_index]
        order = np.argsort(-scores)
        return [(int(i), float(scores[i])) for i in order if i != item_index][:k]

    def explain(
        self, user_index: int, item_index: int, dataset: Dataset, k: int = 3
    ) -> list[Explanation]:
        return super().explain(user_index, item_index, dataset, k)

    def get_state(self) -> dict[str, np.ndarray]:
        return {
            "user_embeddings": self.user_embeddings,
            "item_embeddings": self.item_embeddings,
            "item_popularity": self.item_popularity,
        }

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        self.user_embeddings = state["user_embeddings"].astype(np.float32)
        self.item_embeddings = state["item_embeddings"].astype(np.float32)
        self.n_users, self.embedding_dim = self.user_embeddings.shape
        self.n_items = int(self.item_embeddings.shape[0])
        self.item_popularity = state.get(
            "item_popularity", np.zeros(self.n_items, dtype=np.float32)
        ).astype(np.float32)

    def meta(self) -> dict[str, Any]:
        return {
            "model_type": self.name,
            "params": self.params,
            "n_users": int(self.n_users),
            "n_items": int(self.n_items),
            "embedding_dim": int(self.embedding_dim),
            "training_history": self.training_history,
        }
