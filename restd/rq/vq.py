import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import sinkhorn_algorithm


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        n_e,
        e_dim,
        mu=0.25,
        beta=1,
        kmeans_init=False,
        kmeans_iters=10,
        sk_epsilon=0.01,
        sk_iters=100,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.mu = mu
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):

        centers, _ = self.constrained_km_faiss(data, self.n_e)
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    def constrained_km_faiss(self, data, n_clusters=10):
        import faiss

        x = data.cpu().detach().numpy().astype("float32")

        kmeans = faiss.Kmeans(
            d=x.shape[1],
            k=n_clusters,
            niter=self.kmeans_iters,
            nredo=10,
            verbose=False,
            seed=42,
        )

        kmeans.train(x)

        _, labels = kmeans.index.search(x, 1)
        labels = labels.flatten()

        t_centers = torch.from_numpy(kmeans.centroids)
        t_labels = torch.from_numpy(labels).tolist()

        return t_centers, t_labels

    @staticmethod
    def center_distance_for_constraint(distances):
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def vq_init(self, x, use_sk=True):
        latent = x.view(-1, self.e_dim)

        if not self.initted:
            self.init_emb(latent)

        _distance_flag = "distance"

        if _distance_flag == "distance":
            d = (
                torch.sum(latent**2, dim=1, keepdim=True)
                + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
                - 2 * torch.matmul(latent, self.embedding.weight.t())
            )
        else:
            d = latent @ self.embedding.weight.t()

        if not use_sk or self.sk_epsilon <= 0:
            if _distance_flag == "distance":
                indices = torch.argmin(d, dim=-1)
            else:
                indices = torch.argmax(d, dim=-1)
        else:
            d = self.center_distance_for_constraint(d)
            d = d.double()

            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        return x_q

    def forward(self, x, label, idx, use_sk=True):
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        _distance_flag = "distance"

        if _distance_flag == "distance":
            d = (
                torch.sum(latent**2, dim=1, keepdim=True)
                + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
                - 2 * torch.matmul(latent, self.embedding.weight.t())
            )
        else:
            d = latent @ self.embedding.weight.t()
        if not use_sk or self.sk_epsilon <= 0:
            if _distance_flag == "distance":
                if idx != -1:
                    indices = torch.argmin(d, dim=-1)
                else:
                    temp = 1.0
                    prob_dist = F.softmax(-d / temp, dim=1)
                    indices = torch.multinomial(prob_dist, 1).squeeze()
            else:
                indices = torch.argmax(d, dim=-1)
        else:
            d = self.center_distance_for_constraint(d)
            d = d.double()

            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())

        loss = codebook_loss + self.mu * commitment_loss

        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices
