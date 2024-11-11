import numpy as np
import torch
from solo.utils.embedding_propagation import get_similarity_matrix, get_laplacian, get_distance_matrix, get_disparity_matrix

class ManifoldRegularizer():
    def __init__(self, scale_euclidean_distance: bool = False, return_metrics: bool = False):
        self.scale_euclidean_distance = scale_euclidean_distance
        self.return_metrics = return_metrics
        self.last_laplacian_matrix = None
        self.last_similarity_matrix = None

    def manifold_regularizer_loss(self, x: torch.Tensor, y: torch.Tensor, rbf_scale=1.0, fixed_gamma=None):
        weights_matrix, gamma = get_similarity_matrix(x, rbf_scale=rbf_scale, scaling_factor=self.scale_euclidean_distance, fixed_gamma=fixed_gamma)
        laplacian = get_laplacian(weights_matrix, normalized=True)
        metrics = {}
        metrics['gamma'] = gamma
        if self.return_metrics:
            with torch.no_grad():
                sorted_eigvals = torch.linalg.eigvals(laplacian).real.cpu().numpy()
                sorted_eigvals.sort()
                second_smallest_eigenvalue = sorted_eigvals[1]
                spectral_gap = sorted_eigvals[1] - sorted_eigvals[0]
                laplacian_energy = sum(abs(x - 1) for x in sorted_eigvals)

                metrics = {
                    "Second smallest eigenvalue": second_smallest_eigenvalue,
                    "Spectral gap": spectral_gap,
                    "Laplacian energy": laplacian_energy,
                }

                if self.last_laplacian_matrix is not None:
                    laplacian_diff = torch.linalg.norm(
                        laplacian - self.last_laplacian_matrix
                    )
                    metrics["Laplacian difference"] = laplacian_diff

                if self.last_similarity_matrix is not None:
                    similarity_diff = torch.linalg.norm(
                        weights_matrix - self.last_similarity_matrix
                    )
                    metrics["Similarity difference"] = similarity_diff


        regularizer_loss_term = torch.trace(y.T @ laplacian @ y) / (x.shape[0] ** 2)

        self.last_laplacian_matrix = laplacian
        self.last_similarity_matrix = weights_matrix

        return regularizer_loss_term, metrics
    