from typing import List, Optional, Tuple

import numpy as np
from torch import (
    Tensor,
    arange,
    cat,
    channels_last,
    float32,
    nan_to_num,
    nn,
    rot90,
    zeros,
)
from torchvision.transforms.functional import resize

try:
    from bot.squad_agent.architecture.encoding.entity_encoder import EntityEncoder
    from bot.squad_agent.architecture.encoding.spatial_encoder import SpatialEncoder
# relative import required for training with docker
except ImportError:
    from .entity_encoder import EntityEncoder
    from .spatial_encoder import SpatialEncoder

from torch.nn.functional import relu

SPATIAL_SIZE: List[int] = [120, 120]


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Encoder(nn.Module):
    def __init__(self, device, grid: Optional[np.ndarray], height: int, width: int):
        super(Encoder, self).__init__()
        self.device = device
        self.cropped_cols: Optional[np.ndarray] = None
        self.cropped_rows: Optional[np.ndarray] = None
        self.entity_encoder = EntityEncoder(device)
        self.spatial_encoder = SpatialEncoder()
        self.spatial_encoder.to(memory_format=channels_last)

        self.scalar_fc1 = layer_init(nn.Linear(8, 32))
        self.scalar_fc2 = layer_init(nn.Linear(32, 4))
        self.first_iteration: bool = True
        self.height: int = height
        self.width: int = width
        # not required if learning from saved data
        if grid is not None:
            self._get_cropped_cols_and_rows(grid)

    def forward(
        self, spatial, entity, scalar, locations, process: bool = True
    ) -> Tuple[Tensor, Tensor]:
        entities_scatter, embedded_entity = self.entity_encoder(entity)
        if process:
            scatter_entity = self._scatter_connection(entities_scatter, locations)
            processed_spatial = []
            for grid in spatial:
                processed_spatial.append(self._process_grid(grid))
            processed_spatial = cat(processed_spatial)
            processed_spatial = processed_spatial.unsqueeze(dim=0)
            final_spatial = cat([processed_spatial, scatter_entity], dim=1)

        else:
            final_spatial = spatial

        spatial_embeddings = self.spatial_encoder(final_spatial)
        scalar_embeddings = relu(self.scalar_fc1(scalar))
        scalar_embeddings = relu(self.scalar_fc2(scalar_embeddings))
        embeddings = cat(
            [embedded_entity, spatial_embeddings, scalar_embeddings], dim=1
        )
        return embeddings, final_spatial

    def _process_grid(self, array, divide_by: int = 255):
        array = nan_to_num(array, posinf=0.0)
        # remove borders from the map
        rows = self.cropped_rows
        cols = self.cropped_cols
        array = array[cols[0] : cols[-1] + 1, rows[0] : rows[-1] + 1]

        # cap values to 255
        array[array > 255] = 255
        array = array / divide_by
        # rotate 90 degrees to match minimap features
        array = rot90(array)
        # add a dimension so we can resize the image
        array = array[None, :]
        # resize
        array = resize(array, SPATIAL_SIZE)
        array = array.to(float32)
        return array

    def _scatter_connection(
        self,
        project_embeddings: Tensor,
        entity_locations: Tensor,
        scatter_dim: int = 32,
    ) -> Tensor:
        """
        Integrates non-spatial information (unit attributes) and spatial information
        """
        B, H, W = (entity_locations.shape[0], self.height, self.width)
        entity_num = entity_locations.shape[1]
        index = entity_locations.view(-1, 2).long().to(self.device)
        bias = arange(B).unsqueeze(1).repeat(1, entity_num).view(-1).to(self.device)
        bias *= H * W
        index[:, 0].clamp_(0, W - 1)
        index[:, 1].clamp_(0, H - 1)
        index = (
            index[:, 1] * W + index[:, 0]
        )  # entity_location: (x, y), spatial_info: (y, x)
        index += bias
        index = index.repeat(scatter_dim, 1)
        # flat scatter map and project embeddings
        scatter_map = zeros(scatter_dim, B * H * W, device=self.device)
        project_embeddings = project_embeddings.view(-1, scatter_dim).permute(1, 0)

        scatter_map.scatter_add_(dim=1, index=index, src=project_embeddings)

        scatter_map = scatter_map.reshape(scatter_dim, B, H, W)
        scatter_map = scatter_map.permute(1, 0, 2, 3)
        scatter_map = resize(scatter_map, SPATIAL_SIZE)
        return scatter_map

    def _get_cropped_cols_and_rows(
        self, grid: np.ndarray, threshold: float = 0.0
    ) -> None:
        """
        Crops any edges below or equal to threshold
        Returns cropped image.
        Only called once at start of game
        """
        flatImage = grid
        # ma pads unpathable areas with infinite values, change them to 0
        flatImage = np.where(flatImage == np.inf, 0, flatImage)
        self.cropped_rows = np.where(np.max(flatImage, 0) > threshold)[0]
        self.cropped_cols = np.where(np.max(flatImage, 1) > threshold)[0]
