import numpy as np
from torch import Tensor, arange, mean, nn, sum, tensor
from torch.nn.functional import relu


def layer_init(layer, std: float = np.sqrt(2), bias_const: float = 0.0) -> None:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class EntityEncoder(nn.Module):
    def __init__(self, device):
        super(EntityEncoder, self).__init__()
        self.device = device
        self.embed = layer_init(nn.Linear(408, 128))
        self.tf_layer = nn.TransformerEncoderLayer(128, 2, batch_first=True)
        self.tf = nn.TransformerEncoder(self.tf_layer, 4, enable_nested_tensor=False)
        self.conv1 = nn.Conv1d(128, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc1 = layer_init(nn.Linear(128, 32))
        self.embed_fc = layer_init(nn.Linear(128, 32))

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        tmp_x = mean(x, dim=2, keepdim=False)
        tmp_y = tmp_x != 0
        entity_num = sum(tmp_y, dim=1, keepdim=False)
        entity_num_numpy = np.minimum(256 - 2, entity_num.cpu().numpy())
        entity_num = tensor(
            entity_num_numpy, dtype=entity_num.dtype, device=self.device
        )
        mask = arange(0, 256).float()
        mask = mask.repeat(batch_size, 1)
        mask = mask.to(self.device)

        # mask: [batch_size, max_entities]
        mask = mask > entity_num.unsqueeze(dim=1)

        # mask_seq_len = mask.shape[-1]
        # tran_mask = mask.unsqueeze(1)
        # tran_mask: [batch_seq_size x max_entities x max_entities]
        # tran_mask = tran_mask.repeat(1, mask_seq_len, 1)
        x = self.embed(x)
        # had less issues here not passing in the batch dimension, so shapes are:
        # Turns out, the batch dimension is greater than 1 during learning phase of PPO
        # x[0]: (MAX_ENTITY, NUM_FEATURES)
        # tran_mask[0]: (SEQUENCE_LEN, SEQUENCE_LEN) where MAX_ENTITY == SEQUENCE_LEN
        # TODO: Having issues / getting confused with transformer masks
        #   But are actually optional in the pytorch transformer implementation
        #   So left them out for now. Need to research this properly
        out = self.tf(x, src_key_padding_mask=mask)
        entities_scatter = relu(self.fc1(out))
        # out is wrong size
        masked_out = out * mask.unsqueeze(dim=2)
        z = masked_out.sum(dim=1, keepdim=False)
        z = z / entity_num.unsqueeze(dim=1)
        embedded_entity = relu(self.embed_fc(z))

        return entities_scatter, embedded_entity
