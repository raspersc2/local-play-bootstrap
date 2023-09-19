from typing import Optional, Union

import numpy as np
from torch import Tensor, cat, flatten, nn
from torch.distributions import Categorical

# relative import required for training with docker
try:
    from bot.squad_agent.architecture.encoding import Encoder
except ImportError:
    from ...architecture.encoding.encoder import Encoder


def layer_init(
    layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(
        self,
        action_space_size: int,
        device,
        grid: Optional[np.ndarray],
        height: int,
        width: int,
    ):
        super().__init__()
        self.shared_layers = Encoder(device, grid, height, width)

        self.lstm = nn.LSTM(292, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.policy_layers = layer_init(nn.Linear(128, action_space_size), std=0.01)
        self.value_layers = layer_init(nn.Linear(128, 1), std=1)

    def get_states(
        self,
        spatial: Tensor,
        entity: Tensor,
        scalar: Tensor,
        locations: Tensor,
        lstm_state: tuple[Tensor, Tensor],
        done: Tensor,
        process_spatial: bool = True,
    ) -> tuple:
        hidden, processed_spatial = self.shared_layers(
            spatial, entity, scalar, locations, process_spatial
        )

        # LSTM logic
        batch_size: int = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden: Union[list, Tensor] = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = flatten(cat(new_hidden), 0, 1)
        return new_hidden, lstm_state, processed_spatial

    def get_value(
        self,
        spatial: Tensor,
        entity: Tensor,
        scalar: Tensor,
        locations: Tensor,
        lstm_state: tuple[Tensor, Tensor],
        done: Tensor,
        process_spatial: bool,
    ) -> float:
        hidden, _, _ = self.get_states(
            spatial, entity, scalar, locations, lstm_state, done, process_spatial
        )
        return self.value_layers(hidden)

    def get_action_and_value(
        self,
        spatial: Tensor,
        entity: Tensor,
        scalar: Tensor,
        locations: Tensor,
        lstm_state: tuple[Tensor, Tensor],
        done: Tensor,
        action: Tensor = None,
        process_spatial: bool = True,
    ) -> tuple:
        hidden, lstm_state, processed_spatial = self.get_states(
            spatial, entity, scalar, locations, lstm_state, done, process_spatial
        )
        logits: Tensor = self.policy_layers(hidden)
        probs: Categorical = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.value_layers(hidden),
            lstm_state,
            processed_spatial,
        )

    def forward(
        self, spatial: Tensor, entity: Tensor, scalar: Tensor
    ) -> tuple[Tensor, float]:
        z, processed_spatial = self.shared_layers(spatial, entity, scalar)
        policy_logits = self.policy_layers(z)
        value = self.value_layers(z)
        return policy_logits, value
