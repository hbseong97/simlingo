from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from simlingo_training.utils.custom_types import DrivingExample


def cross_track_error(points: Tensor, path: Tensor):
    """
    Computes the cross track error between a set of points and a path.

    Args:
        points: The set of points to compute the cross track error for with shape [b, n, 2].
        path: The path to compute the cross track error with with shape [b, m, 2]. The path
            can contain nan values which indicates that the path is not available for that position.

    Returns:
        The cross track error for each point in the set of points with shape [b, n].
    """

    points, path = points.float(), path.float()

    ind = torch.arange(path.size(0), device=path.device)[:, None]
    closest = torch.cdist(points, path).nan_to_num_(torch.inf).argmin(-1)
    pt0 = path[ind, (closest - 1).clamp_min(0)]
    pt1 = path[ind, closest]
    pt2 = path[ind, (closest + 1).clamp_max(path.size(1) - 1)]

    tangent = (pt2 - pt1).nan_to_num_(0.0) + (pt1 - pt0).nan_to_num_(0.0)
    normal = torch.stack((tangent[..., 1], -tangent[..., 0]), dim=-1)
    normal = normal / normal.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-2)

    return (points - pt1).mul(normal).sum(-1).abs()

class NormZeroOne(nn.Module):
    def __init__(self, min_max: Tuple[float, float]):
        super().__init__()
        self.register_buffer("min_max", torch.tensor(min_max, dtype=torch.float), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Normalise tensor to [0, 1] using values from min_max"""
        return (x - self.min_max[0]) / (self.min_max[1] - self.min_max[0])
    
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 0, size_average: bool = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target.view(-1, 1)).view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class WaypointInputAdaptor(nn.Module):
    """
    Takes an input of shape [B, N, 2] and returns an output of shape [B, N, token_size]
    Args:
        token_size: feature dimension of output tensor.
        hidden_size: hidden dimension used in Linear layers under the hood.
        norm_layer: the `Module` to use to normalize the values of the input tensor.
    """
    
    def __init__(
        self, token_size: int = 258, hidden_size: int = 64, hidden_size2: int = 128, norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm_layer = norm_layer

        self.mlp = nn.Sequential(nn.Linear(2, hidden_size), nn.ReLU(True), nn.Linear(hidden_size, hidden_size2), nn.ReLU(True), nn.Linear(hidden_size2, token_size))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input with dims [B, N, 2]

        Returns:
            Output with dims [B, N, token_size]
        """
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        x = self.mlp(x)
        return x


class DrivingAdaptor(nn.Module):
    def __init__(self, 
                hidden_size: int, 
                mlp_dim=256, 
                predict_route_as_wps=False, 
                speed_wps_mode=False,
            ):
        super().__init__()
        self.heads = {}
        self.order = []

        self.speed_wps_mode = speed_wps_mode
        self.predict_route_as_wps = predict_route_as_wps

        if predict_route_as_wps:
            self.future_waypoints = 20
            self.query_embeds_wps = nn.Parameter(0.02 * torch.randn((1, self.future_waypoints, hidden_size)))
            self.route_head = nn.Sequential(
                nn.Linear(hidden_size, mlp_dim*2), nn.SiLU(True),nn.Linear(mlp_dim*2, mlp_dim), nn.SiLU(True), nn.Linear(mlp_dim, 2, bias=False)
            )
            
            self.queries = {'route': self.query_embeds_wps}
            self.sizes = {'route': self.future_waypoints}
            self.heads["route"] = self.route_head
            self.order.append('route')

        if speed_wps_mode == '2d':
            dim = 2
        elif speed_wps_mode == '1d':
            dim = 1
        else:
            raise ValueError(f"speed_wps_mode must be '1d' or '2d', not {speed_wps_mode}")
        self.future_speed_waypoints = 10 #TODO: read from config
        self.query_embeds_speed = nn.Parameter(0.02 * torch.randn((1, self.future_speed_waypoints, hidden_size)))
        self.speed_wps_head = nn.Sequential(
                nn.Linear(hidden_size, mlp_dim), nn.SiLU(True), nn.Linear(mlp_dim, dim, bias=False)
            )
        self.heads["speed_wps"] = self.speed_wps_head
        self.queries['speed_wps'] = self.query_embeds_speed
        self.sizes['speed_wps'] = self.future_speed_waypoints
        self.order.append('speed_wps')


    def forward(self, 
            driving_example: DrivingExample,
            **kwargs
            ) -> Dict[str, Tensor]:

        try:
            driving_input = driving_example.driving_input
        except AttributeError:
            driving_input = driving_example
        
        b = driving_input.camera_images.shape[0]
        inputs = None

        for input_type in self.order:
            query_embed = self.queries[input_type]
            if inputs is None:
                inputs = query_embed.expand(b, -1, -1)
            else:
                inputs = torch.cat((inputs, query_embed.expand(b, -1, -1)), dim=1)

        inputs_mask = torch.ones_like(inputs[:, :, 0], dtype=torch.bool)

        return {"inputs": inputs, "inputs_mask": inputs_mask}

    def get_predictions(
        self, 
        features: Tensor,
        logits: Optional[Tensor] = None
    ) -> Dict:

        current_index = 0
        predictions = {}
        for i, input_type in enumerate(self.order):
            size = self.sizes[input_type]

            feature = features[:, current_index: current_index + size]
            prediction = self.heads[input_type](feature).cumsum(1)

            predictions[input_type] = prediction
            current_index += size
        
        return predictions


    def compute_loss(
        self, adaptor_features: Tensor, adaptor_logits: Tensor, _inputs: Dict[str, Tensor], example: DrivingExample
    ) -> Dict[str, Tuple[Tensor, Tensor]]:
        label = example.driving_label
        assert label is not None
        
        if self.predict_route_as_wps:
            label_route = label.path
        else:
            label_route = None

        if self.speed_wps_mode == '2d':
            label_speed_wps = label.waypoints[:, : self.future_waypoints + 1]
        elif self.speed_wps_mode == '1d':
            label_speed_wps = label.waypoints_1d
        else:
            label_speed_wps = None

        current_index = 0
        loss_dict = {}
        for i, input_type in enumerate(self.order):
            size = self.sizes[input_type]
            features_tmp = adaptor_features[:, current_index: current_index + size]
            label = locals()[f'label_{input_type}']

            prediction = self.heads[input_type](features_tmp).cumsum(1)
            loss = F.smooth_l1_loss(prediction, label, reduction="none").sum(-1)
            
            # if input_type == 'waypoints' and self.predict_route_as_wps:
            #     # compute cross track error
            #     cte = cross_track_error(prediction, label_waypoints)
            #     loss_dict[f"{input_type}_cte_loss"] = (cte, torch.ones_like(cte, dtype=torch.long))

            loss_dict[f"{input_type}_loss"] = (loss, torch.ones_like(loss, dtype=torch.long))
            loss_dict[f"{input_type}_prediction"] = prediction
            loss_dict[f"{input_type}_label"] = label
            current_index += size

        return loss_dict


class LanguageAdaptor(nn.Module):
    def __init__(self, language_model):
        super().__init__()
        self.embed_tokens = language_model.model.embed_tokens
        self.lm_head = None

        # Check for different lm_head configurations
        if hasattr(language_model.model, "lm_head"):
            self.lm_head = language_model.model.lm_head
        elif hasattr(language_model.model, "embed_out"):
            self.lm_head = language_model.model.embed_out
        elif hasattr(language_model.model.base_model.model, 'output'):
            self.lm_head = language_model.model.base_model.model.output
        else:
            # Get the actual model (handle PEFT wrapper if present)
            actual_model = language_model.model
            if hasattr(actual_model, 'base_model'):
                # This is a PEFT model, get the base model
                # For InternVL + PEFT: actual_model is PeftModel, base_model is Qwen2ForCausalLM
                peft_base_model = actual_model.base_model
                # The Qwen2ForCausalLM has both .model (Qwen2Model) and .lm_head
                if hasattr(peft_base_model, 'model'):
                    qwen2_model = peft_base_model.model  # This is the Qwen2Model with embed_tokens, layers, etc.
                else:
                    qwen2_model = peft_base_model
            else:
                peft_base_model = actual_model
                qwen2_model = actual_model

            # Check if the model uses tied embeddings (like Qwen2 models)
            # ref: https://skyzh.github.io/tiny-llm/week1-05-qwen2-model.html#task-3-implement-qwen2modelweek1
            # According to the documentation: if tie_word_embeddings is True, use Embedding::as_linear
            # Otherwise, there should be a separate lm_head layer
            config = None
            if hasattr(qwen2_model, 'config'):
                config = qwen2_model.config
            elif hasattr(peft_base_model, 'config'):
                config = peft_base_model.config
            elif hasattr(actual_model, 'config'):
                config = actual_model.config

            if config and hasattr(config, 'tie_word_embeddings'):
                if config.tie_word_embeddings:
                    # For models like Qwen2-0.5b that use tied embeddings
                    self.lm_head = None  # Will use embed_tokens.weight as linear layer
                else:
                    # For models like Qwen2-7b that should have a separate lm_head
                    # Try to find the lm_head in different possible locations
                    # The lm_head should be at the same level as the Qwen2Model (in the Qwen2ForCausalLM)
                    if hasattr(peft_base_model, "lm_head"):
                        self.lm_head = peft_base_model.lm_head
                    elif hasattr(actual_model, "lm_head"):
                        self.lm_head = actual_model.lm_head
                    elif hasattr(qwen2_model, "lm_head"):
                        self.lm_head = qwen2_model.lm_head
                    elif hasattr(peft_base_model, "_modules") and "lm_head" in peft_base_model._modules:
                        self.lm_head = peft_base_model._modules["lm_head"]
                    elif hasattr(actual_model, "_modules") and "lm_head" in actual_model._modules:
                        self.lm_head = actual_model._modules["lm_head"]
                    elif hasattr(qwen2_model, "_modules") and "lm_head" in qwen2_model._modules:
                        self.lm_head = qwen2_model._modules["lm_head"]
                    else:
                        # Special case: InternVL extracts only Qwen2Model (not Qwen2ForCausalLM)
                        # Even though config says tie_word_embeddings=False, there's no lm_head available
                        # So we fall back to using tied embeddings (embed_tokens.weight)
                        print("Warning: Model config indicates separate lm_head (tie_word_embeddings=False) "
                              "but no lm_head found. Falling back to tied embeddings.")
                        self.lm_head = None  # Will use embed_tokens.weight as linear layer
            elif hasattr(language_model.model, "embed_tokens"):
                # Fallback: assume tied embeddings if embed_tokens exists but no lm_head found
                self.lm_head = None  # Will use embed_tokens.weight as linear layer
            else:
                raise ValueError(
                    f"Language model must have `lm_head`, `embed_out`, or `embed_tokens` attribute. "
                    f"Qwen2Model attributes: {list(qwen2_model.__dict__.keys())}, "
                    f"PEFT base model attributes: {list(peft_base_model.__dict__.keys())}, "
                    f"Actual model attributes: {list(actual_model.__dict__.keys())}"
                )


    def forward(self, example: DrivingExample, inference=False, **kwargs) -> Dict[str, Tensor]:
        try:
            driving_input = example.driving_input
        except AttributeError:
            driving_input = example
            
        b = driving_input.camera_images.size(0)
        
        if inference:
            label = driving_input.prompt_inference
        else:
            label = driving_input.prompt
            
        if label is not None:
            ids = label.phrase_ids.long()
            ids_valid = label.phrase_valid  # true => is fed into model
            ids_mask = label.loss_masking # true => takes part in loss

        inputs = self.embed_tokens(ids.clamp(min=0, max=self.embed_tokens.num_embeddings - 1))
        return {"inputs": inputs, "inputs_mask": ids_valid, "_ids": ids, "_ids_mask": ids_mask}

    def compute_loss(
        self, adaptor_features: Tensor, adaptor_logits: Tensor, inputs: Dict[str, Tensor], example: DrivingExample
    ) -> Dict[str, Tuple[Tensor, Tensor]]:
        del example

        print(f"DEBUG COMPUTE_LOSS: adaptor_logits is None: {adaptor_logits is None}")
        if adaptor_logits is not None:
            print(f"DEBUG COMPUTE_LOSS: adaptor_logits shape: {adaptor_logits.shape}")
            print(f"DEBUG COMPUTE_LOSS: adaptor_logits passed in from outside!")

        if adaptor_logits is None:
            print(f"DEBUG BRANCH: lm_head is None: {self.lm_head is None}")
            print(f"DEBUG BRANCH: lm_head type: {type(self.lm_head)}")
            if self.lm_head is not None:
                print(f"DEBUG LM_HEAD: Using lm_head for output projection")
                print(f"DEBUG LM_HEAD: lm_head type: {type(self.lm_head)}")
                if hasattr(self.lm_head, 'weight'):
                    print(f"DEBUG LM_HEAD: lm_head.weight shape: {self.lm_head.weight.shape}")
                adaptor_logits = self.lm_head(adaptor_features[:, :-1])
                print(f"DEBUG LM_HEAD: adaptor_logits shape: {adaptor_logits.shape}")
            else:
                # Use embed_tokens.weight as output projection (weight tying)
                print(f"DEBUG F.LINEAR: Using embed_tokens.weight for output projection")
                features_input = adaptor_features[:, :-1]
                print(f"DEBUG F.LINEAR: features_input shape: {features_input.shape}")
                print(f"DEBUG F.LINEAR: embed_tokens.weight shape: {self.embed_tokens.weight.shape}")
                print(f"DEBUG F.LINEAR: embed_tokens.weight id: {id(self.embed_tokens.weight)}")

                adaptor_logits = F.linear(features_input, self.embed_tokens.weight)
                print(f"DEBUG F.LINEAR: adaptor_logits shape: {adaptor_logits.shape}")
                print(f"DEBUG F.LINEAR: Expected shape should be: [{features_input.shape[0]}, {features_input.shape[1]}, {self.embed_tokens.weight.shape[0]}]")
        else:
            adaptor_logits = adaptor_logits[:, :-1]
        labels = torch.where(inputs["_ids_mask"], inputs["_ids"], -1)
        # Shift by 1 for next token prediction
        labels = labels[:, 1:]

        # DEBUG: Print debug info before cross-entropy loss
        print(f"DEBUG: adaptor_logits shape: {adaptor_logits.shape}")
        print(f"DEBUG: labels shape: {labels.shape}")
        print(f"DEBUG: labels max: {labels.max().item()}")
        print(f"DEBUG: labels min: {labels.min().item()}")
        print(f"DEBUG: vocab size (logits): {adaptor_logits.shape[-1]}")
        print(f"DEBUG: embed_tokens size: {self.embed_tokens.num_embeddings}")
        print(f"DEBUG: embed_tokens.weight shape: {self.embed_tokens.weight.shape}")
        print(f"DEBUG: embed_tokens.weight id: {id(self.embed_tokens.weight)}")

        # Check for out-of-bounds labels
        valid_labels = labels[labels != -1]
        if len(valid_labels) > 0:
            out_of_bounds = valid_labels >= adaptor_logits.shape[-1]
            if out_of_bounds.any():
                oob_labels = valid_labels[out_of_bounds].unique()
                print(f"DEBUG: ⚠️  OUT-OF-BOUNDS LABELS: {oob_labels.tolist()}")
                print(f"DEBUG: Max valid label: {adaptor_logits.shape[-1] - 1}")

                # Also check against embed_tokens size
                out_of_bounds_embed = valid_labels >= self.embed_tokens.num_embeddings
                if out_of_bounds_embed.any():
                    oob_embed_labels = valid_labels[out_of_bounds_embed].unique()
                    print(f"DEBUG: ⚠️  OUT-OF-BOUNDS vs EMBED_TOKENS: {oob_embed_labels.tolist()}")
                else:
                    print(f"DEBUG: ✅ All labels are within embed_tokens bounds")

        language_loss = F.cross_entropy(
            adaptor_logits.flatten(0, -2), labels.flatten(), ignore_index=-1, reduction="none"
        ).view_as(labels)
        return {"language_loss": (language_loss, labels.ne(-1))}

class AdaptorList(nn.Module):
    """
    Each adaptor is responsible for converting a driving example
    to a sequence of tokens and computing the loss on the token outputs.
    Adaptors are only used during training.
    """

    def __init__(
        self,
        driving: Optional[DrivingAdaptor] = None,
        language: Optional[LanguageAdaptor] = None,
    ):
        super().__init__()
        self.driving = driving
        self.language = language

    @property
    def adaptors(self):
        dct: Dict[str, Adaptor] = {}
        if self.language is not None:
            dct["language"] = self.language
        if self.driving is not None:
            dct["driving"] = self.driving
        return dct

    def forward(self, example: DrivingExample, **kwargs) -> Dict[str, Tensor]:
        """
        Construct input embeddings for the given driving example.
        """

        input_dict: Dict[str, Tensor] = {}
        inputs_list: List[Tensor] = []
        inputs_mask_list: List[Tensor] = []

        for key, adaptor in self.adaptors.items():
            adaptor_input_dict = adaptor.forward(example, **kwargs)
            inputs_list.append(adaptor_input_dict["inputs"])
            inputs_mask_list.append(adaptor_input_dict["inputs_mask"])
            input_dict.update({key + "_" + k: v for k, v in adaptor_input_dict.items()})

        inputs = torch.cat(inputs_list, dim=1)
        inputs_mask = torch.cat(inputs_mask_list, dim=1)
        split_sizes = torch.as_tensor([x.size(1) for x in inputs_list])
        arange = torch.arange(inputs.size(0), device=inputs.device)[:, None]

        # Apply random permutation of modalities during training
        rand_perm = torch.arange(inputs.size(1), device=inputs.device).expand(inputs.size(0), -1)
        # Apply permutation to move invalid tokens to end of sequence
        valid_perm = inputs_mask[arange, rand_perm].byte().argsort(dim=-1, descending=True, stable=True)
        perm = rand_perm.gather(1, valid_perm)

        input_dict["inputs"] = inputs[arange, perm]
        input_dict["inputs_mask"] = inputs_mask[arange, perm]
        input_dict["perm"] = perm
        input_dict["split_sizes"] = split_sizes
        return input_dict

    def compute_loss(
        self, features: Tensor, logits: Tensor, input_dict: Dict[str, Tensor], example: DrivingExample
    ) -> Dict[str, Tuple[Tensor, Tensor]]:
        """
        Distributes the output embeddings from the transformer to
        the correct loss function and returns a dictionary of losses.
        """

        features_by_adaptor = self.split_outputs_by_adaptor(input_dict, features)
        logits_by_adaptor = self.split_outputs_by_adaptor(input_dict, logits)

        loss_dict: Dict[str, Tuple[Tensor, Tensor]] = {}

        # Compute loss in each adaptor
        loss_dict: Dict[str, Tuple[Tensor, Tensor]] = {}
        for key, adaptor in self.adaptors.items():
            adaptor_input_dict = _gather_from_dict(input_dict, key + "_")
            adaptor_features = features_by_adaptor[key]
            adaptor_logits = logits_by_adaptor[key]
            losses = adaptor.compute_loss(adaptor_features, adaptor_logits, adaptor_input_dict, example)
            loss_dict.update(losses)

        return loss_dict

    def split_outputs_by_adaptor(self, input_dict: Dict[str, Tensor], outputs: Tensor) -> Dict[str, Tensor]:
        """
        Splits the output tensor into the correct output for each adaptor, according to the
        split_sizes in the input_dict.
        """
        # First reverse permutation
        inv_perm = input_dict["perm"].argsort(-1)
        arange = torch.arange(inv_perm.size(0), device=inv_perm.device)[:, None]
        outputs = outputs[arange, inv_perm]

        # Now split output for each adaptor
        split_sizes = [int(x) for x in input_dict["split_sizes"]]
        outputs_list = list(outputs.split(split_sizes, dim=1))
        return {key: outputs_list[i] for i, key in enumerate(self.adaptors.keys())}


def _gather_from_dict(d: Dict[str, Tensor], prefix: str):
    out: Dict[str, Tensor] = {}  # dict comprehensions with if not supported
    for k, v in d.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
    return out