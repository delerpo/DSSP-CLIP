from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
from clip.adaptor import Adaptor
from clip.idsr import IDSR, idsr_diversity_loss, idsr_similarity_loss
from clip.prompt_bank import gen_prompt_status

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        if dim_out is None:
            dim_out = dim
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask, average_attn_weights=False)

    def forward(self, x: torch.Tensor):
        attn_out, attn_map = self.attention(self.ln_1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, attn_map


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        print("resblocks: ", len(self.resblocks))
        
    def forward(self, x: torch.Tensor, fearure_layers=None, visual_prompt=None, cspf_start=3, cspf_end=13):
        out = []
        # Aggregate attention maps from shallow layers
        # cspf_end is exclusive: the accumulated range is [cspf_start, cspf_end)
        attn_map_sum = None
        prefix_len = len(visual_prompt) if visual_prompt is not None else 0  # deep learnable prompt tuning
        for i in range(len(self.resblocks)):
            if i < prefix_len:
                p_len = visual_prompt[i:i + 1].size(1)
                x = torch.cat([visual_prompt[i:i+1].repeat(x.size(0), 1, 1), x], dim=1)  # prepend visual prompt
            else:
                p_len = 0

            x, attn_map = self.resblocks[i](x)

            # Accumulate attention maps from shallow layers
            if cspf_start <= i < cspf_end:
                current_map = attn_map[..., p_len:, p_len:]
                if attn_map_sum is None:
                    attn_map_sum = current_map
                else:
                    attn_map_sum = attn_map_sum + current_map
            # ---------------------------------------------

            if i < prefix_len:
                x = x[:, p_len:]     # remove visual prompt tokens
            if fearure_layers is not None and i+1 in fearure_layers:
                out.append(x)
        if fearure_layers is None:  # text transformer path
            return x
        else:
            # Normalize aggregated attention map by the number of accumulated layers
            return out, attn_map_sum / (cspf_end - cspf_start)

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        print(self.positional_embedding.size())

    
    def forward(self, x: torch.Tensor, feature_layers=[24], visual_prompt=None, cspf_start=3, cspf_end=13):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        side = int((self.positional_embedding.shape[0] - 1) ** 0.5)
        new_side = int((x.shape[1] - 1) ** 0.5)

        # update the position embedding during inference for varied input size
        if side != new_side:
            new_pos = self.positional_embedding[1:, :].reshape(-1, side, side, x.shape[-1]).permute(0, 3, 1, 2)
            new_pos = torch.nn.functional.interpolate(new_pos, (new_side, new_side), mode='bilinear')
            new_pos = new_pos.reshape(x.shape[-1], new_side * new_side).transpose(0, 1)
            self.positional_embedding.data = torch.cat([self.positional_embedding[:1, :], new_pos], 0)
            
        x = x + self.positional_embedding.to(x.dtype)
         
        if visual_prompt is not None:
            x = torch.cat([x, visual_prompt[:1].repeat(x.size(0), 1, 1)], dim=1)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        out, attn_maps_sum = self.transformer(x, feature_layers, cspf_start=cspf_start, cspf_end=cspf_end)
        for i, o in enumerate(out):
            out[i] = o.permute(1, 0, 2)
            if visual_prompt is not None:
                out[i] = out[i][:, :-visual_prompt.size(1), :]
        return out, attn_maps_sum


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype
    
    def _encode_prompt_batch(self, prompt_texts):
        """Encode a list of text prompts through the text encoder, returning [N, C] features."""
        tokens = self.tokenizer(prompt_texts).to(self.device)
        token_emb = self.token_embedding(tokens).type(self.dtype)
        token_emb = token_emb + self.positional_embedding[:token_emb.size(1)].type(self.dtype)
        token_emb = token_emb.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(token_emb)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)] @ self.text_projection
        return x  # [N, C]

    def _build_prompt_bank(self, args, dataset_list):
        """Build pre-encoded prompt bank from fine-grained attribute descriptions."""
        prompt_normal_dict, prompt_abnormal_dict, cls_map = gen_prompt_status(
            dataset_list=dataset_list
        )
        self.cls_map = cls_map
        self.prompt_bank = {}

        for category in prompt_normal_dict:
            normal_texts = prompt_normal_dict[category]
            abnormal_texts = prompt_abnormal_dict[category]
            with torch.no_grad():
                T_N = self._encode_prompt_batch(normal_texts)    # [N_n, C]
                T_A = self._encode_prompt_batch(abnormal_texts)  # [N_a, C]
            self.prompt_bank[category] = {
                "T_N": T_N,
                "T_A": T_A,
            }

    def insert(self, args, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

        self.adaptor =  Adaptor(inplanes=self.visual.proj.shape[0], outplanes=self.visual.proj.shape[0]).to(device)
        # Learnable fusion coefficient for hybrid attention
        self.ibha_beta = nn.Parameter(torch.tensor(0.3, device=device))
        # Value projection and LayerNorm for feature reconstruction
        self.ibha_v_proj = nn.Linear(self.visual.proj.shape[0], self.visual.proj.shape[0], device=device)
        self.ibha_ln = nn.LayerNorm(self.visual.proj.shape[0], device=device)
        # Shallow-layer range for attention map aggregation
        self.cspf_start = args.cspf_start
        self.cspf_end = args.cspf_end

        # ---- IDSR: Instance-aware Dynamic Semantic Reasoning ----
        # Build prompt bank for all supported datasets
        self._build_prompt_bank(args, dataset_list=['mvtec', 'visa', 'mpdd', 'btad', 'dagm', 'dtd', 'sdd'])
        visual_dim = self.visual.proj.shape[0]
        text_dim = self.text_projection.shape[0]
        self.idsr = IDSR(visual_dim=visual_dim, text_dim=text_dim,
                         num_queries=getattr(args, 'idsr_num_queries', 4)).to(device)

    def get_idsr_parameters(self):
        return list(self.idsr.parameters())

    def encode_state_prompt_idsr(self, F_image, categories):
        """
        IDSR-based state prompt encoding.
        Returns:
            T_N:     [B, C]        mean normal text feature
            T_A_s:   [B, K, C]     instance-aware abnormal text features
        """
        B = F_image.size(0)
        unique_cats = list(set(categories))
        T_N_all = torch.zeros(B, self.text_projection.shape[0]).to(F_image.device)
        T_A_s_all = torch.zeros(B, self.idsr.num_queries, self.text_projection.shape[0]).to(F_image.device)

        for cat in unique_cats:
            idx = [i for i, c in enumerate(categories) if c == cat]
            if cat not in self.prompt_bank:
                # Fallback for categories without fine-grained prompts
                T_N_all[idx] = T_N_cat_default if hasattr(self, '_T_N_default') else torch.zeros_like(T_N_all[idx])
                T_A_s_all[idx] = T_A_s_default if hasattr(self, '_T_A_s_default') else torch.zeros_like(T_A_s_all[idx])
                continue
            bank = self.prompt_bank[cat]
            T_N_cat = bank["T_N"].float()  # [N_n, C]
            T_A_cat = bank["T_A"].float().unsqueeze(0).expand(len(idx), -1, -1)  # [B_cat, N_a, C]

            F_img_cat = F_image[idx]  # [B_cat, C]
            T_A_s_cat = self.idsr(F_img_cat, T_A_cat)  # [B_cat, K, C]

            T_N_all[idx] = T_N_cat.mean(dim=0)  # [C]
            T_A_s_all[idx] = T_A_s_cat

        return T_N_all, T_A_s_all

    # Legacy: fixed-prompt encoding (kept for backward compatibility)
    def encode_state_prompt(self):
        normal_t = self.tokenizer(['without defect.']).to(self.device)
        abnormal_t = self.tokenizer(['with defect.']).to(self.device)
        tokens = torch.cat([normal_t, abnormal_t], dim=0)
        state_x = self.token_embedding(tokens).type(self.dtype)
        state_x = state_x + self.positional_embedding[:state_x.size(1)].type(self.dtype)
        state_x = state_x.permute(1, 0, 2)  # NLD -> LND
        state_x = self.transformer(state_x)
        state_x = state_x.permute(1, 0, 2)  # LND -> NLD
        state_x = self.ln_final(state_x).type(self.dtype)
        state_x = state_x[torch.arange(state_x.shape[0]), tokens.argmax(dim=-1)] @ self.text_projection
        return state_x

    
    def get_trainable_parameters(self):
        params = [self.ibha_beta, self.ibha_v_proj.weight, self.ibha_v_proj.bias]
        params += list(self.adaptor.parameters())
        params += self.get_idsr_parameters()
        return params

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x
    
    def encode_image(self, image, feature_layers=None, cspf_start=None, cspf_end=None):
        if cspf_start is not None and cspf_end is not None:
            return self.visual(image.type(self.dtype), feature_layers, cspf_start=cspf_start, cspf_end=cspf_end)
        return self.visual(image.type(self.dtype), feature_layers)
    
    def custom_attn(self, attn_layer, x, attn_mask=None):
        num_heads = attn_layer.num_heads
        _, bsz, embed_dim = x.size()
        head_dim = embed_dim // num_heads
        scale = head_dim ** -0.5

        q, k, v = F.linear(x, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        ################### fused-attn ######################
        # sum to 1
        attn_mask /= torch.sum(attn_mask, dim=-2, keepdim=True)
        attn_mask /= torch.sum(attn_mask, dim=-1, keepdim=True)
        attn_mask = (attn_mask + attn_mask.transpose(-2, -1)) / 2  # symmetrize the attention mask
        attn_mask -= attn_mask.mean(-2, keepdim=True)
        attn_mask = torch.clamp(attn_mask, 0)
        attn_mask /= torch.sum(attn_mask, dim=-1, keepdim=True)

        attn_mask = attn_mask.flatten(0, 1)
        attn_weights = torch.repeat_interleave(attn_mask, dim=0, repeats=v.shape[0] // attn_mask.shape[0])
        ################### fused-attn ######################

        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
        attn_output = attn_layer.out_proj(attn_output)
        return attn_output, attn_weights
    
    # attention-based feature enhancement for selected ViT layers
    def ori_attn_enhance(self, patch_tokens, attn_map, feature_layers):
        """
        Args:
            patch_tokens: list of [B, N, C] feature tokens from selected ViT layers
            attn_map: [B, num_heads, N, N] aggregated attention map from shallow layers
            feature_layers: list of int, ViT layer indices to enhance
        Returns:
            enhanced_tokens: list of [B, N, C] reconstructed feature tokens
        """
        # ---- Structural-prior attention map ----
        # Remove CLS token and average over attention heads -> [B, N-1, N-1]
        A_struct = attn_map[:, :, 1:, 1:].mean(dim=1)
        A_struct = A_struct / (A_struct.sum(dim=-1, keepdim=True) + 1e-6)

        # Stack all layer tokens for batched processing
        stacked_tokens = torch.stack(patch_tokens, dim=0)
        L, B, N, C = stacked_tokens.shape
        flat_tokens = stacked_tokens.view(L * B, N, C)

        # Split CLS token and patch tokens
        F_patch = flat_tokens[:, 1:, :]   # [LB, N-1, C]
        flat_cls_tok = flat_tokens[:, :1, :]  # [LB, 1, C]

        # ---- Local-detail attention map ----
        A_local = self.adaptor.convAttn(F_patch)  # DeConBlock -> [LB, N-1, N-1]

        # ---- Fuse structural and local attention maps ----
        A_struct_flat = A_struct.unsqueeze(0).expand(L, -1, -1, -1).reshape(L * B, N - 1, N - 1)
        A_hybrid = A_struct_flat + self.ibha_beta * A_local
        A_hybrid = A_hybrid / (A_hybrid.sum(dim=-1, keepdim=True) + 1e-6)

        # ---- Feature reconstruction via hybrid attention ----
        V = self.ibha_v_proj(F_patch)                              # value projection
        context = torch.bmm(A_hybrid, V)                           # spatial weighting
        F_r_patch = self.ibha_ln(F_patch + context)                # residual + LayerNorm

        # Re-attach CLS token and reshape back
        flat_out = torch.cat([flat_cls_tok, F_r_patch], dim=1)     # [LB, N, C]
        out_tensor = flat_out.view(L, B, N, C)

        # Convert back to list of [B, N, C]
        enhanced_tokens = [out_tensor[i] for i in range(L)]
        return enhanced_tokens

    # Encode image features with attention enhancement and adaptor refinement
    def detect_encode_image(self, image, args):
        img_tokens, attn_maps_sum = self.encode_image(image, args.feature_layers,
                                                       cspf_start=self.cspf_start,
                                                       cspf_end=self.cspf_end)
        img_tokens = self.ori_attn_enhance(img_tokens, attn_maps_sum, args.feature_layers)
        img_tokens = [self.visual.ln_post(self.adaptor(img_token)) @ self.visual.proj for img_token in img_tokens]
        return img_tokens
    
    # Forward inference with IDSR (main segmentation path)
    def detect_forward_seg(self, image, args, categories=None):
        img_tokens = self.detect_encode_image(image, args)

        # Global visual feature from the last layer's CLS token
        F_image = img_tokens[-1][:, 0, :]  # [B, C]

        if categories is not None:
            # IDSR: instance-aware text features
            T_N, T_A_s = self.encode_state_prompt_idsr(F_image, categories)
            # Average over K queries for final abnormal feature
            T_A_eff = T_A_s.mean(dim=1)  # [B, C]
            # Store IDSR outputs for loss computation
            self._idsr_outputs = (T_A_s, F_image, T_A_s)
        else:
            # Fallback: fixed prompts
            text_features = self.encode_state_prompt()
            text_features = torch.nn.functional.normalize(text_features, dim=-1)
            T_N = text_features[0:1].expand(image.size(0), -1)  # [B, C]
            T_A_eff = text_features[1:2].expand(image.size(0), -1)  # [B, C]
            self._idsr_outputs = None

        T_N = torch.nn.functional.normalize(T_N, dim=-1)
        T_A_eff = torch.nn.functional.normalize(T_A_eff, dim=-1)

        # Score each feature level against text features
        scores = 0
        for img_token in img_tokens:
            img_token = torch.nn.functional.normalize(img_token, dim=-1)
            # Per-image anomaly scores: [normal, abnormal]
            s_n = torch.bmm(img_token, T_N.unsqueeze(-1))      # [B, N, 1]
            s_a = torch.bmm(img_token, T_A_eff.unsqueeze(-1))  # [B, N, 1]
            score = torch.cat([s_n, s_a], dim=-1) / 0.07       # [B, N, 2]
            scores += score

        prob = torch.softmax(scores, dim=-1)
        cls_label = prob[:, 0, 1].view(-1)
        predict_map = prob[:, 1:, 1]

        b, l = predict_map.size()
        h = w = int(math.sqrt(l))
        predict_map = predict_map.reshape(b, 1, h, w)
        return cls_label, predict_map, img_tokens

    def detect_forward(self, image, args, categories=None):
        cls_label, predict_map, _ = self.detect_forward_seg(image, args, categories=categories)
        return cls_label, predict_map
    

    def forward(self, image, text):
        image_features = self.encode_image(image)
        if isinstance(image_features, (list, tuple)):
            image_features = image_features[0]
        text_features = self.encode_text(text)
        if isinstance(text_features, (list, tuple)):
            text_features = text_features[0]

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()