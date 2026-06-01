# sam3-onnx-cpp/export/src/modules.py
import torch
import torch.nn.functional as F
from torch import nn


def get_1d_sine_pe(pos_inds: torch.Tensor, dim: int, temperature: int = 10000) -> torch.Tensor:
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)
    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    return torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)


def _tokens_to_map(tokens: torch.Tensor, feat_hw: tuple[int, int]) -> torch.Tensor:
    height, width = feat_hw
    batch = tokens.size(1)
    channels = tokens.size(2)
    return tokens.permute(1, 2, 0).reshape(batch, channels, height, width)


def _complex_mult_real(
    x_real: torch.Tensor,
    x_imag: torch.Tensor,
    freqs_real: torch.Tensor,
    freqs_imag: torch.Tensor,
) -> torch.Tensor:
    real_part = x_real * freqs_real - x_imag * freqs_imag
    imag_part = x_real * freqs_imag + x_imag * freqs_real
    return torch.stack((real_part, imag_part), dim=-1)


def _apply_rope_batch_first(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_real: torch.Tensor,
    freqs_imag: torch.Tensor,
    repeat_freqs_k: bool,
    num_k_exclude_rope: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_real = q.float().reshape(q.shape[0], q.shape[1], -1, 2)[..., 0]
    q_imag = q.float().reshape(q.shape[0], q.shape[1], -1, 2)[..., 1]
    rope_len = k.shape[1] - num_k_exclude_rope
    k_rope = k[:, :rope_len]
    k_tail = k[:, rope_len:]
    half_dim = q_real.shape[-1]

    freqs_real_q = freqs_real[: q.shape[1]].unsqueeze(0)
    freqs_imag_q = freqs_imag[: q.shape[1]].unsqueeze(0)
    q_out = _complex_mult_real(q_real, q_imag, freqs_real_q, freqs_imag_q).flatten(2)

    if repeat_freqs_k:
        k_real = k_rope.float().reshape(k_rope.shape[0], -1, q.shape[1], half_dim, 2)[..., 0]
        k_imag = k_rope.float().reshape(k_rope.shape[0], -1, q.shape[1], half_dim, 2)[..., 1]
        freqs_real_k = freqs_real_q.unsqueeze(1)
        freqs_imag_k = freqs_imag_q.unsqueeze(1)
        k_out = _complex_mult_real(k_real, k_imag, freqs_real_k, freqs_imag_k)
        k_out = k_out.flatten(3).reshape(k.shape[0], rope_len, q.shape[2])
    else:
        k_real = k_rope.float().reshape(k_rope.shape[0], k_rope.shape[1], half_dim, 2)[..., 0]
        k_imag = k_rope.float().reshape(k_rope.shape[0], k_rope.shape[1], half_dim, 2)[..., 1]
        k_out = _complex_mult_real(k_real, k_imag, freqs_real_q, freqs_imag_q).flatten(2)
    k_out = torch.cat((k_out, k_tail), dim=1)
    return q_out.type_as(q), k_out.type_as(k)


class _ManualRoPEAttention(nn.Module):
    def __init__(self, attn_module: nn.Module) -> None:
        super().__init__()
        self.attn = attn_module
        self.q_proj = attn_module.q_proj
        self.k_proj = attn_module.k_proj
        self.v_proj = attn_module.v_proj
        self.out_proj = attn_module.out_proj
        self.scale = float(attn_module.internal_dim) ** -0.5

    @torch.no_grad()
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q, k = _apply_rope_batch_first(
            q,
            k,
            self.attn.freqs_cis_real,
            self.attn.freqs_cis_imag,
            repeat_freqs_k=self.attn.rope_k_repeat,
            num_k_exclude_rope=num_k_exclude_rope,
        )
        attn_scores = torch.matmul(q, k.transpose(1, 2)) * self.scale
        attn_probs = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, v)
        return self.out_proj(out)


class _ManualTrackerLayer(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.norm1 = layer.norm1
        self.norm2 = layer.norm2
        self.norm3 = layer.norm3
        self.linear1 = layer.linear1
        self.linear2 = layer.linear2
        self.activation = layer.activation
        self.pos_enc_at_attn = layer.pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = layer.pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = layer.pos_enc_at_cross_attn_keys
        self.cross_attention_first = layer.cross_attention_first
        self.self_attn = _ManualRoPEAttention(layer.self_attn)
        self.cross_attn = _ManualRoPEAttention(layer.cross_attn_image)

    @torch.no_grad()
    def _forward_sa(self, tgt: torch.Tensor, query_pos: torch.Tensor) -> torch.Tensor:
        tgt2 = self.norm1(tgt)
        q = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        return tgt + self.self_attn(q, q, tgt2)

    @torch.no_grad()
    def _forward_ca(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor,
        pos: torch.Tensor,
        num_k_exclude_rope: int,
    ) -> torch.Tensor:
        tgt2 = self.norm2(tgt)
        q = tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2
        k = memory + pos if self.pos_enc_at_cross_attn_keys else memory
        return tgt + self.cross_attn(q, k, memory, num_k_exclude_rope=num_k_exclude_rope)

    @torch.no_grad()
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor,
        pos: torch.Tensor,
        num_k_exclude_rope: int,
    ) -> torch.Tensor:
        if self.cross_attention_first:
            tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
            tgt = self._forward_sa(tgt, query_pos)
        else:
            tgt = self._forward_sa(tgt, query_pos)
            tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.activation(self.linear1(tgt2)))
        return tgt + tgt2


class ImageEncoder(nn.Module):
    def __init__(self, image_model) -> None:
        super().__init__()
        self.image_encoder = image_model.backbone
        self.tracker = image_model.inst_interactive_predictor.model

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        backbone_out = self.image_encoder.forward_image(image)
        backbone_out = backbone_out["sam2_backbone_out"].copy()
        backbone_out["backbone_fpn"] = list(backbone_out["backbone_fpn"])
        backbone_out["vision_pos_enc"] = list(backbone_out["vision_pos_enc"])

        backbone_out["backbone_fpn"][0] = self.tracker.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0]
        )
        backbone_out["backbone_fpn"][1] = self.tracker.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1]
        )

        _, current_feats, current_pos, feat_sizes = self.tracker._prepare_backbone_features(
            backbone_out
        )

        image_embeddings = _tokens_to_map(
            current_feats[-1] + self.tracker.no_mem_embed,
            feat_sizes[-1],
        )
        current_vision_feat = _tokens_to_map(current_feats[-1], feat_sizes[-1])
        high_res_0 = _tokens_to_map(current_feats[0], feat_sizes[0])
        high_res_1 = _tokens_to_map(current_feats[1], feat_sizes[1])

        return (
            image_embeddings,
            high_res_0,
            high_res_1,
            current_vision_feat,
            current_pos[-1],
        )


class ImageDecoder(nn.Module):
    def __init__(self, image_model) -> None:
        super().__init__()
        self.tracker = image_model.inst_interactive_predictor.model

    @torch.no_grad()
    def _forward(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        image_embed: torch.Tensor,
        high_res_feats_0: torch.Tensor,
        high_res_feats_1: torch.Tensor,
        mask_inputs: torch.Tensor | None,
    ):
        point_inputs = {
            "point_coords": point_coords.to(torch.float32),
            "point_labels": point_labels.to(torch.int32),
        }
        (
            _low_res_multimasks,
            _high_res_multimasks,
            iou_scores,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = self.tracker._forward_sam_heads(
            backbone_features=image_embed,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            high_res_features=[high_res_feats_0, high_res_feats_1],
            multimask_output=True,
        )
        return (
            obj_ptr,
            low_res_masks,
            high_res_masks,
            object_score_logits,
            iou_scores,
            _low_res_multimasks,
            _high_res_multimasks,
        )

    @torch.no_grad()
    def forward(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        image_embed: torch.Tensor,
        high_res_feats_0: torch.Tensor,
        high_res_feats_1: torch.Tensor,
    ):
        return self._forward(
            point_coords,
            point_labels,
            image_embed,
            high_res_feats_0,
            high_res_feats_1,
            mask_inputs=None,
        )


class ImageDecoderWithMask(ImageDecoder):
    @torch.no_grad()
    def forward(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        image_embed: torch.Tensor,
        high_res_feats_0: torch.Tensor,
        high_res_feats_1: torch.Tensor,
        mask_inputs: torch.Tensor,
    ):
        return self._forward(
            point_coords,
            point_labels,
            image_embed,
            high_res_feats_0,
            high_res_feats_1,
            mask_inputs=mask_inputs,
        )


class MemAttention(nn.Module):
    def __init__(self, image_model) -> None:
        super().__init__()
        self.tracker = image_model.inst_interactive_predictor.model
        self.hidden_dim = self.tracker.hidden_dim
        self.mem_dim = self.tracker.mem_dim
        self.ptr_tokens_per_obj = self.hidden_dim // self.mem_dim
        self.max_obj_ptrs_in_encoder = self.tracker.max_obj_ptrs_in_encoder
        self.image_embed_size = int(self.tracker.sam_image_embedding_size)
        self._prepare_rope_buffers(torch.device("cpu"), self.image_embed_size, self.image_embed_size)
        encoder = self.tracker.transformer.encoder
        self.pos_enc_at_input = bool(getattr(encoder, "pos_enc_at_input", False))
        self.layers = nn.ModuleList([_ManualTrackerLayer(layer) for layer in encoder.layers])
        self.output_norm = encoder.norm

    def _prepare_rope_buffers(self, device: torch.device, height: int, width: int) -> None:
        for module in self.tracker.transformer.encoder.modules():
            if not hasattr(module, "compute_cis") or not hasattr(module, "freqs_cis"):
                continue
            module.use_rope_real = True
            module.freqs_cis = module.compute_cis(end_x=width, end_y=height, device=device)
            module.freqs_cis_real = module.freqs_cis.real
            module.freqs_cis_imag = module.freqs_cis.imag

    @torch.no_grad()
    def forward(
        self,
        current_vision_feat: torch.Tensor,
        current_vision_pos_embed: torch.Tensor,
        memory_obj_ptrs: torch.Tensor,
        memory_obj_tpos: torch.Tensor,
        memory_mask_feats: torch.Tensor,
        memory_mask_pos: torch.Tensor,
        memory_mask_tpos_idx: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = current_vision_feat.shape
        if not torch.onnx.is_in_onnx_export():
            self._prepare_rope_buffers(current_vision_feat.device, height, width)
        current_tokens = current_vision_feat.permute(0, 2, 3, 1).reshape(
            batch, height * width, channels
        )
        current_pos = current_vision_pos_embed.permute(1, 0, 2).expand(batch, -1, -1)

        num_mem_frames = memory_mask_feats.shape[0]
        mask_tokens = memory_mask_feats.reshape(num_mem_frames, self.mem_dim, height * width)
        mask_tokens = mask_tokens.permute(0, 2, 1).reshape(1, -1, self.mem_dim)
        mask_tokens = mask_tokens.expand(batch, -1, -1)

        mask_tpos = self.tracker.maskmem_tpos_enc.index_select(
            0, memory_mask_tpos_idx.to(torch.int64)
        )
        mask_tpos = mask_tpos.squeeze(1).squeeze(1)[..., None, None]
        mask_pos = memory_mask_pos + mask_tpos
        mask_pos = mask_pos.reshape(num_mem_frames, self.mem_dim, height * width)
        mask_pos = mask_pos.permute(0, 2, 1).reshape(1, -1, self.mem_dim)
        mask_pos = mask_pos.expand(batch, -1, -1)

        obj_ptrs = memory_obj_ptrs.reshape(-1, self.hidden_dim)
        obj_ptr_tokens = obj_ptrs.reshape(-1, self.ptr_tokens_per_obj, self.mem_dim)
        obj_ptr_tokens = obj_ptr_tokens.reshape(1, -1, self.mem_dim)
        obj_ptr_tokens = obj_ptr_tokens.expand(batch, -1, -1)

        obj_pos_norm = memory_obj_tpos.to(torch.float32)
        obj_pos_norm = obj_pos_norm / float(max(self.max_obj_ptrs_in_encoder - 1, 1))
        obj_pos = get_1d_sine_pe(obj_pos_norm, dim=self.hidden_dim)
        obj_pos = self.tracker.obj_ptr_tpos_proj(obj_pos)
        obj_pos = obj_pos.repeat_interleave(self.ptr_tokens_per_obj, dim=0)
        obj_pos = obj_pos.unsqueeze(0).expand(batch, -1, -1)

        prompt = torch.cat((mask_tokens, obj_ptr_tokens), dim=1)
        prompt_pos = torch.cat((mask_pos, obj_pos), dim=1)
        num_obj_ptr_tokens = obj_ptr_tokens.shape[1]

        output = current_tokens
        if self.pos_enc_at_input:
            output = output + 0.1 * current_pos
        for layer in self.layers:
            output = layer(
                output,
                prompt,
                current_pos,
                prompt_pos,
                num_obj_ptr_tokens,
            )
        output = self.output_norm(output)
        return output.reshape(batch, height, width, channels).permute(0, 3, 1, 2)


class MemEncoder(nn.Module):
    def __init__(self, image_model) -> None:
        super().__init__()
        self.tracker = image_model.inst_interactive_predictor.model
        self.scale = float(self.tracker.sigmoid_scale_for_mem_enc)
        self.bias = float(self.tracker.sigmoid_bias_for_mem_enc)

    @torch.no_grad()
    def forward(
        self,
        pred_mask_high_res: torch.Tensor,
        current_vision_feat: torch.Tensor,
        object_score_logits: torch.Tensor,
        is_mask_from_points: torch.Tensor,
    ):
        mask_from_points = is_mask_from_points.to(torch.float32).reshape(-1, 1, 1, 1)
        binary_mask = (pred_mask_high_res > 0).to(torch.float32)
        prob_mask = torch.sigmoid(pred_mask_high_res)
        mask_for_mem = binary_mask * mask_from_points + prob_mask * (1.0 - mask_from_points)
        mask_for_mem = mask_for_mem * self.scale + self.bias

        maskmem_backbone = self.tracker.maskmem_backbone
        downsampler = maskmem_backbone.mask_downsampler
        if downsampler.interpol_size is not None and downsampler.interpol_size != list(
            mask_for_mem.shape[-2:]
        ):
            mask_for_mem = F.interpolate(
                mask_for_mem.float(),
                size=downsampler.interpol_size,
                align_corners=False,
                mode="bilinear",
                antialias=False,
            )
        mask_for_mem = downsampler.encoder(mask_for_mem)

        pix_feat = current_vision_feat.to(mask_for_mem.device)
        maskmem_features = maskmem_backbone.pix_feat_proj(pix_feat)
        maskmem_features = maskmem_features + mask_for_mem
        maskmem_features = maskmem_backbone.fuser(maskmem_features)
        maskmem_features = maskmem_backbone.out_proj(maskmem_features)
        maskmem_pos_enc = maskmem_backbone.position_encoding(maskmem_features).to(
            maskmem_features.dtype
        )

        is_obj_appearing = (object_score_logits > 0).to(maskmem_features.dtype)
        maskmem_features = maskmem_features + (
            1.0 - is_obj_appearing[..., None, None]
        ) * self.tracker.no_obj_embed_spatial[..., None, None].expand_as(maskmem_features)

        return maskmem_features, maskmem_pos_enc
