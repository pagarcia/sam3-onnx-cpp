import torch
import torch.nn.functional as F
from torch import nn


NO_OBJ_SCORE = -1024.0


def _apply_linear_no_obj_ptr(
    tracker,
    obj_ptr: torch.Tensor,
    object_score_logits: torch.Tensor,
) -> torch.Tensor:
    if not getattr(tracker, "use_linear_no_obj_ptr", False):
        raise RuntimeError(
            "The SAM 3.1 ONNX exporter currently requires use_linear_no_obj_ptr=True."
        )
    is_obj_appearing = (object_score_logits > tracker.object_score_logit_threshold).to(
        obj_ptr.dtype
    )
    return is_obj_appearing * obj_ptr + (1.0 - is_obj_appearing) * tracker.no_obj_ptr_linear(
        obj_ptr
    )


class SAM31InteractiveDecoder(nn.Module):
    def __init__(self, tracker) -> None:
        super().__init__()
        self.tracker = tracker

    @torch.no_grad()
    def forward(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        image_embed: torch.Tensor,
        high_res_0: torch.Tensor,
        high_res_1: torch.Tensor,
    ):
        sparse_embeddings, dense_embeddings = self.tracker.interactive_sam_prompt_encoder(
            points=(point_coords.to(torch.float32), point_labels.to(torch.int32)),
            boxes=None,
            masks=None,
        )
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.tracker.interactive_sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=self.tracker.interactive_sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=True,
            high_res_features=[high_res_0, high_res_1],
        )

        if self.tracker.pred_obj_scores:
            is_obj_appearing = object_score_logits > self.tracker.object_score_logit_threshold
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                low_res_multimasks.new_tensor(NO_OBJ_SCORE),
            )

        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.tracker.image_size, self.tracker.image_size),
            mode="bilinear",
            align_corners=False,
        )

        best_iou_inds = torch.argmax(ious, dim=-1)
        batch_inds = torch.arange(ious.shape[0], device=image_embed.device)
        low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
        high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
        sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        obj_ptr = self.tracker.interactive_obj_ptr_proj(sam_output_token)
        obj_ptr = _apply_linear_no_obj_ptr(self.tracker, obj_ptr, object_score_logits)
        return obj_ptr, low_res_masks, high_res_masks, object_score_logits, ious


class SAM31PropagationDecoder(nn.Module):
    def __init__(self, tracker) -> None:
        super().__init__()
        self.tracker = tracker

    @torch.no_grad()
    def forward(
        self,
        image_embed: torch.Tensor,
        high_res_0: torch.Tensor,
        high_res_1: torch.Tensor,
        valid_object_mask: torch.Tensor,
    ):
        output_valid_embed = self.tracker.output_valid_embed.unsqueeze(0)
        output_invalid_embed = self.tracker.output_invalid_embed.unsqueeze(0)
        valid_mask = valid_object_mask.to(image_embed.dtype).unsqueeze(-1)
        extra_per_object_embeddings = (
            valid_mask * output_valid_embed + (1.0 - valid_mask) * output_invalid_embed
        )

        out = self.tracker.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=self.tracker.get_propagation_dense_pe(),
            high_res_features=[high_res_0, high_res_1],
            multimask_output=True,
            extra_per_object_embeddings=extra_per_object_embeddings,
        )

        masks = out["masks"]
        ious = out["iou_pred"]
        sam_tokens = out["sam_tokens_out"]
        object_score_logits = out["object_score_logits"]
        is_obj_appearing = object_score_logits > self.tracker.object_score_logit_threshold
        masks = torch.where(
            is_obj_appearing.unsqueeze(-1).unsqueeze(-1),
            masks,
            masks.new_tensor(NO_OBJ_SCORE),
        )

        batch, slots, num_masks, height, width = masks.shape
        best_iou_inds = torch.argmax(ious, dim=-1)
        flat_idx = torch.arange(batch * slots, device=image_embed.device)
        low_res_masks = (
            masks.reshape(batch * slots, num_masks, height, width)[
                flat_idx,
                best_iou_inds.reshape(-1),
            ]
            .unsqueeze(1)
            .reshape(batch, slots, 1, height, width)
        )
        high_res_masks = F.interpolate(
            low_res_masks.flatten(0, 1),
            size=(self.tracker.image_size, self.tracker.image_size),
            mode="bilinear",
            align_corners=False,
        ).view(batch, slots, 1, self.tracker.image_size, self.tracker.image_size)

        sam_output_token = sam_tokens.reshape(
            batch * slots,
            sam_tokens.shape[2],
            sam_tokens.shape[3],
        )[flat_idx, best_iou_inds.reshape(-1)].reshape(batch, slots, -1)
        obj_ptr = self.tracker.obj_ptr_proj(sam_output_token)
        obj_ptr = _apply_linear_no_obj_ptr(self.tracker, obj_ptr, object_score_logits)
        return obj_ptr, low_res_masks, high_res_masks, object_score_logits, ious


class SAM31MemoryEncoder(nn.Module):
    def __init__(self, tracker) -> None:
        super().__init__()
        self.tracker = tracker

    @torch.no_grad()
    def forward(
        self,
        pred_masks_high_res: torch.Tensor,
        current_vision_feat: torch.Tensor,
        object_score_logits: torch.Tensor,
        conditioning_mask: torch.Tensor,
    ):
        mask_for_mem = torch.sigmoid(pred_masks_high_res)
        if self.tracker.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * float(self.tracker.sigmoid_scale_for_mem_enc)
        if self.tracker.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + float(self.tracker.sigmoid_bias_for_mem_enc)

        mux_mask_for_mem = mask_for_mem
        if self.tracker.condition_as_mask_input:
            embedded_conditions = conditioning_mask.to(mask_for_mem.dtype).unsqueeze(-1).unsqueeze(
                -1
            )
            embedded_conditions = embedded_conditions.expand_as(mask_for_mem)
            mux_mask_for_mem = torch.cat([mux_mask_for_mem, embedded_conditions], dim=1)

        maskmem_backbone = self.tracker.maskmem_backbone
        downsampler = maskmem_backbone.mask_downsampler
        if downsampler.interpol_size is not None and downsampler.interpol_size != list(
            mux_mask_for_mem.shape[-2:]
        ):
            mux_mask_for_mem = F.interpolate(
                mux_mask_for_mem.float(),
                size=downsampler.interpol_size,
                align_corners=False,
                mode="bilinear",
                antialias=False,
            )
        mux_mask_for_mem = downsampler.encoder(mux_mask_for_mem)

        maskmem_features = maskmem_backbone.pix_feat_proj(current_vision_feat)
        maskmem_features = maskmem_features + mux_mask_for_mem
        maskmem_features = maskmem_backbone.fuser(maskmem_features)
        maskmem_features = maskmem_backbone.out_proj(maskmem_features)
        maskmem_pos_enc = maskmem_backbone.position_encoding(maskmem_features).to(
            maskmem_features.dtype
        )

        if self.tracker.no_obj_embed_spatial is not None:
            is_obj_appearing = (
                object_score_logits > self.tracker.object_score_logit_threshold
            ).to(maskmem_features.dtype)
            no_obj_embed = (
                (1.0 - is_obj_appearing) * self.tracker.no_obj_embed_spatial.unsqueeze(0)
            ).sum(dim=1)
            maskmem_features = maskmem_features + no_obj_embed[..., None, None].expand_as(
                maskmem_features
            )

        return maskmem_features, maskmem_pos_enc


class SAM31MemoryAttentionCore(nn.Module):
    def __init__(self, tracker) -> None:
        super().__init__()
        self.tracker = tracker
        self.encoder = tracker.transformer.encoder
        self._prepare_rope_buffers(torch.device("cpu"))

    def _prepare_rope_buffers(self, device: torch.device) -> None:
        for layer in self.encoder.layers:
            for rope in (layer.self_attention_rope, layer.cross_attention_rope):
                rope.freqs_cis = rope.compute_cis(end_x=72, end_y=72, device=device)
                rope.freqs_cis_real = rope.freqs_cis.real
                rope.freqs_cis_imag = rope.freqs_cis.imag

    @torch.no_grad()
    def forward(
        self,
        image_tokens: torch.Tensor,
        src_tokens: torch.Tensor,
        memory_image_tokens: torch.Tensor,
        memory_tokens: torch.Tensor,
        image_pos: torch.Tensor,
        src_pos: torch.Tensor,
        memory_image_pos: torch.Tensor,
        memory_pos: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.encoder.layers:
            for rope in (layer.self_attention_rope, layer.cross_attention_rope):
                rope.freqs_cis = rope.freqs_cis.to(image_tokens.device)
                rope.freqs_cis_real = rope.freqs_cis_real.to(image_tokens.device)
                rope.freqs_cis_imag = rope.freqs_cis_imag.to(image_tokens.device)

        out = self.encoder(
            image=image_tokens,
            src=src_tokens,
            memory_image=memory_image_tokens,
            memory=memory_tokens,
            image_pos=image_pos,
            src_pos=src_pos,
            memory_image_pos=memory_image_pos,
            memory_pos=memory_pos,
            num_obj_ptr_tokens=0,
        )
        return out["memory"]
