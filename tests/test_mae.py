import pytest
import torch

from ecfm.models.mae import EventMAE


def _model(use_relative_bias: bool) -> EventMAE:
    torch.manual_seed(0)
    return EventMAE(
        patch_size=8,
        embed_dim=24,
        num_heads=3,
        num_layers=2,
        decoder_embed_dim=16,
        decoder_num_heads=4,
        decoder_num_layers=1,
        mlp_ratio=2.0,
        plane_embed_dim=8,
        metadata_dim=9,
        num_tokens=6,
        num_planes=3,
        use_pos_embedding=False,
        use_relative_bias=use_relative_bias,
    ).eval()


@pytest.mark.parametrize("use_relative_bias", [False, True])
def test_padding_does_not_change_valid_outputs(use_relative_bias: bool) -> None:
    model = _model(use_relative_bias)
    torch.manual_seed(1)
    patches = torch.randn(1, 6, 2, 8, 8)
    metadata = torch.rand(1, 6, 9)
    plane_ids = torch.randint(0, 3, (1, 6))
    valid_mask = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.float32)

    changed_patches = patches.clone()
    changed_metadata = metadata.clone()
    changed_plane_ids = plane_ids.clone()
    changed_patches[:, 3:] = torch.randn_like(changed_patches[:, 3:]) * 100
    changed_metadata[:, 3:] = torch.rand_like(changed_metadata[:, 3:]) * 100
    changed_plane_ids[:, 3:] = torch.randint(0, 3, changed_plane_ids[:, 3:].shape)

    with torch.no_grad():
        first = model(
            patches,
            metadata,
            plane_ids,
            valid_mask=valid_mask,
        )
        second = model(
            changed_patches,
            changed_metadata,
            changed_plane_ids,
            valid_mask=valid_mask,
        )

    for first_tensor, second_tensor in zip(first, second):
        torch.testing.assert_close(first_tensor[:, :3], second_tensor[:, :3])
        assert torch.count_nonzero(first_tensor[:, 3:]) == 0
        assert torch.count_nonzero(second_tensor[:, 3:]) == 0


def test_valid_mask_shape_is_checked() -> None:
    model = _model(use_relative_bias=True)
    patches = torch.zeros(1, 6, 2, 8, 8)
    metadata = torch.zeros(1, 6, 9)
    plane_ids = torch.zeros(1, 6, dtype=torch.long)

    with pytest.raises(ValueError, match="valid_mask must have shape"):
        model.encode(
            patches,
            metadata,
            plane_ids,
            valid_mask=torch.ones(1, 5),
        )
