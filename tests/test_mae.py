import pytest
import torch

from ecfm.models.rel_attention import RelativeBias
from ecfm.models.mae import EventMAE
from ecfm.training.train import train_one_epoch


def test_variable_token_count_without_absolute_positions():
    model = _model(True)
    patches = torch.rand(2, 9, 2, 8, 8)
    metadata = torch.rand(2, 9, 9)
    planes = torch.zeros(2, 9, dtype=torch.long)
    valid = torch.ones(2, 9, dtype=torch.bool)
    reconstructed, _, _ = model(patches, metadata, planes, valid_mask=valid)
    assert reconstructed.shape == patches.shape
    model.use_pos_embedding = True
    with pytest.raises(ValueError, match='num_tokens mismatch'):
        model(patches, metadata, planes, valid_mask=valid)


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


def test_relative_bias_shape_and_gradients() -> None:
    module = RelativeBias(num_heads=3, hidden_dim=8)
    metadata = (torch.rand(2, 4, 9) + 0.1).requires_grad_()

    bias = module(metadata)
    assert bias.shape == (2, 3, 4, 4)
    assert torch.isfinite(bias).all()

    bias.square().mean().backward()
    assert metadata.grad is not None
    assert torch.isfinite(metadata.grad).all()
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_relative_bias_encoder_is_permutation_equivariant() -> None:
    model = _model(use_relative_bias=True)
    torch.manual_seed(2)
    patches = torch.randn(2, 6, 2, 8, 8)
    metadata = torch.rand(2, 6, 9) + 0.1
    plane_ids = torch.randint(0, 3, (2, 6))
    valid_mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]],
        dtype=torch.float32,
    )
    permutation = torch.tensor([2, 5, 0, 4, 1, 3])

    with torch.no_grad():
        encoded = model.encode(
            patches,
            metadata,
            plane_ids,
            valid_mask=valid_mask,
        )
        permuted = model.encode(
            patches[:, permutation],
            metadata[:, permutation],
            plane_ids[:, permutation],
            valid_mask=valid_mask[:, permutation],
        )

    torch.testing.assert_close(permuted, encoded[:, permutation])


def test_masked_relative_bias_forward_backward() -> None:
    model = _model(use_relative_bias=True).train()
    torch.manual_seed(3)
    patches = torch.randn(2, 6, 2, 8, 8)
    metadata = torch.rand(2, 6, 9) + 0.1
    plane_ids = torch.randint(0, 3, (2, 6))
    valid_mask = torch.tensor(
        [[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [[1, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0]],
        dtype=torch.bool,
    )

    pred_patch, pred_count, decoded = model(
        patches,
        metadata,
        plane_ids,
        mask=mask,
        valid_mask=valid_mask,
    )
    assert pred_patch.shape == (2, 6, 2, 8, 8)
    assert pred_count.shape == (2, 6, 1)
    assert decoded.shape == (2, 6, 16)

    loss = pred_patch[mask].square().mean() + pred_count[mask].square().mean()
    loss.backward()
    assert model.mask_token.grad is not None
    assert torch.isfinite(model.mask_token.grad).all()
    assert model.rel_bias.mlp[0].weight.grad is not None
    assert torch.isfinite(model.rel_bias.mlp[0].weight.grad).all()


def test_train_one_epoch_handles_variable_region_counts() -> None:
    model = _model(use_relative_bias=True).train()
    torch.manual_seed(4)
    batch = {
        "patches": torch.randn(2, 6, 2, 8, 8),
        "metadata": torch.rand(2, 6, 9) + 0.1,
        "plane_ids": torch.randint(0, 3, (2, 6)),
        "event_counts": torch.rand(2, 6, 1),
        "valid_mask": torch.tensor(
            [[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]],
            dtype=torch.float32,
        ),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.mask_token.detach().clone()

    metrics = train_one_epoch(
        model,
        [batch],
        optimizer,
        torch.device("cpu"),
        mask_ratio=0.5,
        count_loss_weight=0.1,
    )

    assert metrics["loss"] > 0
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert not torch.equal(before, model.mask_token.detach())
