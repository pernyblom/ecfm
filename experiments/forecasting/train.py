from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.forecasting.data.dataset import ForecastDataset, split_dataset
from experiments.forecasting.metrics import ade_fde
from experiments.forecasting.models.fusion import MultiRepForecast
from experiments.forecasting.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    dataset = ForecastDataset(
        images_root=Path(data_cfg["images_root"]),
        labels_root=Path(data_cfg["labels_root"]),
        representations=data_cfg["representations"],
        past_steps=data_cfg["past_steps"],
        future_steps=data_cfg["future_steps"],
        stride=data_cfg["stride"],
        image_size=tuple(data_cfg["image_size"]),
        select_box=data_cfg.get("select_box", "largest"),
        drop_empty=bool(data_cfg.get("drop_empty", True)),
    )
    train_set, val_set = split_dataset(
        dataset, train_split=data_cfg["train_split"], seed=data_cfg["seed"]
    )

    train_loader = DataLoader(
        train_set, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=0
    )

    device = torch.device(
        train_cfg.get("device", "cpu")
        if torch.cuda.is_available()
        else "cpu"
    )
    model = MultiRepForecast(
        reps=data_cfg["representations"],
        cnn_channels=model_cfg["cnn_channels"],
        feature_dim=model_cfg["feature_dim"],
        use_past_boxes=model_cfg["use_past_boxes"],
        rnn_hidden=model_cfg["rnn_hidden"],
        rnn_layers=model_cfg["rnn_layers"],
        future_steps=data_cfg["future_steps"],
    ).to(device)

    optim = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    loss_fn = nn.L1Loss()

    for epoch in range(train_cfg["epochs"]):
        model.train()
        for step, batch in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in batch.inputs.items()}
            past_boxes = batch.past_boxes.to(device)
            future = batch.future_boxes.to(device)

            pred = model(inputs, past_boxes)
            loss = loss_fn(pred, future)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if step % train_cfg["log_every"] == 0:
                ade, fde = ade_fde(pred.detach(), future)
                print(
                    f"epoch {epoch} step {step} loss {loss.item():.4f} "
                    f"ADE {ade.item():.4f} FDE {fde.item():.4f}"
                )

        model.eval()
        with torch.no_grad():
            losses = []
            ades = []
            fdes = []
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.inputs.items()}
                past_boxes = batch.past_boxes.to(device)
                future = batch.future_boxes.to(device)
                pred = model(inputs, past_boxes)
                loss = loss_fn(pred, future)
                ade, fde = ade_fde(pred, future)
                losses.append(loss.item())
                ades.append(ade.item())
                fdes.append(fde.item())
            if losses:
                print(
                    f"val epoch {epoch} loss {sum(losses)/len(losses):.4f} "
                    f"ADE {sum(ades)/len(ades):.4f} FDE {sum(fdes)/len(fdes):.4f}"
                )


if __name__ == "__main__":
    main()
