import torch
from core.config import Config
from core.factory import build_components


def main():
    config = Config.load_config("config/config.toml")
    comp = build_components(config)

    train_loader = comp.train_loader
    model = comp.model
    device = comp.device

    frame_t, frame_tp1, action = next(iter(train_loader))

    print("input:", frame_t.shape)
    print("target:", frame_tp1.shape)

    with torch.no_grad():
        pred = model(frame_t.to(device), action.to(device))

    print("pred:", pred.shape)


if __name__ == "__main__":
    main()
