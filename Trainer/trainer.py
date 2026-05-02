import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast  # type:ignore
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from core.config import Config
from model.model import NextFrameModel


class Trainer:
    def __init__(
        self,
        config: Config,
        model: NextFrameModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: torch.device,
        tokenizer=None,
        diffusion=None,
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.device = device
        self.scaler = GradScaler(
            device=self.device.type, enabled=(device.type == "cuda")
        )
        self.step = 0
        self.epoch = 0

        self.ckpt_dir = Path("checkpoints")
        self.ckpt_dir.mkdir(exist_ok=True)

    def train(self):
        max_steps = self.config.training.max_steps
        grad_accum = self.config.training.grad_accum_steps

        data_iter = iter(self.train_loader)
        pbar = tqdm(total=max_steps, desc="Training")

        self.optimizer.zero_grad(set_to_none=True)

        while self.step < max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)
                self.epoch += 1

            loss = self._train_step(batch)
            raw_loss = loss.item()

            loss = loss / grad_accum
            self.scaler.scale(loss).backward()

            if (self.step + 1) % grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.grad_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

            self.step += 1
            pbar.update(1)
            pbar.set_postfix(loss=raw_loss)

            if self.step % self.config.training.log_interval == 0:
                tqdm.write(f"Step {self.step}: train_loss={raw_loss:.4f}")

            if self.step % self.config.training.eval_interval == 0:
                val_loss = self.evaluate()
                tqdm.write(f"Step {self.step}: val_loss={val_loss:.4f}")
                self.save_checkpoint(val_loss=val_loss)

        pbar.close()

    def _train_step(self, batch):
        frame_t, frame_tp1, action = batch
        frame_t = frame_t.to(self.device)
        frame_tp1 = frame_tp1.to(self.device)
        action = action.to(self.device)

        with autocast(
            device_type=self.device.type, enabled=(self.device.type == "cuda")
        ):
            pred = self.model(frame_t, action)
            loss = F.mse_loss(pred, frame_tp1)

        return loss

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        for batch in self.val_loader:
            frame_t, frame_tp1, action = batch
            frame_t = frame_t.to(self.device)
            frame_tp1 = frame_tp1.to(self.device)
            action = action.to(self.device)

            with autocast(
                device_type=self.device.type, enabled=(self.device.type == "cuda")
            ):
                pred = self.model(frame_t, action)
                loss = F.mse_loss(pred, frame_tp1)

            total_loss += loss.item()
            total_batches += 1

        self.model.train()
        return total_loss / max(1, total_batches)

    def save_checkpoint(self, val_loss):
        state = {
            "step": self.step,
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "val_loss": val_loss,
        }

        torch.save(state, self.ckpt_dir / "last.pt")

        if not hasattr(self, "best_loss") or val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(state, self.ckpt_dir / "best.pt")
