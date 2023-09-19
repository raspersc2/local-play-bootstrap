import torch
from filelock import FileLock


def save_checkpoint(checkpoint_path, epoch, model, optimizer, beta=0.0, update_steps=0):
    lock_file = f"{str(checkpoint_path)}.lock"
    lock = FileLock(lock_file)
    with lock:
        state = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "beta": beta,
            "update_steps": update_steps,
        }
        f_path = checkpoint_path
        torch.save(state, f_path)


def load_checkpoint(checkpoint_path, model, optimizer, device):
    lock_file = f"{str(checkpoint_path)}.lock"
    lock = FileLock(lock_file)
    with lock:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        return (
            model,
            optimizer,
            checkpoint["epoch"],
            checkpoint["beta"],
            checkpoint["update_steps"],
        )
