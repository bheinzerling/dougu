import torch


def save_model(model, model_file, log=None):
    """Save a pytorch model to model_file"""
    model_file.parent.mkdir(parents=True, exist_ok=True)
    with model_file.open("wb") as out:
        torch.save(model.state_dict(), out)
    if log:
        log.info("saved %s", model_file)
