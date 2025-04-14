import os
import torch
import hydra
from util import refine_cfg, seed_torch
from data import get_data_loaders
from algorithm import get_alg_buffer

@hydra.main(config_path="configs", config_name="main")
def main(cfg):
    cfg = refine_cfg(cfg)
    seed_torch(cfg.seed)
    device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() and cfg.device >= 0 else "cpu")
    print(f"Device: {device}")
    print(f"Work directory: {os.getcwd()}")

    #### LLM constraint embedding step
    if cfg.llm != 'none':
        from llm import embed_constraints
        embed_constraints(cfg)

    if cfg.eval_model != 'none':
        train_loader, test_loader = get_data_loaders(cfg)
        alg, buffer = get_alg_buffer(cfg, device)

        from train import train_loop
        train_loop(cfg, device, alg, buffer, train_loader, test_loader)

        if cfg.eval:
            from eval import evaluate
            from util import get_logr_scaler

            logr_scaler = get_logr_scaler(cfg, process_ratio=0.0, reward_exp=None)
            result = {"set_size": {}, "logr_scaled": {}, "train_step": {}, "train_data_used": {}}

            evaluate(cfg, device, test_loader, alg, 0, 0, logr_scaler, result, ep=0)

    else:
        assert cfg.eval
        _, test_loader = get_data_loaders(cfg)
        alg, _ = get_alg_buffer(cfg, device)
        alg.load(cfg.eval_model)

        from eval import evaluate
        from util import get_logr_scaler

        evaluate(cfg, device, test_loader, alg, 0, 0, logr_scaler, result, ep=0)

if __name__ == "__main__":
    main()