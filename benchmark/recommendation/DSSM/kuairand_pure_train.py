from Nexus.training.embedder.recommendation.modeling import DSSMInBathcRetriever
from Nexus.training.embedder.recommendation.runner import RetrieverRunner

import time

import torch, random, numpy as np

def main():
    
    start_t = time.time()
    
    data_config_path = "./data_kuairand_pure_config.json"
    train_config_path = "./kuairand_pure_training_config.json"
    model_config_path = "./kuairand_pure_model_config.json"
    
    runner = RetrieverRunner(
        model_config_or_path=model_config_path,
        data_config_or_path=data_config_path,
        train_config_or_path=train_config_path,
        model_class=DSSMInBathcRetriever,
    )
    
    runner.run()
    
    print(f"Total time: {time.time() - start_t}s")

if __name__ == "__main__":
    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    start_t = time.time()
    main()
    end_t = time.time()
    print(f"Total time: {end_t - start_t:.2f}s")