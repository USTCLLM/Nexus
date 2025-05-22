import yaml
import argparse
import pandas as pd
from Nexus.inference.embedder.recommendation import BaseEmbedderInferenceEngine
import pycuda.driver as cuda

import time
from datetime import timedelta


if __name__ == '__main__':

    infer_config_path = "/share/project/haoranjin/Nexus/examples/recommendation/inference/config/kuairand_infer_retrieval_config.yaml"
    infer_df = pd.read_parquet('/share/project/liuqi/Nexus/benchmark/recommendation/KuaiRand/infer.parquet')    

    with open(infer_config_path, 'r') as f:
        config = yaml.safe_load(f)
        print(config)
    
    retriever_inference_engine = BaseEmbedderInferenceEngine(config)
        
    
    
    retriever_inference_engine.cumsum_flag = False
    # start_time = time.time()

    for batch_idx in range(11):
        # 第一个batch都是初始化，一次性的开销，排除掉
        if batch_idx == 1:
            # 记录开始时间
            start_time = time.time()
            retriever_inference_engine.cumsum_flag = True
        print(f"This is batch {batch_idx}")
        batch_st = batch_idx * 128 
        batch_ed = (batch_idx + 1) * 128 
        batch_infer_df = infer_df.iloc[batch_st:batch_ed]
        retriever_outputs = retriever_inference_engine.batch_inference(batch_infer_df)
        print(type(retriever_outputs), retriever_outputs.shape, retriever_outputs)
        
    # 记录结束时间
    end_time = time.time()

    # 计算时间差
    elapsed_time = timedelta(seconds=end_time - start_time)
    print(f"Total Time Cost: {elapsed_time}")

    print(f"get feature time: {retriever_inference_engine.get_features_time:4f} s")
    print(f"model time: {retriever_inference_engine.model_time:4f} s")
    

    if retriever_inference_engine.config['infer_mode'] == 'trt':
        retriever_inference_engine.context.pop()