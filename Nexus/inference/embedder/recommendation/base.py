import os
from tqdm import tqdm, trange
from typing import cast, Any, List, Union, Optional, Tuple, Type
import onnx
import onnxruntime as ort
import tensorrt as trt
import pandas as pd
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from dataclasses import dataclass, field
from Nexus.abc.inference import AbsEmbedder, InferenceEngine, AbsInferenceArguments
import os
import sys
sys.path.append('.')
import argparse
import time

from pandas import DataFrame
import yaml
import torch
import numpy as np

import tensorrt as trt

# from inference.inference.inference_engine import InferenceEngine
from Nexus.inference.utils import gen_item_index, gen_i2i_index
from Nexus.training.embedder.recommendation.modeling import BaseRetriever

import faiss
import redis
import os
import time
import re
import importlib 
from collections import defaultdict
from collections.abc import Iterable
import argparse
from copy import deepcopy 
from pandas import DataFrame

from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import onnxruntime as ort
import yaml
import json 
import redis 

import onnx
import onnxruntime as ort

import tensorrt as trt
import pycuda.driver as cuda




class BaseEmbedderInferenceEngine(InferenceEngine):

    def __init__(self, config: dict) -> None:
        # super().__init__(config)
        import pycuda.autoinit
        # load config 
        self.config = config
        with open(os.path.join(config['model_ckpt_path'], 'model_config.json'), 'r', encoding='utf-8') as f:
            self.model_ckpt_config = json.load(f)
        self.feature_config = deepcopy(self.model_ckpt_config['data_config'])
        with open(config['feature_cache_config_path'], 'r') as f:
            self.feature_cache_config = yaml.safe_load(f)

        # load cache protos 
        self.key_temp2proto = {}
        for key_temp, proto_dict in self.feature_cache_config['key_temp2proto'].items():
            proto_spec = importlib.util.spec_from_file_location(
                f'{key_temp}_proto_module', proto_dict['module_path'])
            proto_module = importlib.util.module_from_spec(proto_spec)
            proto_spec.loader.exec_module(proto_module)

            self.key_temp2proto[key_temp] = getattr(proto_module, proto_dict['class_name'])

        # connect to redis for feature cache
        self.redis_client = redis.Redis(host=self.feature_cache_config['host'], 
                                        port=self.feature_cache_config['port'], 
                                        db=self.feature_cache_config['db'],
                                        password=os.getenv('REDIS_PW'))
        
        # put seq into context_features
        # self.feature_config is deepcopy of self.model_ckpt_config['data_config']
        if 'seq_features' in self.model_ckpt_config['data_config']:
            self.feature_config['context_features'].append(self.model_ckpt_config['data_config']['seq_features'])
        
        # load model session
        if config['infer_mode'] == 'normal':
            self.model = self.load_model(config['model_ckpt_path'])
        if config['infer_mode'] == 'ort':
            self.convert_to_onnx()
            self.ort_session = self.get_ort_session()
            print(f'Session is using : {self.ort_session.get_providers()}')
        if config['infer_mode'] == 'trt':
            self.convert_to_onnx()
            self.engine = self.get_trt_session()    
        
            
        self.retrieve_index_config = config['retrieve_index_config']

        if config['stage'] == 'retrieve':
            if self.retrieve_index_config.get('gen_item_index', True):
                gen_item_index(os.path.join(config['model_ckpt_path'], 'item_vectors.pt'), 
                            self.retrieve_index_config['item_index_path'],
                            self.retrieve_index_config['item_ids_path'])
            if config['retrieval_mode'] == 'u2i':
                self.item_index = faiss.read_index(self.retrieve_index_config['item_index_path'])
                self.res = faiss.StandardGpuResources()  
                self.item_index = faiss.index_cpu_to_gpu(self.res, 0, self.item_index) 
                self.item_index.nprobe = self.retrieve_index_config['nprobe']
                self.item_ids_table = np.load(self.retrieve_index_config['item_ids_path'])
            elif config['retrieval_mode'] == 'i2i':
                if self.retrieve_index_config.get('gen_i2i_index', True):
                    gen_i2i_index(config['output_topk'],
                                config['model_ckpt_path'], 
                                self.retrieve_index_config['i2i_redis_host'],
                                self.retrieve_index_config['i2i_redis_port'],
                                self.retrieve_index_config['i2i_redis_db'],
                                item_index_path=self.retrieve_index_config['item_index_path'])
                self.i2i_redis_client = redis.Redis(host=self.retrieve_index_config['i2i_redis_host'], 
                                    port=self.retrieve_index_config['i2i_redis_port'], 
                                    db=self.retrieve_index_config['i2i_redis_db'],
                                    password=os.getenv('REDIS_PW'))
        self.get_features_time = 0
        self.model_time = 0
        self.faiss_time = 0
                
    def batch_inference(self, batch_infer_df:DataFrame):
        """
        Perform batch inference for a given batch of data.
        Args:
            batch_infer_df (DataFrame): A pandas DataFrame containing the batch of data to perform inference on.

        Returns:
            (np.ndarray):
                shape [batch_size, output_topk], the recommended items based on the provided inputs.
                - If retrieval_mode is 'u2i', returns a list of item IDs corresponding to the top-k recommendations for each user.
                - If retrieval_mode is 'i2i', returns a list of lists of item IDs representing the recommendations for each sequence of video IDs.
        """
        # iterate infer data 
        time1 = time.time_ns()

        # get user_context features 
        batch_user_context_dict = self.get_user_context_features(batch_infer_df)
        
        feed_dict = {}
        feed_dict.update(batch_user_context_dict)
        for key in feed_dict:
            feed_dict[key] = np.array(feed_dict[key])
        
        time2 = time.time_ns()

        if self.config['retrieval_mode'] == 'u2i':
            if self.config['infer_mode'] == 'normal':
                self.model.to(self.config['infer_device'])
                for key, value in batch_user_context_dict.items():
                    if isinstance(value, dict):
                        batch_user_context_dict[key] = {sub_key: torch.tensor(sub_value).to(self.config['infer_device']) for sub_key, sub_value in value.items()}
                    else:
                        batch_user_context_dict[key] = torch.tensor(value).to(self.config['infer_device'])
                batch_user_embedding = self.model.encode_query(batch_user_context_dict).to('cpu')
                
            elif self.config['infer_mode'] == 'ort':
                batch_user_embedding = self.ort_session.run(
                    output_names=["output"],
                    input_feed=feed_dict
                )[0]
            elif self.config['infer_mode'] == 'trt':
                batch_user_embedding = self.infer_with_trt(feed_dict)

            user_embedding_np = batch_user_embedding[:batch_infer_df.shape[0]]
            # user_embedding_noise = np.random.normal(loc=0.0, scale=0.01, size=user_embedding_np.shape)
            # user_embedding_np = user_embedding_np + user_embedding_noise
            time3 = time.time_ns()
            D, I = self.item_index.search(user_embedding_np, self.config['output_topk'])
            batch_outputs = I

            time4 = time.time_ns()
        elif self.config['retrieval_mode'] == 'i2i':
            seq_video_ids = feed_dict['seq_video_id']
            batch_outputs = self.get_i2i_recommendations(seq_video_ids)
        
            time3 = time.time_ns()
        if self.cumsum_flag:
            self.get_features_time += (time2 - time1) / 1_000_000_000
            self.model_time += (time3 - time2) / 1_000_000_000
            self.faiss_time += (time4 - time3) / 1_000_000_000
            print(f'get features time: {(time2 - time1) / 1_000_000_000}s')
            print(f'model time: {(time3 - time2) / 1_000_000_000}s')
            print(f'faiss time: {(time4 - time3) / 1_000_000_000}s')

        # print(batch_outputs.shape)
        # print(f'batch time: {batch_ed_time - batch_st_time}s')

        if self.config['retrieval_mode'] == 'u2i':
            return self.item_ids_table[batch_outputs]
        else:
            return batch_outputs

    
    def infer_with_trt(self, inputs):

        with self.engine.create_execution_context() as context:
            stream = cuda.Stream()
            bindings = [0] * self.engine.num_io_tensors

            input_memory = []
            output_buffers = {}
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    if -1 in tuple(self.engine.get_tensor_shape(tensor_name)):  # dynamic
                        context.set_input_shape(tensor_name, tuple(self.engine.get_tensor_profile_shape(tensor_name, 0)[2]))
                    input_mem = cuda.mem_alloc(inputs[tensor_name].nbytes)
                    bindings[i] = int(input_mem)
                    context.set_tensor_address(tensor_name, int(input_mem))
                    cuda.memcpy_htod_async(input_mem, inputs[tensor_name], stream)
                    input_memory.append(input_mem)
                else:  # output
                    shape = tuple(context.get_tensor_shape(tensor_name))
                    print(tensor_name,'shape:',shape)
                    output_buffer = np.empty(shape, dtype=dtype)
                    output_buffer = np.ascontiguousarray(output_buffer)
                    output_memory = cuda.mem_alloc(output_buffer.nbytes)
                    bindings[i] = int(output_memory)
                    context.set_tensor_address(tensor_name, output_memory)
                    cuda.memcpy_htod_async(output_memory, output_buffer, stream)
                    output_buffers[tensor_name] = (output_buffer, output_memory)

            context.execute_async_v3(stream_handle=stream.handle)
            stream.synchronize()

            for tensor_name, (output_buffer, output_memory) in output_buffers.items():
                cuda.memcpy_dtoh(output_buffer, output_memory)
        
        
        return output_buffers['output'][0]
    
    
    def get_i2i_recommendations(self, seq_video_ids_batch):
        pipeline = self.i2i_redis_client.pipeline()
        for seq_video_ids in seq_video_ids_batch:
            for video_id in seq_video_ids:
                redis_key = f'item:{video_id}'
                pipeline.get(redis_key)
        results = pipeline.execute()

        all_top10_items = []

        for result in results:
            if result:
                top10_items = result.decode('utf-8').split(',')
                all_top10_items.extend(top10_items)
            else:
                print('Redis returned None for a key')

        all_top10_items = list(map(int, all_top10_items))
        all_top10_items = np.array(all_top10_items).reshape(seq_video_ids_batch.shape[0], -1)

        return all_top10_items

    def convert_to_onnx(self):
        model = BaseRetriever.from_pretrained(self.config['model_ckpt_path'])
        
        model.eval()
        model.forward = model.encode_query

        input_names = []
        dynamic_axes = {} 

        B = self.config['infer_batch_size']

        # user/context input
        context_input = {}
        for feat in self.feature_config['context_features']:
            if isinstance(feat, str):
                context_input[feat] = torch.randint(self.feature_config['stats'][feat], (B,))
                input_names.append(feat)
                dynamic_axes[feat] = {0: "batch_size"}
            elif isinstance(feat, dict): # for nested features
                for feat_name, sub_feat_list in feat.items():
                    sub_context_input = {}
                    for sub_feat in sub_feat_list:
                        if feat_name in self.feature_config['seq_features']:
                            sub_context_input[sub_feat] = torch.randint(self.feature_config['stats'][sub_feat], 
                                                                        (B, self.feature_config['seq_lengths'][feat_name]))
                        else:
                            sub_context_input[sub_feat] = torch.randint(self.feature_config['stats'][sub_feat], (B,))
                        input_names.append(feat_name + '_' + sub_feat)
                        dynamic_axes[feat_name + '_' + sub_feat] = {0: "batch_size"}
                    context_input[feat_name] = sub_context_input

        model_onnx_path = os.path.join(self.config['model_ckpt_path'], 'model_onnx.pb')
        torch.onnx.export(
            model,
            (context_input, {}), 
            model_onnx_path,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=15,
            verbose=False
        )
        
    def get_user_context_features(self, batch_infer_df: DataFrame):
        '''
        get user and context features from redis
        Args:
            batch_infer_df: pd.DataFrame
        Returns:
            user_context_dict: dict
        '''
        # batch_infer_df : [B, M] 
        '''
        context_features:
        [
            "user_id",
            "device_id",
            "age",
            "gender",
            "province", 
            {"user_seq_effective_50" : ["video_id", "author_id", "category_level_two", "category_level_one", "upload_type"]}
        ]
        ''' 
        
        batch_infer_df = batch_infer_df.rename(mapper=(lambda col: col.strip(' ')), axis='columns')
        
        # user and context side features 
        context_features = [sub_feat for feat in self.feature_config['context_features'] 
                            for sub_feat in (list(feat.keys()) if isinstance(feat, dict) else [feat])]
        
        no_redis_features = []
        for feat in context_features:
            if feat not in self.feature_cache_config['features']:
                no_redis_features.append(feat)

        context_features = [feat for feat in context_features if feat not in no_redis_features]


        # user_context_dict = self._row_get_features(
        user_context_dict = self._row_get_features_new(
            batch_infer_df, 
            context_features, 
            [self.feature_cache_config['features'][feat] for feat in context_features],
            no_redis_features)
        
        # expand features with nested keys
        for feat in self.feature_config['context_features']:
            if isinstance(feat, dict):
                for feat_name, feat_fields in feat.items():
                    cur_dict = defaultdict(list) 
                    if isinstance(user_context_dict[feat_name][0], Iterable):
                        # the case is for sequence features 
                        for proto_seq in user_context_dict[feat_name]:
                            for field in feat_fields: 
                                cur_list = [getattr(proto, field) for proto in proto_seq]
                                cur_dict[field].append(cur_list)
                        
                        del user_context_dict[feat_name]
                        if self.config['infer_mode'] == 'normal':
                            user_context_dict[feat_name] = {}
                            for field in feat_fields:
                                user_context_dict[feat_name][field] = cur_dict[field]
                        else:
                            for field in feat_fields:
                                user_context_dict[feat_name + '_' + field] = cur_dict[field]
                    else:
                        for proto in user_context_dict[feat_name]:
                            for field in feat_fields:
                                cur_dict[field].append(getattr(proto, field))
                        
                        del user_context_dict[feat_name]
                        if self.config['infer_mode'] == 'normal':
                            user_context_dict[feat_name] = {}
                            for field in feat_fields:
                                user_context_dict[feat_name][field] = cur_dict[field]
                        else:
                            for field in feat_fields:
                                user_context_dict[feat_name + '_' + field] = cur_dict[field]
        
        return user_context_dict
    
    def _row_get_features(self, row_df:pd.DataFrame, feats_list, feats_cache_list, no_redis_features=None):
        # each row represents one entry 
        # row_df: [B, M]
        res_dict = defaultdict(list)
        # key_temp list 
        feats_key_temp_list = list(set([cache['key_temp'] for cache in feats_cache_list]))
        
        # get all keys and values related to these rows in one time 
        with self.redis_client.pipeline() as pipe:
            feats_all_key_and_temp = set()
            for row in row_df.itertuples():
                for key_temp in feats_key_temp_list:
                    key_feats = re.findall('{(.*?)}', key_temp) 
                    cur_key = key_temp
                    for key_feat in key_feats:
                        cur_key = cur_key.replace(f'{{{key_feat}}}', str(getattr(row, key_feat)))
                    feats_all_key_and_temp.add((cur_key, key_temp))
            feats_all_key_and_temp = list(feats_all_key_and_temp)

            redis_st_time = time.time()
            for key, _ in feats_all_key_and_temp:
                pipe.get(key)
            feats_all_values = pipe.execute()
            redis_ed_time = time.time()
            # print(f'redis time : {(redis_ed_time - redis_st_time)}s')
        
        parse_st_time = time.time()
        feats_k2p = {}
        for (key, key_temp), value in zip(feats_all_key_and_temp, feats_all_values):
            value_proto = self.key_temp2proto[key_temp]()
            value_proto.ParseFromString(value)
            feats_k2p[key] = value_proto
        parse_ed_time = time.time()
        # print(f'parse time : {(parse_ed_time - parse_st_time)}s')

        # get feats from these values
        for row in row_df.itertuples():
            cur_all_values = dict()
            # get all protos that this row needs 
            for key_temp in feats_key_temp_list:
                key_feats = re.findall('{(.*?)}', key_temp) 
                cur_key = key_temp
                for key_feat in key_feats:
                    cur_key = cur_key.replace(f'{{{key_feat}}}', str(getattr(row, key_feat)))
                cur_all_values[key_temp] = feats_k2p[cur_key]

            if no_redis_features is not None:
                for feat in no_redis_features:
                    res_dict[feat].append(getattr(row, feat))

            # get features from these protos 
            for feat, cache in zip(feats_list, feats_cache_list):
                res_dict[feat].append(getattr(cur_all_values[cache['key_temp']], cache['field']))
        
        return res_dict
    

    def _row_get_features_new(self, row_df:pd.DataFrame, feats_list, feats_cache_list, no_redis_features=None):
        # each row represents one entry 
        # row_df: [B, M]
        res_dict = defaultdict(list)
        feats_key_temp_list = list(set([cache['key_temp'] for cache in feats_cache_list]))
       
        with self.redis_client.pipeline() as pipe:
            feats_all_key_and_temp = set()

            key_temp_vars = {}
            for key_temp in feats_key_temp_list:
                key_temp_vars[key_temp] = re.findall('{(.*?)}', key_temp)
            for row in row_df.itertuples():
                # print("row:", row)
                for key_temp in feats_key_temp_list:
                    cur_key = key_temp
                    # print("key_temp_vars[key_temp]", key_temp_vars[key_temp])
                    for key_feat in key_temp_vars[key_temp]:
                        cur_key = cur_key.replace(f'{{{key_feat}}}', str(getattr(row, key_feat)))
                    feats_all_key_and_temp.add((cur_key, key_temp))

            feats_all_key_and_temp = list(feats_all_key_and_temp)

            redis_st_time = time.time_ns()

            # 批量获取Redis值
            keys_to_get = [key for key, _ in feats_all_key_and_temp]
            with self.redis_client.pipeline() as pipe:
                pipe.mget(keys_to_get)
                feats_all_values = pipe.execute()[0]  # 注意mget返回列表

            redis_ed_time = time.time_ns()
        
        
        feats_k2p = {}
        all_fields = set(cache['field'] for cache in feats_cache_list)

        for (key, key_temp), value in zip(feats_all_key_and_temp, feats_all_values):
            value_proto = self.key_temp2proto[key_temp]()
            value_proto.ParseFromString(value)
            feats_k2p[key] = {field: getattr(value_proto, field, None) for field in all_fields}

        
        for row in row_df.itertuples():
            cur_all_values = dict()

            for key_temp in feats_key_temp_list:
                cur_key = key_temp
                for key_feat in key_temp_vars[key_temp]:
                    cur_key = cur_key.replace(f'{{{key_feat}}}', str(getattr(row, key_feat)))
                cur_all_values[key_temp] = feats_k2p[cur_key]

            if no_redis_features is not None:
                for feat in no_redis_features:
                    res_dict[feat].append(getattr(row, feat))

            # get features from these protos 
            for feat, cache in zip(feats_list, feats_cache_list):
                res_dict[feat].append(cur_all_values[cache['key_temp']][cache['field']])
        return res_dict

    
    def get_ort_session(self) -> ort.InferenceSession:
        model_onnx_path = os.path.join(self.config['model_ckpt_path'], 'model_onnx.pb')
        # print graph
        onnx_model = onnx.load(model_onnx_path)
        onnx.checker.check_model(onnx_model)
        print("=" * 25 + 'comp graph : ' + "=" * 25)
        print(onnx.helper.printable_graph(onnx_model.graph))

        if self.config['infer_device'] == 'cpu':
            providers = ["CPUExecutionProvider"]
        elif isinstance(self.config['infer_device'], int):
            providers = [("CUDAExecutionProvider", {"device_id": self.config['infer_device']})]
        return ort.InferenceSession(model_onnx_path, providers=providers)
    
    
    def get_trt_session(self):
        model_onnx_path = os.path.join(self.config['model_ckpt_path'], 'model_onnx.pb')
        trt_engine_path = os.path.join(self.config['model_ckpt_path'], 'model_trt.engine')

        # Set the GPU device
        self.context = cuda.Device(self.config['infer_device']).make_context()

        # Build or load the engine
        if not os.path.exists(trt_engine_path):
            serialized_engine = self.build_engine(model_onnx_path, trt_engine_path)
            TRT_LOGGER = trt.Logger(trt.Logger.INTERNAL_ERROR)
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(serialized_engine)
        else:
            engine = self.load_engine(trt_engine_path)

        return engine
    
    def load_engine(self, engine_file_path):
        TRT_LOGGER = trt.Logger(trt.Logger.INTERNAL_ERROR)
        with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    def build_engine(self, onnx_file_path, engine_file_path):
        TRT_LOGGER = trt.Logger(trt.Logger.INTERNAL_ERROR)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            # Parse ONNX model
            with open(onnx_file_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    raise RuntimeError('Failed to parse ONNX model')

            # Create builder config
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

            # Create optimization profile
            profile = builder.create_optimization_profile()
            for i in range(network.num_inputs):
                input_name = network.get_input(i).name
                input_shape = network.get_input(i).shape
                # Set the min, opt, and max shapes for the input
                min_shape = [1 if dim == -1 else dim for dim in input_shape]
                opt_shape = [self.config['infer_batch_size'] if dim == -1 else dim for dim in input_shape]
                max_shape = [self.config['infer_batch_size'] if dim == -1 else dim for dim in input_shape]
                
                for dim in input_shape:
                    if dim == -1:
                        print('input_name:', input_name, 'input_shape:', input_shape, 'opt_shape:', opt_shape, 'max_shape:', max_shape)
                profile.set_shape(input_name, tuple(min_shape), tuple(opt_shape), tuple(max_shape))
                
            config.add_optimization_profile(profile)

            # Build and serialize the engine
            serialized_engine = builder.build_serialized_network(network, config)
            with open(engine_file_path, 'wb') as f:
                f.write(serialized_engine)
            return serialized_engine
        
    
    def load_model(self, model_path) -> BaseRetriever:
        retriever = None
        retriever = BaseRetriever.from_pretrained(model_path)
        checkpoint = torch.load(os.path.join(model_path, 'model.pt'), map_location=torch.device('cpu'), weights_only=True)
        retriever.load_state_dict(checkpoint)
        retriever.eval()
        return retriever