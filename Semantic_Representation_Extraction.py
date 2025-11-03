import os
import random
import time
import sys
import gc

import numpy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.model_selection import StratifiedKFold


# setting the seed for reproducibility
def set_deterministic(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# specify GPU
device = torch.device("cuda")

# read the parameters
dataset_path = "/dataset/FlakeFlagger/FlakeFlagger_dataset.csv"
output_features_file = "/dataset/CodeBERT_features.csv"

df = pd.read_csv(dataset_path)
input_data = df['final_code']
target_data = df['flaky']

# 移除包含 NaN 的行
nan_indices = input_data[input_data.isna()].index.union(target_data[target_data.isna()].index)
input_data = input_data.drop(nan_indices)
target_data = target_data.drop(nan_indices)

# define CodeBERT model
model_name = "../codebert-base"
model_config = AutoConfig.from_pretrained(model_name, return_dict=False, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_data(text_data):
    tokens = tokenizer.batch_encode_plus(
        text_data.tolist(),
        max_length=510,
        pad_to_max_length=True,
        truncation=True)
    return tokens


def extract_codebert_features(text_data, batch_size=16):
    """
    提取CodeBERT表征特征
    """
    tokens = tokenize_data(text_data)
    input_ids = torch.tensor(tokens['input_ids'])
    attention_mask = torch.tensor(tokens['attention_mask'])

    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 每次重新加载模型以确保独立性
    auto_model = AutoModel.from_pretrained(model_name, config=model_config)
    auto_model.to(device)
    auto_model.eval()

    all_features = []

    with torch.no_grad():
        for batch in dataloader:
            batch = [r.to(device) for r in batch]
            input_ids_batch, attention_mask_batch = batch

            outputs = auto_model(input_ids_batch, attention_mask=attention_mask_batch)
            last_hidden_state = outputs[0]
            cls_features = last_hidden_state[:, 0, :]  # [CLS] token representation

            all_features.append(cls_features.cpu().numpy())

    features_array = np.concatenate(all_features, axis=0)

    # 清理内存
    del auto_model
    torch.cuda.empty_cache()

    return features_array


# 设置随机种子
seed = 42
set_deterministic(seed)

execution_time = time.time()
print("Start time of the experiment", execution_time)

# 使用分层K折交叉验证提取特征
skf = StratifiedKFold(n_splits=10, shuffle=False, random_state=seed)
fold_number = 0

all_fold_features = []

for train_index, test_index in skf.split(input_data, target_data):
    print(f"Processing fold {fold_number}...")

    X_train, X_test = input_data.iloc[list(train_index)], input_data.iloc[list(test_index)]
    y_train, y_test = target_data.iloc[list(train_index)], target_data.iloc[list(test_index)]

    # 为测试集提取特征
    test_features = extract_codebert_features(X_test)

    # 创建测试集特征DataFrame
    feature_columns = [f'codebert_feature_{i}' for i in range(test_features.shape[1])]
    test_features_df = pd.DataFrame(test_features, columns=feature_columns)

    # 添加元数据
    test_features_df['final_code'] = X_test.values
    test_features_df['flaky'] = y_test.values
    test_features_df['original_index'] = X_test.index.values
    test_features_df['fold'] = fold_number
    test_features_df['split'] = 'test'

    # 为训练集提取特征（可选）
    train_features = extract_codebert_features(X_train)
    train_features_df = pd.DataFrame(train_features, columns=feature_columns)
    train_features_df['final_code'] = X_train.values
    train_features_df['flaky'] = y_train.values
    train_features_df['original_index'] = X_train.index.values
    train_features_df['fold'] = fold_number
    train_features_df['split'] = 'train'

    # 合并训练和测试特征
    fold_features_df = pd.concat([train_features_df, test_features_df], ignore_index=True)
    all_fold_features.append(fold_features_df)

    fold_number += 1

# 合并所有折叠的特征
final_features_df = pd.concat(all_fold_features, ignore_index=True)

# 保存到CSV文件
final_features_df.to_csv(output_features_file, index=False)
print(f"CodeBERT features saved to: {output_features_file}")
print(f"Final features shape: {final_features_df.shape}")

print("The process completed in : (%s) seconds. " % round((time.time() - execution_time), 5))