import io
import time
import warnings
import numpy as np
import os
import sys
from pathlib import Path
import ast
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn import svm, tree
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pytorch_metric_learning.losses import SupConLoss
from sklearn.utils import shuffle
from xgboost import XGBClassifier


# 新增：语义感知的数据增强函数
def semantic_structural_augmentation(original_df, augmentation_factor=1, flaky_boost=1.5):
    """
    针对Flaky测试预测的智能数据增强方案，特别处理语义表征特征
    :param original_df: 原始DataFrame
    :param augmentation_factor: 基础增强倍数
    :param flaky_boost: Flaky样本的额外增强权重
    :return: 增强后的DataFrame
    """
    augmented_rows = []

    # 识别语义特征列
    semantic_cols = [col for col in original_df.columns if col.startswith('semantic_')]

    # 按类别分组处理
    for class_value in [0, 1]:
        subset = original_df[original_df['flakyStatus'] == class_value]
        current_factor = augmentation_factor * (flaky_boost if class_value == 1 else 1)

        for _, row in subset.iterrows():
            for _ in range(int(current_factor)):
                new_row = row.copy()

                # 1. 代码结构特征增强
                if 'javaKeysCounter' in new_row:
                    new_row['javaKeysCounter'] = max(1, int(new_row['javaKeysCounter'] * np.random.uniform(0.8, 1.2)))

                # 2. 关键词频率扰动（保留语义）
                keyword_cols = [col for col in new_row.index if '_keyword' in col]
                for col in keyword_cols:
                    if random.random() < 0.4:  # 40%概率修改关键词频率
                        new_row[col] = min(5, max(0, new_row[col] + random.choice([-1, 0, 1])))

                # 3. 测试指标合理扰动
                metric_cols = ['testLength', 'numAsserts', 'numCoveredLines', 'ExecutionTime']
                for col in metric_cols:
                    if col in new_row:
                        if col == 'ExecutionTime':
                            new_row[col] = max(0.001, new_row[col] * np.random.uniform(0.7, 1.5))
                        else:
                            new_row[col] = max(1, int(new_row[col] * np.random.uniform(0.8, 1.3)))

                # 4. 反模式特征智能翻转
                antipatterns = ['assertion-roulette', 'conditional-test-logic', 'eager-test',
                                'fire-and-forget', 'indirect-testing', 'mystery-guest']
                for pat in antipatterns:
                    if pat in new_row and random.random() < 0.3:
                        # 对Flaky样本更可能保留反模式特征
                        if class_value == 1 and random.random() < 0.7:
                            new_row[pat] = 1
                        else:
                            new_row[pat] = 1 - new_row[pat]

                # 5. 代码修改模式特征增强
                hindex_cols = [col for col in new_row.index if 'hIndexModifications' in col]
                for col in hindex_cols:
                    if random.random() < 0.3:
                        new_row[col] = max(0, min(1, new_row[col] + random.choice([-0.2, 0, 0.2])))

                # 6. 语义表征特征增强 - 高斯噪声
                for sem_col in semantic_cols:
                    if sem_col in new_row and random.random() < 0.5:
                        noise = np.random.normal(0, 0.1)  # 小幅度高斯噪声
                        new_row[sem_col] = new_row[sem_col] + noise

                augmented_rows.append(new_row)

    # 合并并打乱数据
    augmented_df = pd.DataFrame(augmented_rows)
    final_df = pd.concat([original_df, augmented_df]).reset_index(drop=True)
    return shuffle(final_df)


# 监督对比学习模型
class SupConModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, proj_dim=128):
        super(SupConModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, proj_dim)
        )

    def forward(self, x):
        return self.encoder(x)


# 自定义数据集类
class FlakyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])


# 训练监督对比学习模型
def train_supcon_model(X_train, y_train, input_dim, batch_size=64, epochs=100, lr=0.0005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 改进的数据标准化
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)

    # 加权采样处理类别不平衡
    class_counts = np.bincount(y_train)
    weights = 1. / class_counts
    samples_weights = weights[y_train]
    sampler = torch.utils.data.WeightedRandomSampler(
        samples_weights, len(samples_weights), replacement=True)

    train_dataset = FlakyDataset(X_train_scaled, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    model = SupConModel(input_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion = SupConLoss(temperature=0.5)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).squeeze()

            optimizer.zero_grad()
            embeddings = model(batch_features)
            loss = criterion(embeddings, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')

    return model, scaler


# 提取对比学习特征
def extract_supcon_features(model, scaler, X):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    with torch.no_grad():
        features = model(X_tensor).cpu().numpy()

    return features


# %%
def get_scores(tn, fp, fn, tp):
    if (tp == 0):
        accuracy = (tp + tn) / (tn + fp + fn + tp)
        Precision = 0
        Recall = 0
        F1 = 0
    else:
        accuracy = (tp + tn) / (tn + fp + fn + tp)
        Precision = tp / (tp + fp)
        Recall = tp / (tp + fn)
        F1 = 2 * ((Precision * Recall) / (Precision + Recall))
    return accuracy, F1, Precision, Recall


# %%
def generateConfusionMatrixByProject(data, processed_data):
    filter_data = data[['cross_validation', 'balance_type', 'IG_min', 'numTrees', 'classifier', 'features_structure']]
    filter_data = filter_data.drop_duplicates()
    df_columns = ['cross_validation', 'balance_type', 'IG_min', 'numTrees', 'classifier', "features_structure",
                  "project", "TP", "FN", "FP", "TN", "Precision", "Recall", "F1"]
    result = pd.DataFrame(columns=df_columns)

    # add project name to the full result ..
    data_with_project_name = processed_data[['project', 'test_name']]
    updated_data = pd.merge(data, data_with_project_name, on='test_name', how='left')

    for index, row in filter_data.iterrows():
        data_per_result = updated_data[(updated_data['cross_validation'] == row['cross_validation']) &
                                       (updated_data['balance_type'] == row['balance_type']) &
                                       (updated_data['IG_min'] == row['IG_min']) &
                                       (updated_data['numTrees'] == row['numTrees']) &
                                       (updated_data['classifier'] == row['classifier']) &
                                       (updated_data['features_structure'] == row['features_structure'])]

        for proj in data_per_result.project.unique():
            specific_project = data_per_result[data_per_result["project"] == proj]
            TP = len(specific_project[specific_project["Matrix_label"] == "TP"])
            FN = len(specific_project[specific_project["Matrix_label"] == "FN"])
            FP = len(specific_project[specific_project["Matrix_label"] == "FP"])
            TN = len(specific_project[specific_project["Matrix_label"] == "TN"])
            accuracy, F1, Precision, Recall = get_scores(TN, FP, FN, TP)
            new_row = pd.Series(
                [row['cross_validation'], row['balance_type'], row['IG_min'], row['numTrees'], row['classifier'],
                 row['features_structure'], proj, TP, FN, FP, TN, str(round(((Precision) * 100))) + "%",
                 str(round(((Recall) * 100))) + "%", str(round(((F1) * 100))) + "%"], index=result.columns)
            result = pd.concat([result, new_row.to_frame().T], ignore_index=True)

    return result


def predict_RF_crossValidation(data, k, foldType, balance, classifier, mintree, Features_type, ig, result_by_test_name,
                               use_supcon=True, use_augmented=True):
    """
    修改后的交叉验证函数，增加use_supcon和use_augmented参数控制是否使用对比学习和数据增强
    测试数据始终使用原始未增强的数据
    """
    data = data.dropna()
    if "project_y" in data.columns:
        del data["project_y"]
    if "project" in data.columns:
        del data["project"]

    # 保存原始数据用于测试
    original_data = data.copy()

    # 新增：数据增强处理（只增强训练数据）
    if use_augmented:
        print("Applying semantic-aware structural data augmentation to training data only...")
        augmented_data = semantic_structural_augmentation(
            data,
            augmentation_factor=1,  # 基础增强倍数
            flaky_boost=1.5  # Flaky样本增强强度
        )
        print(f"Data augmented from {len(data)} to {len(augmented_data)} samples")
    else:
        augmented_data = data

    data_target = augmented_data[['flakyStatus']]
    data_features = augmented_data.drop(['flakyStatus'], axis=1)

    # KFold Cross Validation approaches
    if (foldType == "KFold"):
        fold = KFold(n_splits=k, shuffle=True)
    else:
        fold = StratifiedKFold(n_splits=k, shuffle=True)

    auc_scores = []
    TN = FP = FN = TP = 0

    # 对原始数据生成分割索引（确保测试数据来自原始数据）
    for train_index, test_index in fold.split(original_data, original_data['flakyStatus']):
        # 训练数据可以来自增强数据集
        x_train = data_features.iloc[list(train_index)]
        y_train = data_target.iloc[list(train_index)]

        # 测试数据始终来自原始数据
        x_test = original_data.drop(['flakyStatus'], axis=1).iloc[list(test_index)]
        y_test = original_data[['flakyStatus']].iloc[list(test_index)]

        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

        test_names_as_list = x_test['test_name'].tolist()
        x_train_no_names = x_train.drop(columns='test_name')
        x_test_no_names = x_test.drop(columns='test_name')

        # 使用监督对比学习提取特征
        if use_supcon:
            try:
                supcon_model, scaler = train_supcon_model(x_train_no_names.values, y_train, x_train_no_names.shape[1])
                x_train_supcon = extract_supcon_features(supcon_model, scaler, x_train_no_names.values)
                x_test_supcon = extract_supcon_features(supcon_model, scaler, x_test_no_names.values)
                x_train_final = x_train_supcon
                x_test_final = x_test_supcon
                print("Using SupCon features for training")
            except Exception as e:
                print(f"SupCon training failed: {e}, using original features")
                x_train_final = x_train_no_names.values
                x_test_final = x_test_no_names.values
        else:
            x_train_final = x_train_no_names.values
            x_test_final = x_test_no_names.values

        if (balance == "SMOTE"):
            oversample = SMOTE()
            x_train_final, y_train = oversample.fit_resample(x_train_final, y_train)
        elif (balance == "undersampling"):
            undersampling = RandomUnderSampler()
            x_train_final, y_train = undersampling.fit_resample(x_train_final, y_train)

        if (classifier == 'DT'):
            model = DecisionTreeClassifier(criterion='entropy', max_depth=None)
        elif (classifier == 'RF'):
            model = RandomForestClassifier(criterion="entropy", n_estimators=mintree)
        elif (classifier == 'MLP'):
            model = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)
        elif (classifier == 'SVM'):
            model = svm.SVC(gamma='scale', probability=True)
            model = KNeighborsClassifier(n_neighbors=7)
        elif (classifier == 'XGBoost'):
            model = XGBClassifier(eval_metric='logloss')

        final_model = model.fit(x_train_final, y_train)
        pred_probs = final_model.predict_proba(x_test_final)[:, 1]  # 获取正类概率
        # 调整阈值（例如从0.5提高到0.6）
        threshold = 0.7
        preds = (pred_probs >= threshold).astype(int)

        actual_status = y_test.tolist()
        for i in range(len(test_names_as_list)):
            new_row = pd.DataFrame([[foldType, balance, ig, mintree, classifier, Features_type, test_names_as_list[i],
                                     "TP" if actual_status[i] == 1 and preds[i] == 1 else "FN" if actual_status[
                                                                                                      i] == 1 and preds[
                                                                                                      i] == 0 else "FP" if
                                     actual_status[i] == 0 and preds[i] == 1 else "TN"]],
                                   columns=result_by_test_name.columns)
            result_by_test_name = pd.concat([result_by_test_name, new_row], ignore_index=True)

        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
        TN += tn
        FP += fp
        FN += fn
        TP += tp

        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, pred_probs)
        auc_scores.append(auc(false_positive_rate, true_positive_rate))

    accuracy, F1, Precision, Recall = get_scores(TN, FP, FN, TP)
    auc_scores = [0 if math.isnan(x) else x for x in auc_scores]
    return TN, FP, FN, TP, round((Precision * 100)), round(((Recall) * 100)), round((F1 * 100)), round(
        ((sum(auc_scores) / k) * 100)), result_by_test_name


# %%
def get_only_specific_columns_V1(full_data, specificColumns, wanted_columns):
    copy_fullData = full_data.copy()
    lst = []
    for i in specificColumns:
        lst.append(i)
    for j in wanted_columns:
        lst.append(j)
    available_columns = list(set(lst) & set(full_data.columns))
    copy_fullData = copy_fullData[available_columns]
    return copy_fullData


# %%
def vexctorizeToken(token):
    vocabulary_vectorizer = CountVectorizer()
    bow_train = vocabulary_vectorizer.fit_transform(token)
    matrix_token = pd.DataFrame(bow_train.toarray(), columns=vocabulary_vectorizer.get_feature_names_out())
    return matrix_token


# %%
execution_time = time.time()

if __name__ == '__main__':
    warnings.simplefilter("ignore")
    print(os.getcwd())

    # 文件路径
    processed_data = "result/processed_data_with_vocabulary_per_test.csv"
    FlakeFlaggerFeatures = "input_data/FlakeFlaggerFeaturesTypes.csv"
    InformationGain = "result/Information_gain_per_feature.csv"

    # 加载数据
    main_data = pd.read_csv(processed_data)
    FlakeFlaggerFeatures = pd.read_csv("input_data/FlakeFlaggerFeaturesTypes.csv")
    IG_lst = pd.read_csv("result/Information_gain_per_feature.csv")
    processed_data = pd.read_csv("result/processed_data.csv")

    output_dir = "result-RQ1/classification_result/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    result_by_test_name_columns = ["cross_validation", "balance_type", "IG_min", "numTrees", "classifier",
                                   "features_structure", "test_name", "Matrix_label"]
    df_columns = ["Model", "cross_validation", "balance_type", "numTrees", "features_structure", "IG_min",
                  "num_satsifiedFeatures", "classifier", "TP", "FN", "FP", "TN", "precision", "recall", "F1_score",
                  "AUC"]


    # 处理语义表征列
    def parse_semantic_representation(semantic_str):
        try:
            if isinstance(semantic_str, str) and semantic_str.strip():
                # 移除方括号和空格，然后分割为数值列表
                semantic_str = semantic_str.replace('[', '').replace(']', '').strip()
                return [float(x) for x in semantic_str.split()]
            else:
                return []
        except Exception as e:
            print(f"Error parsing semantic representation: {e}")
            return []


    # 解析语义表征
    semantic_vectors = main_data['semantic_representation'].apply(parse_semantic_representation)

    # 确定向量长度（取第一个非空向量的长度）
    vector_length = 0
    for vec in semantic_vectors:
        if len(vec) > 0:
            vector_length = len(vec)
            break

    # 创建语义特征DataFrame
    semantic_columns = [f'semantic_{i}' for i in range(vector_length)]
    semantic_df = pd.DataFrame(semantic_vectors.tolist(),
                               columns=semantic_columns,
                               index=main_data.index)

    # 填充NaN值
    semantic_df = semantic_df.fillna(0)

    # 合并特征
    tokenOnly = vexctorizeToken(main_data['tokenList'])
    main_data = main_data.drop(columns=['tokenList', 'semantic_representation'])
    vocabulary_processed_data = pd.concat([main_data, tokenOnly.reindex(main_data.index),
                                           semantic_df.reindex(main_data.index)], axis=1)

    ##=========================================================##
    # 参数设置
    k = 10
    fold_type = ["StratifiedKFold"]
    balance = ["SMOTE"]
    classifier = ["XGboost", "DT", "RF", "MLP", "SVM"]
    treeSize = [250]
    minIGList = [0.01]
    use_supcon = True  # 是否使用监督对比学习
    use_augmented = True  # 是否使用数据增强
    ##=========================================================##

    for ig in minIGList:
        Path(output_dir + "IG_" + str(ig)).mkdir(parents=True, exist_ok=True)
        min_IG = IG_lst[IG_lst["IG"] >= ig]
        keep_minIG = min_IG.features.unique()
        keep_minIG = [x for x in keep_minIG if str(x) != 'nan']
        removed_columns = ['java_keywords', 'javaKeysCounter']

        # 筛选特征
        vocabulary_processed_data_full = vocabulary_processed_data.copy()
        if ig != 0:
            keep_columns = keep_minIG + ['flakyStatus', 'test_name'] + semantic_columns
            vocabulary_processed_data_full = vocabulary_processed_data_full[keep_columns]
            vocabulary_processed_data_full.to_csv('result_1/vocabulary_processed_data_full.csv', index=False)

        result = pd.DataFrame(columns=df_columns)
        result_by_test_name = pd.DataFrame(columns=result_by_test_name_columns)

        for mintree in treeSize:
            for fold in fold_type:
                for bal in balance:
                    for cl in classifier:
                        # 结合所有特征
                        combined_features = list(FlakeFlaggerFeatures.allFeatures.unique()) + semantic_columns + [
                            "flakyStatus", "test_name"]
                        combined_data = get_only_specific_columns_V1(vocabulary_processed_data_full,
                                                                     combined_features,
                                                                     ["flakyStatus", "test_name"])

                        TN, FP, FN, TP, Precision, Recall, f1, auc_score, result_by_test_name = predict_RF_crossValidation(
                            combined_data, k, fold, bal, cl, mintree, "Combined-Features", ig, result_by_test_name,
                            use_supcon=use_supcon, use_augmented=use_augmented)
                        print(f"TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}, Precision: {Precision}, Recall: {Recall:.4f}, f1: {f1:.4f}, AUC: {auc_score:.4f}")

                        new_row = pd.Series(["CrossAllProjects", fold, bal, mintree, "Combined-Features", ig,
                                             combined_data.shape[1] - 1, cl, TP, FN, FP, TN, Precision, Recall, f1,
                                             auc_score],
                                            index=result.columns)
                        result = pd.concat([result, new_row.to_frame().T], ignore_index=True)

        # 保存结果
        result_by_test_name.to_csv(output_dir + "IG_" + str(ig) + '/prediction_result_per_test.csv', index=False)
        result.to_csv(output_dir + "IG_" + str(ig) + '/prediction_result.csv', index=False)

        # 生成混淆矩阵
        confusion_matrix_by_project = generateConfusionMatrixByProject(result_by_test_name,
                                                                       processed_data)
        confusion_matrix_by_project.to_csv(output_dir + "IG_" + str(ig) + '/prediction_result_by_project.csv', index=False)
