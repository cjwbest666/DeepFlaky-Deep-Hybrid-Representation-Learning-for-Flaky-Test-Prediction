import io
import time
import warnings
import numpy as np
import os
import sys
from pathlib import Path
import ast
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
from xgboost import XGBClassifier

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
            TN = len(processed_data[processed_data["project"] == proj]) - (TP + FN + FP)
            accuracy, F1, Precision, Recall = get_scores(TN, FP, FN, TP)
            new_row = pd.Series(
                [row['cross_validation'], row['balance_type'], row['IG_min'], row['numTrees'], row['classifier'],
                 row['features_structure'], proj, TP, FN, FP, TN, str(round(((Precision) * 100))) + "%",
                 str(round(((Recall) * 100))) + "%", str(round(((F1) * 100))) + "%"], index=result.columns)
            result = pd.concat([result, new_row.to_frame().T], ignore_index=True)

    return result


# %%
def predict_RF_crossValidation(data, k, foldType, balance, classifier, mintree, Features_type, ig, result_by_test_name):
    data = data.dropna()
    if "project_y" in data.columns:
        del data["project_y"]
    if "project" in data.columns:
        del data["project"]
    data_target = data[['flakyStatus']]
    data = data.drop(['flakyStatus'], axis=1)

    # KFold Cross Validation approaches
    if (foldType == "KFold"):
        fold = KFold(n_splits=k, shuffle=True)
    else:
        fold = StratifiedKFold(n_splits=k, shuffle=True)

    auc_scores = []
    TN = FP = FN = TP = 0
    for train_index, test_index in fold.split(data, data_target):
        x_train, x_test = data.iloc[list(train_index)], data.iloc[list(test_index)]
        y_train, y_test = data_target.iloc[list(train_index)], data_target.iloc[list(test_index)]

        # Ensure y_train and y_test are 1D arrays
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

        test_names_as_list = x_test['test_name'].tolist()
        x_train = x_train.drop(columns='test_name')
        x_test = x_test.drop(columns='test_name')

        if (balance == "SMOTE"):
            oversample = SMOTE()
            x_train, y_train = oversample.fit_resample(x_train, y_train)
        elif (balance == "undersampling"):
            undersampling = RandomUnderSampler()
            x_train, y_train = undersampling.fit_resample(x_train, y_train)

        if (classifier == 'RF'):
            model = RandomForestClassifier(criterion="entropy", n_estimators=mintree)
        elif (classifier == 'MLP'):
            model = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=50)
        elif (classifier == 'XGBoost'):  # 添加XGBoost选项
            model = XGBClassifier(eval_metric='logloss')

        final_model = model.fit(x_train, y_train)

        # 修改部分：获取预测概率并根据阈值0.7进行分类
        pred_proba = final_model.predict_proba(x_test)
        threshold = 0.7
        # 获取正类（类别1）的概率
        positive_proba = pred_proba[:, 1]
        # 根据阈值进行分类
        preds = (positive_proba >= threshold).astype(int)

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

        # auc computation and others ..
        # 注意：AUC计算仍然使用概率，不需要修改
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, positive_proba)
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
        lst.append(i)  # 使用 append 将列名添加到列表中
    for j in wanted_columns:
        lst.append(j)  # 同样使用 append
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
# command : python3 cross-all-projects-model-vocabulary.py input_data/data/full_data.csv input_data/FlakeFlaggerFeaturesTypes.csv token_by_IG/IG_vocabulary_and_FlakeFlagger_features.csv

if __name__ == '__main__':
    warnings.simplefilter("ignore")
    print(os.getcwd())

    # 文件路径
    processed_data = "result/processed_data_with_vocabulary_per_test.csv"
    FlakeFlaggerFeatures = "input_data/FlakeFlaggerFeaturesTypes.csv"
    InformationGain = "result/Information_gain_per_feature.csv"

    # vocabulary data _ processed data
    main_data = pd.read_csv(processed_data)
    FlakeFlaggerFeatures = pd.read_csv("input_data/FlakeFlaggerFeaturesTypes.csv")
    IG_lst = pd.read_csv("result/Information_gain_per_feature.csv")
    processed_data = pd.read_csv("result/processed_data.csv")

    output_dir = "result-RQ3/classification_result/"
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
    # arguments
    k = 10
    fold_type = ["StratifiedKFold"]
    balance = ["SMOTE"]
    classifier = ["XGBoost"]
    treeSize = [250]
    minIGList = [0.01]
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
                        # 1. 仅使用FlakeFlagger特征
                        only_processed_data = get_only_specific_columns_V1(vocabulary_processed_data_full,
                                                                           FlakeFlaggerFeatures.allFeatures.unique(),
                                                                           ["flakyStatus", "test_name"])
                        TN, FP, FN, TP, Precision, Recall, f1, auc_score, result_by_test_name = predict_RF_crossValidation(
                            only_processed_data, k, fold, bal, cl, mintree, "Flake-Flagger-Features", ig,
                            result_by_test_name)
                        new_row = pd.Series(["CrossAllProjects", fold, bal, mintree, "Flake-Flagger-Features", ig,
                                             only_processed_data.shape[1] - 1, cl, TP, FN, FP, TN, Precision, Recall,
                                             f1, auc_score],
                                            index=result.columns)
                        result = pd.concat([result, new_row.to_frame().T], ignore_index=True)
                        print(f"TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}, Precision: {Precision}, Recall: {Recall:.4f}, f1: {f1:.4f}, AUC: {auc_score:.4f}")

                        # 2. 仅使用语义表征特征
                        only_semantic_data = vocabulary_processed_data_full[semantic_columns + ['flakyStatus', 'test_name']]
                        TN, FP, FN, TP, Precision, Recall, f1, auc_score, result_by_test_name = predict_RF_crossValidation(
                            only_semantic_data, k, fold, bal, cl, mintree, "Semantic-Features", ig, result_by_test_name)
                        new_row = pd.Series(["CrossAllProjects", fold, bal, mintree, "Semantic-Features", ig,
                                             only_semantic_data.shape[1] - 1, cl, TP, FN, FP, TN, Precision, Recall, f1,
                                             auc_score],
                                            index=result.columns)
                        result = pd.concat([result, new_row.to_frame().T], ignore_index=True)
                        print(f"TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}, Precision: {Precision}, Recall: {Recall:.4f}, f1: {f1:.4f}, AUC: {auc_score:.4f}")


                        # 3. 结合所有特征
                        combined_features = list(FlakeFlaggerFeatures.allFeatures.unique()) + semantic_columns + [
                            "flakyStatus", "test_name"]
                        combined_data = get_only_specific_columns_V1(vocabulary_processed_data_full,
                                                                     combined_features,
                                                                     ["flakyStatus", "test_name"])
                        TN, FP, FN, TP, Precision, Recall, f1, auc_score, result_by_test_name = predict_RF_crossValidation(
                            combined_data, k, fold, bal, cl, mintree, "Combined-Features", ig, result_by_test_name)
                        new_row = pd.Series(["CrossAllProjects", fold, bal, mintree, "Combined-Features", ig,
                                             combined_data.shape[1] - 1, cl, TP, FN, FP, TN, Precision, Recall, f1,
                                             auc_score],
                                            index=result.columns)
                        result = pd.concat([result, new_row.to_frame().T], ignore_index=True)
                        print(f"TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}, Precision: {Precision}, Recall: {Recall:.4f}, f1: {f1:.4f}, AUC: {auc_score:.4f}")

                        # 保存结果
                        result_by_test_name.to_csv(output_dir + "IG_" + str(ig) + '/prediction_result_per_test_without_contrastive_learning.csv',
                                                   index=False)
                        result.to_csv(output_dir + "IG_" + str(ig) + '/prediction_result_without_contrastive_learning.csv', index=False)

                        # 生成混淆矩阵
                        confusion_matrix_by_project = generateConfusionMatrixByProject(result_by_test_name,
                                                                                       processed_data)
                        confusion_matrix_by_project.to_csv(
                            output_dir + "IG_" + str(ig) + '/prediction_result_by_project_without_contrastive_learning.csv', index=False)
