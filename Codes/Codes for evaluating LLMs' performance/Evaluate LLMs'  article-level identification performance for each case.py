# 评价每道题的得分 Wilcoxon
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import re
import os

# 定义输入和输出文件夹路径
input_folder_path = r"C:\Users\User\Desktop\Law_data_Final\1_直接训练"
output_folder_path = r"C:\Users\User\Desktop\Law_data_Final\1_直接训练结果w"

# 定义清洗标签数据的函数
def clean_tags(tags):
    cleaned_tags = []
    for tag in tags:
        # 去除多余符号和空白字符
        cleaned_tag = re.sub(r"[\[\]']", "", tag).strip()
        cleaned_tags.append(cleaned_tag)
    return cleaned_tags

# 定义计算 F1 Score、Precision 和 Recall 的函数
def calculate_metrics(row):
    true_labels = row["实际值"]
    predicted_labels = row["预测值"]

    # 二值化标签
    all_labels = list(set(true_labels + predicted_labels))
    y_true = [1 if label in true_labels else 0 for label in all_labels]
    y_pred = [1 if label in predicted_labels else 0 for label in all_labels]

    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return pd.Series({"每题F1得分": f1, "每题Precision": precision, "每题Recall": recall})

# 遍历输入文件夹中的所有 Excel 文件
for filename in os.listdir(input_folder_path):
    if filename.endswith(".xlsx"):
        input_file_path = os.path.join(input_folder_path, filename)
        output_file_path = os.path.join(output_folder_path, filename)

        # 读取 Excel 文件
        df = pd.read_excel(input_file_path)

        # 应用清洗函数到“实际值”和“预测值”列
        df["实际值"] = df["实际值"].apply(lambda x: clean_tags(eval(x)) if pd.notnull(x) else [])
        df["预测值"] = df["预测值"].apply(lambda x: clean_tags(eval(x)) if pd.notnull(x) else [])

        # 计算每一行的 F1 Score、Precision 和 Recall，并写入相应的列
        df[["每题F1得分", "每题Precision", "每题Recall"]] = df.apply(calculate_metrics, axis=1)

        # 保存处理后的结果到输出文件夹
        df.to_excel(output_file_path, index=False)

        print(f"Processed file saved to: {output_file_path}")

print("Batch processing completed.")