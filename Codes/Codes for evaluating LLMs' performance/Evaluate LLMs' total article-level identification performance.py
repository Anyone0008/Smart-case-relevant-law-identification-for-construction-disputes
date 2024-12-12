# 评价最终结果
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import re

# 定义输入和输出文件夹路径
input_folder = r"C:\Users\User\Desktop\Law_data_Final\1_直接训练"
output_folder = r"C:\Users\User\Desktop\Law_data_Final\1_直接训练1"
os.makedirs(output_folder, exist_ok=True)  # 确保输出文件夹存在


# 定义清洗标签的函数
def clean_tags(tags):
    cleaned_tags = []
    for tag in tags:
        cleaned_tag = re.sub(r"[\[\]']", "", tag)  # 去除多余符号
        cleaned_tags.append(cleaned_tag)
    return ", ".join(cleaned_tags)


# 定义处理单个文件的函数
def process_file(file_path):
    df = pd.read_excel(file_path)

    # 清洗并生成标签列表
    true_labels = df["Correct_Answer"].apply(lambda x: [item.strip() for item in str(x).split(",")])
    predicted_labels = df["Answer3"].apply(lambda x: [item.strip() for item in str(x).split(",")])

    df["实际值"] = true_labels
    df["预测值"] = predicted_labels

    # 计算多标签分类的性能指标
    mlb = MultiLabelBinarizer()
    true_labels_bin = mlb.fit_transform(true_labels)
    predicted_labels_bin = mlb.transform(predicted_labels)

    # 计算样本级别的Precision、Recall和F1分数
    df["Precision"] = precision_score(true_labels_bin, predicted_labels_bin, average="samples")
    df["Recall"] = recall_score(true_labels_bin, predicted_labels_bin, average="samples")
    df["F1 Score"] = 2 * (df["Recall"] * df["Precision"]) / (df["Recall"] + df["Precision"])

    return df


# 批量处理文件夹中的Excel文件
for filename in os.listdir(input_folder):
    if filename.endswith(".xlsx"):
        file_path = os.path.join(input_folder, filename)
        processed_df = process_file(file_path)

        # 导出处理后的文件到指定的输出文件夹
        output_path = os.path.join(output_folder, filename)
        processed_df.to_excel(output_path, index=False)

print("所有文件处理完成，已将结果保存到目标文件夹。")
