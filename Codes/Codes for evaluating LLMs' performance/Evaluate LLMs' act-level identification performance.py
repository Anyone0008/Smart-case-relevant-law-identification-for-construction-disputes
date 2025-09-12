import pandas as pd
import os
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer


# 文件夹路径
source_dir = r"C:\Users\User\Desktop\Law_data_Final\3_法律名推荐"
dest_dir = r"C:\Users\User\Desktop\Law_data_Final\3_法律名推荐1"

# 确保目标文件夹存在
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# 遍历文件夹中的所有文件
for filename in os.listdir(source_dir):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(source_dir, filename)

        # 读取Excel文件
        df = pd.read_excel(file_path)

        # 真实值和预测值
        true_labels = df["Correct_Answer"].apply(lambda x: [f'{item.strip()}' for item in str(x).split(",")])
        df["实际值"] = true_labels
        predicted_labels = df["Answer3"].apply(lambda x: [f'{item.strip()}' for item in str(x).split(",")])
        df["预测值"] = predicted_labels

        # 计算多标签分类的性能指标
        mlb = MultiLabelBinarizer()
        true_labels_bin = mlb.fit_transform(true_labels)
        predicted_labels_bin = mlb.transform(predicted_labels)

        # 计算各项指标
        precision = precision_score(true_labels_bin, predicted_labels_bin, average="samples", zero_division=0)
        recall = recall_score(true_labels_bin, predicted_labels_bin, average="samples")
        f1 = f1_score(true_labels_bin, predicted_labels_bin, average="samples")


        # 添加性能指标到数据框
        df["Precision"] = precision
        df["Recall"] = recall
        df["F1 Score"] = 2* (precision * recall) / (precision + recall)


        # 保存结果到目标文件夹
        output_file_path = os.path.join(dest_dir, filename)
        df.to_excel(output_file_path, index=False)

        print(f"Evaluation results saved for {filename} to: {output_file_path}")

print("批量处理完成，所有结果已保存。")
