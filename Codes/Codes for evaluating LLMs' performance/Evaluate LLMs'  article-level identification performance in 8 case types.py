import os
import pandas as pd
import re
from glob import glob

from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

# 定义输入输出文件夹路径
input_folder_path = r"C:\Users\User\Desktop\Law_data_Final\2_二阶段结果"
output_folder_path = r"C:\Users\User\Desktop\Law_data_Final\2_分阶段结果"

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder_path, exist_ok=True)

# 定义关键词列表
keywords = [
    "项目决策和用地获取阶段", "招标与投标阶段", "勘察设计阶段", "施工阶段",
    "竣工验收和交付阶段", "运营维护阶段", "买卖阶段", "其他"
]

# 获取所有Excel文件路径
excel_files = glob(os.path.join(input_folder_path, "*.xlsx"))

# 创建一个空的DataFrame用于汇总结果
summary_df = pd.DataFrame(columns=["File_Name", "Keyword", "Precision", "Recall", "得分"])

# 遍历所有Excel文件进行处理
for file_path in excel_files:
    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 分割文件名，用于创建子文件夹
    file_name = os.path.basename(file_path)
    folder_name = os.path.splitext(file_name)[0]  # 去掉扩展名，用文件名作为子文件夹名
    subfolder_path = os.path.join(output_folder_path, folder_name)

    # 创建子文件夹（如果不存在）
    os.makedirs(subfolder_path, exist_ok=True)

    # 遍历每个分类（关键词）
    for keyword in keywords:
        # 筛选数据
        filtered_df = df[df["分类"].str.contains(keyword, na=False)]

        # 如果筛选出的数据为空，则跳过
        if filtered_df.empty:
            continue

        # 保存分类后的数据
        output_file = os.path.join(subfolder_path, f"{keyword}.xlsx")
        filtered_df.to_excel(output_file, index=False)
        print(f"导出包含关键词 '{keyword}' 的数据到文件: {output_file}")

        # 真实值和预测值处理逻辑
        true_labels = filtered_df["Correct_Answer"].apply(lambda x: [f'{item.strip()}' for item in str(x).split(",")])
        filtered_df["实际值"] = true_labels

        predicted_labels = filtered_df["Answer3"].apply(lambda x: [f'{item.strip()}' for item in str(x).split(",")])
        filtered_df["预测值"] = predicted_labels

        # 计算多标签分类的性能指标
        mlb = MultiLabelBinarizer()
        true_labels_bin = mlb.fit_transform(true_labels)
        predicted_labels_bin = mlb.transform(predicted_labels)

        precision = precision_score(true_labels_bin, predicted_labels_bin, average="samples", zero_division=0)
        recall = recall_score(true_labels_bin, predicted_labels_bin, average="samples")

        # 添加性能指标到DataFrame
        filtered_df["Precision_f"] = precision
        filtered_df["Recall_f"] = recall
        filtered_df["得分"] = 2* (precision * recall) / (precision + recall)

        # 构建输出文件路径
        output_file_path = os.path.join(subfolder_path, f"{keyword}_{file_name}")

        # 保存结果到Excel
        filtered_df.to_excel(output_file_path, index=False)

        # 将当前文件的汇总信息添加到summary_df
        summary_row = pd.DataFrame([{
            "File_Name": file_name,
            "Keyword": keyword,
            "Precision": precision,
            "Recall": recall,
            "得分": (precision + recall) / 2
        }])
        summary_df = pd.concat([summary_df, summary_row], ignore_index=True)

        print(f"Evaluation results for '{keyword}' saved to: {output_file_path}")

# 保存汇总结果到Excel
summary_output_path = os.path.join(output_folder_path, "summary_results.xlsx")
summary_df.to_excel(summary_output_path, index=False)
print(f"Summary results saved to: {summary_output_path}")

