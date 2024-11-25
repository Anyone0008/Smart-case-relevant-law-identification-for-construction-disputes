import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import torch.nn as nn
import os

device = "cuda"
model_path = "/root/autodl-tmp/Two-stage_act-level_Qwen1.5-14B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参


messages = []
import pandas as pd
import re
import os
import csv
from bs4 import BeautifulSoup
import pandas as pd
#from docx import Document
import uuid
import pandas as pd
import openpyxl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer

def split_correct_answers(text):
    # 定义正则表达式模式
    matches = re.findall(r"《([^》]*)》", text)
    unique_matches = list(set(matches))  # 去重
    formatted_matches = [f"《{unique_matches}》" for unique_matches in unique_matches]
#    return ', '.join(formatted_matches)
    return formatted_matches


def save_df_to_excel(df, file_path, sheet_name):
  # 创建一个Excel写入器对象
  writer = pd.ExcelWriter(file_path)

  # 将DataFrame写入指定工作表
  df.to_excel(writer, sheet_name=sheet_name, index=False)

  # 保存并关闭Excel文件
  writer.close()


def read_excel_column(file_path, sheet_name, column_name):
  # 读取Excel文件
  df = pd.read_excel(file_path, sheet_name=sheet_name)
  # 提取指定列的数据，转换为列表返回
  column_data = df[column_name].tolist()

  return column_data


years=["Table S3 The test dataset for construction case-relevant article-level law identification"]
for year in years:
  print("Year",year)
  Questions=read_excel_column(year+".xlsx", "Sheet1", "Questions for act-level identification LLMs")
  Questions_part2=read_excel_column(year+".xlsx", "Sheet1", "Questions for article-level identification LLMs")
  Answers=read_excel_column(year+".xlsx", "Sheet1", "Cited acts")
  Answers_part2=read_excel_column(year+".xlsx", "Sheet1", "Cited articles with specific content")

  df = pd.DataFrame(columns=["Question","Correct_Answer",'Answer1', "Answer2", "信息2", "Correct_Answer_part2"])


  for i in range(len(Questions)):
    message = Questions[i]
    history=[]
    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": message}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    answers_from_model = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    answer0 = answers_from_model
    answer1 = split_correct_answers(answer0)
  
    df2 = pd.DataFrame([
        #{"Question":Questions[i],"Correct_Answer":Answers[i],"Answer1":answer1,"Answer2":answer0,"Score":answer2}
        {"Question":Questions[i],"Correct_Answer":Answers[i],"Answer1":answer0,"Answer2":answer1, "信息2":Questions_part2[i], "Correct_Answer_part2":Answers_part2[i]}
    ])
    
    print("No Question",i+1,"\n","Right_Answer:",Answers[i],"\nAnswer_from_Two-stage_act-level_Qwen1.5-14B-Chat:",answer0)

    df = pd.concat([df, df2], axis=0)
  df["信息2"] = df.apply(lambda row: f"{row['信息2']}请根据以上法律事实，在以下法律中推荐该案件所适用的法律具体条目，禁止胡乱编造：{row['Answer1']}", axis=1)
  save_df_to_excel(df, "Two-stage_act-level_Qwen1.5-14B-Chat.xlsx", "sheet1")
