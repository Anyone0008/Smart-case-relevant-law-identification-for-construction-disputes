from transformers import AutoTokenizer, AutoModel
import os


tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/Two-stage_article-level_Deepspeek-LLM-7B-Chat", trust_remote_code=True)
model = AutoModel.from_pretrained("/root/autodl-tmp/Two-stage_article-level_Deepspeek-LLM-7B-Chat",  trust_remote_code=True,top_p=0, 
  temperature=0).cuda()


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


years=["Two-stage_act-level_Deepspeek-LLM-7B-Chat"]
for year in years:
  print("Year",year)
  Questions=read_excel_column(year+".xlsx", "sheet1", "信息2")
  Answers=read_excel_column(year+".xlsx", "sheet1", "Correct_Answer_part2")

  df = pd.DataFrame(columns=["Question","Correct_Answer",'Answer1', "Answer2"])

  for i in range(len(Questions)):
    message = Questions[i]
    history=[]
    answers_from_model,history = model.chat(tokenizer, Questions[i],history=[])

    answer0=answers_from_model
    answer1=split_correct_answers(answer0)

    df2 = pd.DataFrame([
        #{"Question":Questions[i],"Correct_Answer":Answers[i],"Answer1":answer1,"Answer2":answer0,"Score":answer2}
        {"Question":Questions[i],"Correct_Answer":Answers[i],"Answer1":answer0,"Answer2":answer1}
    ])
    

    print("No Question",i+1,"\n","Right_Answer:",Answers[i],"\nAnswer_from_Two-stage_article-level_Deepspeek-LLM-7B-Chat:",answer1.strip().replace("\n",""))

    df = pd.concat([df, df2], axis=0)

  save_df_to_excel(df, "Two-stage_article-level_Deepspeek-LLM-7B-Chat", "sheet1")
