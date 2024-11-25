from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import os

model_path = "/root/autodl-tmp/Two-stage_article-level_Chinese-LLaMA2-7B"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path).half().cuda()
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


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


years=["Two-stage_act-level_Chinese-LLaMA2-7B"]
for year in years:
  print("Year",year)
  Questions=read_excel_column(year+".xlsx", "sheet1", "信息2")
  Answers=read_excel_column(year+".xlsx", "sheet1", "Correct_Answer_part2")

  df = pd.DataFrame(columns=["Question","Correct_Answer",'Answer1', "Answer2"])

  instruction = """[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{} [/INST]"""

  for i in range(len(Questions)):
    prompt = instruction.format(Questions[i])
    prompt_1 = Questions[i]
    generate_ids = model.generate(tokenizer(prompt, return_tensors='pt').input_ids.cuda(), max_new_tokens=128, streamer=streamer)
    answer0 = ""
    for batch in generate_ids:
        generated_text = tokenizer.decode(batch, skip_special_tokens=True)
        generated_text = generated_text.replace(prompt_1, "")
        answer0 += generated_text
    answer1 = split_correct_answers(answer0)
  
    df2 = pd.DataFrame([
        #{"Question":Questions[i],"Correct_Answer":Answers[i],"Answer1":answer1,"Answer2":answer0,"Score":answer2}
        {"Question":Questions[i],"Correct_Answer":Answers[i],"Answer1":answer0,"Answer2":answer1}
    ])
    
    print("No Question",i+1,"\n","Right_Answer:",Answers[i],"\nAnswer_from_Two-stage_article-level_Chinese-LLaMA2-7B:",answer1)

    df = pd.concat([df, df2], axis=0)

  save_df_to_excel(df, "Two-stage_article-level_Qwen1.5-7B-Chat.xlsx", "sheet1")
