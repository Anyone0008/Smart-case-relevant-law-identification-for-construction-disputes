# Construction case-relevant article-level law identification using fine-tuned large language models: A study in China 

## !!! As the paper is under review, all materials in this repository currently are not allowed to be re-used by anyone until this announcement is deleted.

# 0. Videos of running easy-to-use interface to fine-tune LLMs and identify relevant acts and articles
To avoid potential video quality reduction mentioned in the README, please download the video directly from the repository above.
<video src="https://github.com/user-attachments/assets/bbc7e998-7fdb-4053-a4b7-2e3fb071e0ac" controls="controls" width="500" height="300"></video>
↑↑↑ The fine-tuning process with the easy-to-use interface

<video src="https://github.com/user-attachments/assets/733c8baa-b868-4c5f-a394-0dc99a76af1a" controls="controls" width="500" height="300"></video>
↑↑↑ The act-article law identification process with the easy-to-use interface

# 1. General introduction of this repository
1.1 This repository aims at providing the codes and data regarding the paper entitled “……” for the public, and it is developed by the XXX university in China, The University of XX in Hong Kong SAR.

1.2 We greatly appreciate the selfless spirits of these voluntary contributors of a series of open python libraries, including Baichuan-inc (https://huggingface.co/baichuan-inc), LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory), ChatGLM3-6b (https://huggingface.co/THUDM/chatglm3-6b), Qwen (https://huggingface.co/Qwen), numpy, and so on. Our work stands on the shoulders of these giants

1.3 All of the codes have been tested to be well-performed. Even so, we are not able to guarantee their operation in other computing environments due to the differences in the Python version, computer operating system, and adopted hardware.

# 2. Summary of supplemental materials in this repository
The table below shows all supplemental materials.
<img width="952" alt="3092ad9b0b55c607800bf1737304448" src="https://github.com/user-attachments/assets/38666214-c376-4c1a-a534-fb56b76d60ba" />


# 3. Prototype of article-level law identification for construction cases
The deploy of the easy-to-use interface in paper is based on LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory). You can start with the instruction shown in the repository.
<img width="808" alt="1613ba6adffb3b4d3fa506ec6f345f0" src="https://github.com/user-attachments/assets/6e9bec9f-b16e-4ac8-9231-a3e0944c0ca9">

3.1 Download original model, such as Qwen1.5-14B-Chat (https://huggingface.co/Qwen/Qwen1.5-14B-Chat).

3.2 Create a folder named by the original model in the desired directory like "/LLaMA-Factory/saves/Qwen1.5-14B-Chat/lora"

3.3 Download the act-level identification and article-level identification models from Google Drive (https://drive.google.com/drive/folders/1T58vR0lq8g_RBs9Be_7lpeJ48dSLRPU_?usp=drive_link)

3.4 Place the downloaded models into the newly created folder.

3.5 Run src/train_web.py to start the law identification models.


# 4. Reuse of the construction case dataset
The dataset includes 81,472 real-world construction-related cases, and each case is dissected and labelled for "Facts", "Case types", "Cited acts", "Cited articles", and "Cited articles with specific content".
<img width="651" alt="fd372d28d82be95e722684b5a60820f" src="https://github.com/user-attachments/assets/06554dc5-cb0f-44e1-8226-053d5315d4ad" />



