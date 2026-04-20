# # from modelscope import snapshot_download
# # model_dir = snapshot_download('misstl/JanusVLN_Base',cache_dir='JanusVLN_Extra/')


# from transformers import AutoModel, AutoTokenizer

# # 修改前
# model = AutoModel.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# print("Model and tokenizer loaded successfully.")

# # 1. 检查 name_or_path (最直观)
# # 如果是本地加载，这里通常会显示你的绝对路径
# # 如果是缓存加载，这里通常显示 'bert-base-uncased'
# print(f"▶ 模型名/路径 (model.name_or_path): \n   {model.name_or_path}")

# # 2. 检查 Config 中的路径 (有时候比 name_or_path 更详细)
# print(f"▶ Config 记录的路径 (model.config._name_or_path): \n   {model.config._name_or_path}")

# # 3. 【最硬核的判断】查看 Tokenizer 的物理文件路径
# # 分词器必须加载一个真实的 vocab.txt 或 tokenizer.json 文件
# # 这个文件的路径会直接暴露它在磁盘的哪个位置
# try:
#     # 针对 BERT 这种使用 vocab.txt 的模型
#     if hasattr(tokenizer, 'vocab_file'):
#         print(f"▶ 真实物理文件位置 (vocab.txt): \n   {os.path.abspath(tokenizer.vocab_file)}")
#     # 针对 Fast Tokenizer (如 Qwen, LLaMA 等)
#     elif hasattr(tokenizer, 'backend_tokenizer'):
#         # 有些版本可以直接打印 tokenizer 对象看到路径
#         print(f"▶ Tokenizer 后端信息: \n   {tokenizer.backend_tokenizer}")
#     else:
#         print("▶ 无法直接获取物理文件路径，请依据上方 name_or_path 判断")
# except Exception as e:
#     print(f"   (获取物理路径时发生错误: {e})")

# print("="*40 + "\n")

import os
from transformers import AutoModel, AutoTokenizer
from transformers.utils.hub import cached_file
from huggingface_hub import constants as hf_const

MODEL_ID = "bert-base-uncased"

# 原来的加载
model = AutoModel.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("模型和分词器已成功加载。")

# 1) 打印缓存根目录（默认 ~/.cache/huggingface/hub，除非你设置了环境变量覆盖）
print("\n[Cache env]")
print("HF_HOME =", os.environ.get("HF_HOME"))
print("HF_HUB_CACHE =", os.environ.get("HF_HUB_CACHE"))
print("TRANSFORMERS_CACHE =", os.environ.get("TRANSFORMERS_CACHE"))
print("huggingface_hub.constants.HF_HUB_CACHE =", hf_const.HF_HUB_CACHE)

# 2) 打印“实际解析到的本地文件路径”（命中缓存时会返回真实路径）
print("\n[Resolved local files used from cache]")
files_to_check = [
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
    "special_tokens_map.json",
]
for fname in files_to_check:
    try:
        path = cached_file(MODEL_ID, fname)  # 返回缓存中的真实文件路径（若存在）
        if path is not None:
            print(f"{fname:22s} -> {path}")
    except Exception as e:
        # 文件不存在/模型不含该文件/未缓存等都会进这里
        pass

# 3) 额外：打印“传给 from_pretrained 的字符串”（它不等于缓存路径）
print("\n[Name/path recorded in objects]")
print("model.name_or_path =", getattr(model, "name_or_path", None))
print("model.config.name_or_path =", getattr(model.config, "name_or_path", None))
print("tokenizer.name_or_path =", getattr(tokenizer, "name_or_path", None))
