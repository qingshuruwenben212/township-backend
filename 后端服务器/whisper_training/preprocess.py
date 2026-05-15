# D:\township_project\whisper_training\preprocess.py
import os
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import json

# ==================== 配置路径 ====================
# 原始数据路径（您下载并解压后的数据集）
DATA_ROOT = r"D:\各种压缩包\cv-corpus-25.0-2026-03-09\zh-HK"

# 输出路径（处理后的数据）
OUTPUT_DIR = r"D:\township_project\后端服务器\whisper_training\data\canto_processed"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "audio", "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "audio", "dev"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "audio", "test"), exist_ok=True)

print("="*50)
print("Common Voice 25.0 粤语数据集预处理")
print("="*50)

# ==================== 检查数据是否存在 ====================
if not os.path.exists(DATA_ROOT):
    print(f"错误: 数据目录不存在！")
    print(f"请确认路径: {DATA_ROOT}")
    print("提示: 请先解压下载的 .tar.gz 文件")
    exit(1)

print(f"数据目录: {DATA_ROOT}")
print(f"输出目录: {OUTPUT_DIR}")

# ==================== 读取数据 ====================
print("\n读取数据文件...")
train_df = pd.read_csv(os.path.join(DATA_ROOT, "train.tsv"), sep="\t")
dev_df = pd.read_csv(os.path.join(DATA_ROOT, "dev.tsv"), sep="\t")
test_df = pd.read_csv(os.path.join(DATA_ROOT, "test.tsv"), sep="\t")

print(f"训练集: {len(train_df)} 条")
print(f"验证集: {len(dev_df)} 条")
print(f"测试集: {len(test_df)} 条")

# ==================== 文本清洗 ====================
def clean_text(text):
    if not text or pd.isna(text):
        return ""
    text = str(text).strip()
    text = ' '.join(text.split())
    return text

# ==================== 音频处理 ====================
def process_audio(row, split_name):
    audio_path = os.path.join(DATA_ROOT, "clips", row["path"])
    
    if not os.path.exists(audio_path):
        return None
    
    wav_filename = row["path"].replace(".mp3", ".wav")
    wav_path = os.path.join(OUTPUT_DIR, "audio", split_name, wav_filename)
    
    if os.path.exists(wav_path):
        return wav_path
    
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        sf.write(wav_path, audio, 16000)
        return wav_path
    except Exception as e:
        print(f"处理失败: {audio_path}")
        return None

# ==================== 处理数据 ====================
def process_split(df, split_name):
    print(f"\n处理 {split_name} 集...")
    data_list = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        wav_path = process_audio(row, split_name)
        if wav_path is None:
            continue
        
        text = clean_text(row["sentence"])
        if len(text) < 2:
            continue
        
        data_list.append({
            "audio_path": wav_path,
            "text": text
        })
    
    # 保存索引文件
    output_file = os.path.join(OUTPUT_DIR, f"{split_name}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)
    
    print(f"{split_name} 集完成: {len(data_list)} 条")
    return data_list

# 处理各个数据集
train_data = process_split(train_df, "train")
dev_data = process_split(dev_df, "dev")
test_data = process_split(test_df, "test")

print("\n" + "="*50)
print("数据预处理完成！")
print(f"训练集: {len(train_data)} 条")
print(f"验证集: {len(dev_data)} 条")
print(f"测试集: {len(test_data)} 条")
print(f"输出目录: {OUTPUT_DIR}")
print("="*50)