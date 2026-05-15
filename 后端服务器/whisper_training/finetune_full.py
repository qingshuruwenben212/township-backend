# D:\township_project\后端服务器\whisper_training\finetune_full.py
import os
import tempfile

# ==================== 0. 强制临时文件到 D 盘 ====================
D_TEMP = "D:/township_project/temp_files"
os.makedirs(D_TEMP, exist_ok=True)

os.environ["TEMP"] = D_TEMP
os.environ["TMP"] = D_TEMP
os.environ["TMPDIR"] = D_TEMP
tempfile.tempdir = D_TEMP

print(f"✓ Python 临时目录: {tempfile.gettempdir()}")
print(f"✓ 临时文件将写入: {D_TEMP}")

# ==================== 1. 设置模型缓存到 D 盘 ====================
os.environ["HF_HOME"] = "D:/township_project/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/township_project/huggingface_cache"
os.environ["TORCH_HOME"] = "D:/township_project/torch_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

os.makedirs("D:/township_project/huggingface_cache", exist_ok=True)
os.makedirs("D:/township_project/torch_cache", exist_ok=True)

# ==================== 主程序入口 ====================
if __name__ == '__main__':
    # 配置
    PROCESSED_DIR = r"D:\township_project\后端服务器\whisper_training\data\canto_processed"
    MODEL_ID = "openai/whisper-small"
    OUTPUT_DIR = r"D:\township_project\后端服务器\whisper_training\models\whisper-canto-small"
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    
    TRAIN_EPOCHS = 1
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 2
    LEARNING_RATE = 3e-5
    CHECKPOINT_INTERVAL = 1  # 每条数据都保存
    KEEP_LATEST_CHECKPOINTS = 1  # 只保留最新的 1 个

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*60)
    print("Whisper 粤语微调 - 完整训练版本（small 模型，内存优化版）")
    print(f"临时目录: {D_TEMP}")
    print(f"缓存目录: D:/township_project/huggingface_cache")
    print("="*60)
    print(f"训练配置:")
    print(f"  模型: {MODEL_ID}")
    print(f"  训练轮数: {TRAIN_EPOCHS}")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  梯度累积: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  检查点间隔: 每条数据都保存")
    print(f"  保留检查点数: 只保留最新的 {KEEP_LATEST_CHECKPOINTS} 个")
    print("="*60)

    # 导入库
    import json
    import torch
    import librosa
    import shutil
    import gc
    import signal
    import re
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        WhisperFeatureExtractor,
        WhisperTokenizer,
        WhisperProcessor,
        WhisperForConditionalGeneration,
        get_linear_schedule_with_warmup
    )
    from tqdm import tqdm

    # ==================== 全局变量 ====================
    interrupted = False
    current_epoch = 0
    current_data_index = 0
    current_global_step = 0
    current_best_eval_loss = float('inf')
    current_checkpoint_save_count = 0
    current_model = None
    current_processor = None
    current_optimizer = None
    current_scheduler = None
    current_checkpoint_dir = None

    def signal_handler(sig, frame):
        global interrupted
        global current_epoch, current_data_index, current_global_step
        global current_best_eval_loss, current_checkpoint_save_count
        global current_model, current_processor, current_optimizer, current_scheduler
        global current_checkpoint_dir
        
        if not interrupted:
            print("\n\n⚠️ 收到中断信号，正在保存检查点...")
            interrupted = True
            
            if current_model is not None and current_data_index > 0:
                checkpoint_path = os.path.join(current_checkpoint_dir, f"checkpoint-epoch{current_epoch+1}-data{current_data_index}")
                os.makedirs(checkpoint_path, exist_ok=True)
                
                current_model.save_pretrained(checkpoint_path)
                current_processor.save_pretrained(checkpoint_path)
                
                training_state = {
                    "epoch": current_epoch,
                    "data_index": current_data_index,
                    "global_step": current_global_step,
                    "best_eval_loss": current_best_eval_loss,
                    "checkpoint_save_count": current_checkpoint_save_count,
                    "optimizer": current_optimizer.state_dict(),
                    "scheduler": current_scheduler.state_dict(),
                }
                torch.save(training_state, os.path.join(checkpoint_path, "training_state.pt"), _use_new_zipfile_serialization=False)
                print(f"\n💾 已保存中断检查点: epoch{current_epoch+1}-data{current_data_index}")
                
                # 中断时也清理旧检查点，只保留最新的
                cleanup_old_checkpoints(current_checkpoint_dir, 1)
    
    signal.signal(signal.SIGINT, signal_handler)

    def clean_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def parse_checkpoint_name(checkpoint_name):
        """从检查点文件夹名解析 epoch 和 data 数字"""
        pattern = r"checkpoint-epoch(\d+)-data(\d+)"
        match = re.match(pattern, checkpoint_name)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None

    def cleanup_old_checkpoints(checkpoint_dir, keep_latest=1):
        """删除旧的检查点，只保留最新的 keep_latest 个"""
        if not os.path.exists(checkpoint_dir):
            return
        
        checkpoints = []
        for item in os.listdir(checkpoint_dir):
            item_path = os.path.join(checkpoint_dir, item)
            if os.path.isdir(item_path) and item.startswith("checkpoint-"):
                epoch_num, data_num = parse_checkpoint_name(item)
                if epoch_num is not None and data_num is not None:
                    checkpoints.append((item_path, epoch_num, data_num))
        
        if len(checkpoints) <= keep_latest:
            return
        
        # 按 epoch 降序，然后按 data 降序排序（最新的在前）
        checkpoints.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        # 保留前 keep_latest 个，删除后面的
        to_delete = checkpoints[keep_latest:]
        
        for checkpoint_path, epoch_num, data_num in to_delete:
            try:
                shutil.rmtree(checkpoint_path)
                print(f"  🗑️ 删除旧检查点: checkpoint-epoch{epoch_num}-data{data_num}")
            except Exception as e:
                print(f"  ⚠️ 删除失败: checkpoint-epoch{epoch_num}-data{data_num}")

    # 1. 加载数据
    print("\n[1/5] 加载数据集...")
    
    def load_json_data(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    if not os.path.exists(PROCESSED_DIR):
        print(f"❌ 数据目录不存在: {PROCESSED_DIR}")
        sys.exit(1)

    train_data = load_json_data(os.path.join(PROCESSED_DIR, "train.json"))
    eval_data = load_json_data(os.path.join(PROCESSED_DIR, "dev.json"))

    print(f"  训练集: {len(train_data)} 条")
    print(f"  验证集: {len(eval_data)} 条")

    # 2. 加载模型
    print("\n[2/5] 加载模型...")
    
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_ID, language="yue", task="transcribe")
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="yue", task="transcribe")
    
    start_epoch = 0
    start_data_index = 0
    global_step = 0
    best_eval_loss = float('inf')
    checkpoint_save_count = 0
    
    # 收集所有检查点，找最新的
    checkpoint_files = []
    if os.path.exists(CHECKPOINT_DIR):
        for item in os.listdir(CHECKPOINT_DIR):
            item_path = os.path.join(CHECKPOINT_DIR, item)
            if os.path.isdir(item_path) and item.startswith("checkpoint-"):
                epoch_num, data_num = parse_checkpoint_name(item)
                if epoch_num is not None and data_num is not None:
                    checkpoint_files.append((item_path, epoch_num, data_num))
    
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: (x[1], x[2]), reverse=True)
        latest_checkpoint_path = checkpoint_files[0][0]
        latest_epoch = checkpoint_files[0][1]
        latest_data = checkpoint_files[0][2]
        
        print(f"  🔄 发现最新检查点: {os.path.basename(latest_checkpoint_path)}")
        print(f"     第 {latest_epoch} 轮，第 {latest_data} 条数据")
        
        model = WhisperForConditionalGeneration.from_pretrained(latest_checkpoint_path)
        
        state_path = os.path.join(latest_checkpoint_path, "training_state.pt")
        if os.path.exists(state_path):
            try:
                training_state = torch.load(state_path, weights_only=False)
                start_epoch = training_state.get("epoch", latest_epoch - 1)
                start_data_index = training_state.get("data_index", latest_data)
                global_step = training_state.get("global_step", 0)
                best_eval_loss = training_state.get("best_eval_loss", float('inf'))
                checkpoint_save_count = training_state.get("checkpoint_save_count", 0)
                print(f"  ✅ 从第 {start_epoch + 1} 轮第 {start_data_index} 条数据继续训练")
            except Exception as e:
                print(f"  ⚠️ 检查点状态文件损坏: {e}")
                start_epoch = latest_epoch - 1
                start_data_index = latest_data
        else:
            start_epoch = latest_epoch - 1
            start_data_index = latest_data
            print(f"  ✅ 从第 {start_epoch + 1} 轮第 {start_data_index} 条数据继续训练")
    else:
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
        print("  ✅ 从头开始训练")
    
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="yue", task="transcribe")
    model.config.suppress_tokens = []
    model.config.use_cache = False

    # 3. 创建 Dataset
    print("\n[3/5] 创建数据集...")
    
    class WhisperDataset(Dataset):
        def __init__(self, data, feature_extractor, tokenizer):
            self.data = data
            self.feature_extractor = feature_extractor
            self.tokenizer = tokenizer
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]
            audio_path = item["audio_path"]
            
            try:
                audio, sr = librosa.load(audio_path, sr=16000)
            except Exception as e:
                print(f"  ⚠️ 加载音频失败 [{idx}]: {audio_path}")
                return None
            
            try:
                input_features = self.feature_extractor(
                    audio, sampling_rate=16000, return_tensors="pt"
                ).input_features[0]
            except Exception as e:
                print(f"  ⚠️ 提取特征失败 [{idx}]")
                return None
            
            try:
                labels = self.tokenizer(item["text"]).input_ids
            except Exception as e:
                print(f"  ⚠️ 编码标签失败 [{idx}]")
                return None
            
            return {
                "input_features": input_features,
                "labels": labels
            }
    
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        
        input_features = torch.stack([b["input_features"] for b in batch])
        
        labels_list = [torch.tensor(b["labels"]) for b in batch]
        max_len = max(len(l) for l in labels_list)
        padded_labels = []
        for labels in labels_list:
            if len(labels) < max_len:
                padding = torch.full((max_len - len(labels),), -100, dtype=labels.dtype)
                padded = torch.cat([labels, padding])
            else:
                padded = labels
            padded_labels.append(padded)
        labels = torch.stack(padded_labels)
        
        return {
            "input_features": input_features,
            "labels": labels
        }
    
    train_dataset = WhisperDataset(train_data, feature_extractor, tokenizer)
    eval_dataset = WhisperDataset(eval_data, feature_extractor, tokenizer)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )

    # 4. 训练
    print("\n[4/5] 开始训练...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    total_steps = len(train_loader) * TRAIN_EPOCHS // GRADIENT_ACCUMULATION_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=500, 
        num_training_steps=total_steps
    )
    
    # 尝试恢复优化器状态
    if checkpoint_files and os.path.exists(state_path):
        try:
            optimizer.load_state_dict(training_state["optimizer"])
            scheduler.load_state_dict(training_state["scheduler"])
            print(f"  ✅ 恢复优化器和调度器状态")
        except Exception as e:
            print(f"  ⚠️ 优化器状态恢复失败: {e}")
    
    print(f"  设备: {device}")
    print(f"  总训练步数: {total_steps}")
    print(f"  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("="*60)
    print("训练进行中，预计 2-3 小时...")
    print("按 Ctrl+C 可安全中断训练")
    print("="*60)
    
    current_model = model
    current_processor = processor
    current_optimizer = optimizer
    current_scheduler = scheduler
    current_checkpoint_dir = CHECKPOINT_DIR
    current_epoch = start_epoch
    current_data_index = start_data_index
    current_global_step = global_step
    current_best_eval_loss = best_eval_loss
    current_checkpoint_save_count = checkpoint_save_count
    
    model.train()
    
    try:
        for epoch in range(start_epoch, TRAIN_EPOCHS):
            if interrupted:
                break
            
            current_epoch = epoch
            
            print(f"\nEpoch {epoch + 1}/{TRAIN_EPOCHS}")
            epoch_loss = 0
            epoch_processed = 0
            
            start_idx = start_data_index if epoch == start_epoch else 0
            
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training")
            optimizer.zero_grad()
            
            for idx, batch in progress_bar:
                if interrupted:
                    break
                
                if idx < start_idx:
                    continue
                
                if batch is None:
                    continue
                
                input_features = batch["input_features"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(
                    input_features=input_features,
                    labels=labels
                )
                
                loss = outputs.loss
                loss = loss / GRADIENT_ACCUMULATION_STEPS
                loss.backward()
                
                epoch_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
                epoch_processed += 1
                
                current_data_index = idx + 1
                current_global_step = global_step
                
                if (idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    current_global_step = global_step
                    
                    progress_bar.set_postfix({
                        "loss": f"{epoch_loss / epoch_processed:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                    })
                
                # 每条数据都保存检查点
                absolute_data_index = start_idx + epoch_processed
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint-epoch{epoch+1}-data{absolute_data_index}")
                
                # 保存检查点
                os.makedirs(checkpoint_path, exist_ok=True)
                model.save_pretrained(checkpoint_path)
                processor.save_pretrained(checkpoint_path)
                
                training_state = {
                    "epoch": epoch,
                    "data_index": absolute_data_index,
                    "global_step": global_step,
                    "best_eval_loss": best_eval_loss,
                    "checkpoint_save_count": checkpoint_save_count,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                torch.save(training_state, os.path.join(checkpoint_path, "training_state.pt"), _use_new_zipfile_serialization=False)
                
                print(f"\n  💾 保存检查点: epoch{epoch+1}-data{absolute_data_index}")
                
                # 每次保存后立即清理，只保留最新的 1 个
                cleanup_old_checkpoints(CHECKPOINT_DIR, 1)
                
                clean_memory()
            
            if interrupted:
                break
            
            avg_epoch_loss = epoch_loss / epoch_processed if epoch_processed > 0 else 0
            print(f"Epoch {epoch + 1} 完成, 平均损失: {avg_epoch_loss:.4f}")
            
            # 评估
            model.eval()
            eval_loss = 0
            eval_steps = 0
            with torch.no_grad():
                for eval_batch in eval_loader:
                    if eval_batch is None:
                        continue
                    eval_input_features = eval_batch["input_features"].to(device)
                    eval_labels = eval_batch["labels"].to(device)
                    eval_outputs = model(
                        input_features=eval_input_features,
                        labels=eval_labels
                    )
                    eval_loss += eval_outputs.loss.item()
                    eval_steps += 1
            
            if eval_steps > 0:
                avg_eval_loss = eval_loss / eval_steps
                print(f"  📊 Epoch {epoch + 1}: train_loss={avg_epoch_loss:.4f}, eval_loss={avg_eval_loss:.4f}")
                
                if avg_eval_loss < best_eval_loss:
                    best_eval_loss = avg_eval_loss
                    current_best_eval_loss = best_eval_loss
                    print(f"  🎉 保存最佳模型 (eval_loss={avg_eval_loss:.4f})")
                    model.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
                    processor.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
                    clean_memory()
            
            model.train()
            start_idx = 0
        
        if interrupted:
            print(f"\n⚠️ 训练已中断，检查点已保存")
            print(f"   再次运行 python finetune_full.py 可继续训练")
        else:
            # 5. 保存最终模型
            print("\n[5/5] 保存最终模型...")
            model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
            processor.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
            
            config_info = {
                "model_id": MODEL_ID,
                "train_epochs": TRAIN_EPOCHS,
                "train_samples": len(train_data),
                "eval_samples": len(eval_data),
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "checkpoint_interval": CHECKPOINT_INTERVAL,
                "keep_latest_checkpoints": 1,
                "best_eval_loss": best_eval_loss,
            }
            with open(os.path.join(OUTPUT_DIR, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config_info, f, ensure_ascii=False, indent=2)
            
            # 最终清理
            cleanup_old_checkpoints(CHECKPOINT_DIR, 1)
            clean_memory()
            
            print(f"\n✅ 完整训练完成！")
            print(f"   最终模型: {OUTPUT_DIR}/final")
            print(f"   最佳模型: {OUTPUT_DIR}/best_model")
        
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*60)