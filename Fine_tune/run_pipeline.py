"""
完整執行腳本
============
一鍵執行整個 fine-tune 流程

執行方式:
    python run_pipeline.py [--skip-train] [--compare]

參數:
    --skip-train: 跳過訓練步驟（使用已有的微調模型）
    --compare: 同時輸出原始模型和微調模型的結果進行比較
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# 專案根目錄
PROJECT_ROOT = Path(__file__).parent


def run_command(cmd: list, description: str) -> bool:
    """
    執行命令
    
    Args:
        cmd: 命令列表
        description: 描述
        
    Returns:
        是否成功
    """
    print(f"\n{'='*60}")
    print(f"📌 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 執行失敗: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="執行完整的 fine-tune pipeline")
    
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="跳過訓練步驟"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="比較原始模型和微調模型的結果"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Pyannote Fine-tune Pipeline")
    print("=" * 60)
    
    # Step 0: 建立目錄
    print("\n[Step 0] 建立目錄結構...")
    run_command(
        [sys.executable, str(PROJECT_ROOT / "config.py")],
        "建立目錄"
    )
    
    # Step 1: 準備資料
    print("\n[Step 1] 準備資料集...")
    success = run_command(
        [sys.executable, str(PROJECT_ROOT / "01_prepare_dataset.py")],
        "準備資料集"
    )
    
    if not success:
        print("❌ 資料準備失敗，停止執行")
        return
    
    # Step 2: 訓練模型
    if not args.skip_train:
        print("\n[Step 2] 訓練模型...")
        success = run_command(
            [sys.executable, str(PROJECT_ROOT / "02_train_model.py")],
            "訓練模型"
        )
        
        if not success:
            print("⚠️ 訓練失敗，但將繼續使用原始模型進行轉錄")
    else:
        print("\n[Step 2] 跳過訓練步驟")
    
    # Step 3: 轉錄測試
    print("\n[Step 3] 使用模型進行轉錄...")
    
    transcribe_cmd = [sys.executable, str(PROJECT_ROOT / "03_transcribe_with_finetuned.py")]
    
    if args.compare:
        transcribe_cmd.append("--compare")
    
    run_command(
        transcribe_cmd,
        "語音轉錄和語者分離"
    )
    
    print("\n" + "=" * 60)
    print("🎉 Pipeline 執行完成!")
    print("=" * 60)
    
    # 顯示輸出目錄
    output_dir = PROJECT_ROOT / "outputs"
    if output_dir.exists():
        outputs = list(output_dir.glob("*.srt"))
        if outputs:
            print(f"\n📁 輸出檔案:")
            for f in outputs:
                print(f"   - {f}")


if __name__ == "__main__":
    main()
