# -*- coding: utf-8 -*-
"""
論文復現：Exploring Speech Foundation Models for Speaker Diarization in Child-Adult Dyadic Interactions
評估模組：DER 計算、Confusion Matrix、甘特圖視覺化

作者：AI Engineering Team
日期：2026-01-19
修訂：2026-01-19 - 改為 3 分類任務 (Child=0, Adult=1, Silence=2)
             - 移除 Overlap 相關邏輯
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib
from sklearn.metrics import confusion_matrix, classification_report

# 設定中文字體支援
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from config import Config


# ==================== DER 計算器 ====================

@dataclass
class DERResult:
    """DER 計算結果"""
    der: float                  # Diarization Error Rate
    false_alarm: float          # False Alarm Rate
    missed_detection: float     # Missed Detection Rate
    speaker_confusion: float    # Speaker Confusion Rate
    total_duration: float       # 總時長（秒）


class DERCalculator:
    """
    Diarization Error Rate (DER) 計算器
    =====================================
    
    DER = (False Alarm + Missed Detection + Speaker Confusion) / Total Reference Duration
    
    論文使用 100ms 的 forgiveness collar
    
    3 分類任務: Child(0), Adult(1), Silence(2)
    """
    
    def __init__(
        self,
        collar_ms: float = 100.0,
        step_size_ms: float = 20.0
    ):
        """
        初始化 DER 計算器
        
        Args:
            collar_ms: Forgiveness collar（毫秒）
            step_size_ms: Frame step size（毫秒）
        """
        self.collar_ms = collar_ms
        self.step_size_ms = step_size_ms
        
        # 計算 collar 對應的 frame 數
        self.collar_frames = int(collar_ms / step_size_ms)
    
    def calculate(
        self,
        predictions: np.ndarray,
        references: np.ndarray,
        apply_collar: bool = True
    ) -> DERResult:
        """
        計算 DER
        
        Args:
            predictions: 預測標籤, shape: (num_frames,)
            references: 真實標籤, shape: (num_frames,)
            apply_collar: 是否應用 forgiveness collar
            
        Returns:
            DERResult: DER 計算結果
        """
        # 確保長度一致
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]
        
        # 3 分類: 0=Child, 1=Adult, 2=Silence
        # 語音 = Child(0) 或 Adult(1), 靜音 = Silence(2)
        
        pred_is_speech = predictions != 2
        ref_is_speech = references != 2
        
        total_frames = len(references)
        ref_speech_frames = ref_is_speech.sum()
        
        # False Alarm: 預測為語音但實際為靜音
        false_alarm_frames = ((pred_is_speech) & (~ref_is_speech)).sum()
        
        # Missed Detection: 實際為語音但預測為靜音
        missed_frames = ((~pred_is_speech) & (ref_is_speech)).sum()
        
        # Speaker Confusion: 都是語音但語者不同
        both_speech = pred_is_speech & ref_is_speech
        confusion_frames = (both_speech & (predictions != references)).sum()
        
        # 計算各項比率
        total_duration = total_frames * self.step_size_ms / 1000.0
        ref_duration = ref_speech_frames * self.step_size_ms / 1000.0
        
        if ref_speech_frames == 0:
            return DERResult(
                der=0.0,
                false_alarm=0.0,
                missed_detection=0.0,
                speaker_confusion=0.0,
                total_duration=total_duration
            )
        
        false_alarm_rate = false_alarm_frames / ref_speech_frames * 100
        missed_rate = missed_frames / ref_speech_frames * 100
        confusion_rate = confusion_frames / ref_speech_frames * 100
        
        der = false_alarm_rate + missed_rate + confusion_rate
        
        return DERResult(
            der=der,
            false_alarm=false_alarm_rate,
            missed_detection=missed_rate,
            speaker_confusion=confusion_rate,
            total_duration=total_duration
        )


# ==================== Confusion Matrix 視覺化 ====================

class ConfusionMatrixVisualizer:
    """
    混淆矩陣視覺化器
    ================
    
    3 分類: Child(0), Adult(1), Silence(2)
    """
    
    def __init__(self, class_names: List[str] = None):
        """
        初始化視覺化器
        
        Args:
            class_names: 類別名稱列表
        """
        self.class_names = class_names or ["Child", "Adult", "Silence"]
    
    def compute(
        self,
        predictions: np.ndarray,
        references: np.ndarray
    ) -> np.ndarray:
        """
        計算混淆矩陣
        
        Args:
            predictions: 預測標籤
            references: 真實標籤
            
        Returns:
            np.ndarray: 混淆矩陣
        """
        return confusion_matrix(references, predictions, labels=[0, 1, 2])
    
    def plot(
        self,
        cm: np.ndarray,
        save_path: Optional[str] = None,
        normalize: bool = True,
        title: str = "混淆矩陣"
    ):
        """
        繪製混淆矩陣
        
        Args:
            cm: 混淆矩陣
            save_path: 儲存路徑
            normalize: 是否正規化
            title: 圖表標題
        """
        if normalize:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)  # 處理 NaN
            cm_display = cm_normalized
        else:
            cm_display = cm
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(cm_display, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(
            xticks=np.arange(len(self.class_names)),
            yticks=np.arange(len(self.class_names)),
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ylabel='真實標籤',
            xlabel='預測標籤'
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 旋轉 x 軸標籤
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 在每個格子中顯示數值
        fmt = '.2f' if normalize else 'd'
        thresh = cm_display.max() / 2.
        
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                value = cm_display[i, j]
                if normalize:
                    text = f'{value:.2%}\n({cm[i, j]})'
                else:
                    text = f'{value}'
                
                ax.text(
                    j, i, text,
                    ha="center", va="center",
                    color="white" if value > thresh else "black",
                    fontsize=10
                )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"混淆矩陣已儲存至: {save_path}")
        
        plt.close()


# ==================== 甘特圖視覺化 ====================

class GanttChartVisualizer:
    """
    甘特圖視覺化器
    ==============
    
    顯示語者分離結果的時間軸視覺化
    
    3 分類: Child(0), Adult(1), Silence(2)
    """
    
    # 顏色定義 - 3 分類
    COLORS = {
        0: '#2196F3',  # Child - 藍色
        1: '#4CAF50',  # Adult - 綠色
        2: '#9E9E9E',  # Silence - 灰色
    }
    
    CLASS_NAMES = ['Child (小孩)', 'Adult (測試者)', 'Silence (靜音)']
    
    def __init__(self, step_size_ms: float = 20.0):
        """
        初始化甘特圖視覺化器
        
        Args:
            step_size_ms: Frame step size（毫秒）
        """
        self.step_size_ms = step_size_ms
    
    def _labels_to_segments(
        self,
        labels: np.ndarray
    ) -> List[Tuple[int, float, float]]:
        """
        將 frame-level 標籤轉換為時間片段
        
        Args:
            labels: Frame-level 標籤
            
        Returns:
            List[Tuple[int, float, float]]: [(label, start_time, end_time), ...]
        """
        segments = []
        
        if len(labels) == 0:
            return segments
        
        current_label = labels[0]
        start_frame = 0
        
        for i, label in enumerate(labels):
            if label != current_label:
                # 片段結束
                start_time = start_frame * self.step_size_ms / 1000.0
                end_time = i * self.step_size_ms / 1000.0
                segments.append((current_label, start_time, end_time))
                
                # 開始新片段
                current_label = label
                start_frame = i
        
        # 處理最後一個片段
        start_time = start_frame * self.step_size_ms / 1000.0
        end_time = len(labels) * self.step_size_ms / 1000.0
        segments.append((current_label, start_time, end_time))
        
        return segments
    
    def plot(
        self,
        predictions: np.ndarray,
        references: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "語者分離結果",
        max_duration: Optional[float] = None
    ):
        """
        繪製甘特圖
        
        Args:
            predictions: 預測標籤
            references: 真實標籤
            save_path: 儲存路徑
            title: 圖表標題
            max_duration: 最大顯示時長（秒），None 表示顯示全部
        """
        # 轉換為片段
        pred_segments = self._labels_to_segments(predictions)
        ref_segments = self._labels_to_segments(references)
        
        # 計算總時長
        total_duration = len(predictions) * self.step_size_ms / 1000.0
        
        if max_duration and total_duration > max_duration:
            # 截斷
            display_duration = max_duration
        else:
            display_duration = total_duration
        
        # 建立圖表
        fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
        
        # 繪製預測結果
        self._draw_segments(axes[0], pred_segments, display_duration, "預測結果")
        
        # 繪製真實標籤
        self._draw_segments(axes[1], ref_segments, display_duration, "真實標籤")
        
        # 設定共用 x 軸
        axes[1].set_xlabel('時間 (秒)', fontsize=12)
        
        # 建立圖例 - 3 分類
        legend_patches = [
            mpatches.Patch(color=self.COLORS[i], label=self.CLASS_NAMES[i])
            for i in range(3)
        ]
        fig.legend(
            handles=legend_patches,
            loc='upper center',
            ncol=3,
            bbox_to_anchor=(0.5, 1.02),
            fontsize=10
        )
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.08)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"甘特圖已儲存至: {save_path}")
        
        plt.close()
    
    def _draw_segments(
        self,
        ax,
        segments: List[Tuple[int, float, float]],
        max_duration: float,
        title: str
    ):
        """繪製片段到軸"""
        for label, start, end in segments:
            if start > max_duration:
                continue
            
            end = min(end, max_duration)
            duration = end - start
            
            ax.barh(
                y=0.5,
                width=duration,
                left=start,
                height=0.8,
                color=self.COLORS.get(label, '#000000'),
                edgecolor='white',
                linewidth=0.5
            )
        
        ax.set_xlim(0, max_duration)
        ax.set_ylim(0, 1)
        ax.set_ylabel(title, fontsize=11)
        ax.set_yticks([])
        ax.grid(True, axis='x', alpha=0.3)
    
    def plot_comparison(
        self,
        predictions: np.ndarray,
        references: np.ndarray,
        save_path: Optional[str] = None,
        window_size: float = 60.0,
        num_windows: int = 3
    ):
        """
        繪製多個時間窗口的比較圖
        
        Args:
            predictions: 預測標籤
            references: 真實標籤
            save_path: 儲存路徑
            window_size: 每個窗口的大小（秒）
            num_windows: 窗口數量
        """
        total_duration = len(predictions) * self.step_size_ms / 1000.0
        
        # 計算窗口起始位置
        if total_duration <= window_size * num_windows:
            # 如果總時長不足，只使用一個窗口
            starts = [0]
            window_size = total_duration
        else:
            # 均勻分布窗口
            gap = (total_duration - window_size * num_windows) / (num_windows + 1)
            starts = [gap + i * (window_size + gap) for i in range(num_windows)]
        
        fig, axes = plt.subplots(
            len(starts) * 2, 1,
            figsize=(16, 4 * len(starts)),
            sharex=False
        )
        
        if len(starts) == 1:
            axes = [axes] if not hasattr(axes, '__len__') else axes
        
        for i, start in enumerate(starts):
            # 計算 frame 範圍
            start_frame = int(start * 1000 / self.step_size_ms)
            end_frame = int((start + window_size) * 1000 / self.step_size_ms)
            end_frame = min(end_frame, len(predictions))
            
            # 擷取片段
            pred_window = predictions[start_frame:end_frame]
            ref_window = references[start_frame:end_frame]
            
            # 轉換為片段並調整時間
            pred_segments = self._labels_to_segments(pred_window)
            ref_segments = self._labels_to_segments(ref_window)
            
            # 繪製
            ax_idx = i * 2
            self._draw_segments(
                axes[ax_idx],
                pred_segments,
                window_size,
                f"預測 ({start:.0f}s - {start+window_size:.0f}s)"
            )
            self._draw_segments(
                axes[ax_idx + 1],
                ref_segments,
                window_size,
                f"真實 ({start:.0f}s - {start+window_size:.0f}s)"
            )
            
            axes[ax_idx + 1].set_xlabel('時間 (秒)', fontsize=10)
        
        # 圖例 - 3 分類
        legend_patches = [
            mpatches.Patch(color=self.COLORS[i], label=self.CLASS_NAMES[i])
            for i in range(3)
        ]
        fig.legend(
            handles=legend_patches,
            loc='upper center',
            ncol=3,
            bbox_to_anchor=(0.5, 1.02),
            fontsize=10
        )
        
        fig.suptitle('語者分離結果比較', fontsize=14, fontweight='bold', y=1.05)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"比較圖已儲存至: {save_path}")
        
        plt.close()

    def plot_full_timeline(
        self,
        predictions: np.ndarray,
        references: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "完整語者分離時間軸"
    ):
        """
        繪製完整錄音的甘特圖（自動調整比例）
        
        Args:
            predictions: 預測標籤
            references: 真實標籤
            save_path: 儲存路徑
            title: 圖表標題
        """
        # 轉換為片段
        pred_segments = self._labels_to_segments(predictions)
        ref_segments = self._labels_to_segments(references)
        
        # 計算總時長
        total_duration = len(predictions) * self.step_size_ms / 1000.0
        total_minutes = total_duration / 60.0
        
        # 根據時長自動調整圖表大小和刻度
        if total_duration <= 120:  # <= 2 分鐘
            fig_width = 16
            major_tick = 10  # 每 10 秒一個主刻度
            minor_tick = 5
            x_label = '時間 (秒)'
            x_scale = 1.0
        elif total_duration <= 600:  # <= 10 分鐘
            fig_width = 20
            major_tick = 30  # 每 30 秒一個主刻度
            minor_tick = 10
            x_label = '時間 (秒)'
            x_scale = 1.0
        elif total_duration <= 1800:  # <= 30 分鐘
            fig_width = 24
            major_tick = 60  # 每 1 分鐘一個主刻度
            minor_tick = 30
            x_label = '時間 (分鐘)'
            x_scale = 60.0
        else:  # > 30 分鐘
            fig_width = 28
            major_tick = 120  # 每 2 分鐘一個主刻度
            minor_tick = 60
            x_label = '時間 (分鐘)'
            x_scale = 60.0
        
        # 建立圖表
        fig, axes = plt.subplots(2, 1, figsize=(fig_width, 6), sharex=True)
        
        # 繪製預測結果
        self._draw_segments_full(axes[0], pred_segments, total_duration, "預測結果", x_scale)
        
        # 繪製真實標籤
        self._draw_segments_full(axes[1], ref_segments, total_duration, "真實標籤", x_scale)
        
        # 設定 x 軸刻度
        if x_scale == 60.0:
            # 以分鐘為單位
            tick_locs = np.arange(0, total_duration + major_tick, major_tick)
            tick_labels = [f"{t/60:.0f}" for t in tick_locs]
            axes[1].set_xticks(tick_locs)
            axes[1].set_xticklabels(tick_labels)
        else:
            # 以秒為單位
            tick_locs = np.arange(0, total_duration + major_tick, major_tick)
            axes[1].set_xticks(tick_locs)
        
        axes[1].set_xlabel(x_label, fontsize=12)
        
        # 建立圖例 - 3 分類
        legend_patches = [
            mpatches.Patch(color=self.COLORS[i], label=self.CLASS_NAMES[i])
            for i in range(3)
        ]
        fig.legend(
            handles=legend_patches,
            loc='upper center',
            ncol=3,
            bbox_to_anchor=(0.5, 1.02),
            fontsize=10
        )
        
        # 添加時長資訊
        duration_info = f"總時長: {int(total_minutes)}分{int(total_duration % 60)}秒 ({total_duration:.1f}秒)"
        fig.suptitle(f"{title}\n{duration_info}", fontsize=14, fontweight='bold', y=1.10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"完整時間軸甘特圖已儲存至: {save_path}")
        
        plt.close()

    def _draw_segments_full(
        self,
        ax,
        segments: List[Tuple[int, float, float]],
        total_duration: float,
        title: str,
        x_scale: float = 1.0
    ):
        """繪製完整片段到軸"""
        for label, start, end in segments:
            duration = end - start
            
            ax.barh(
                y=0.5,
                width=duration,
                left=start,
                height=0.8,
                color=self.COLORS.get(label, '#000000'),
                edgecolor='none',  # 完整圖不需要邊線，避免太擁擠
                linewidth=0
            )
        
        ax.set_xlim(0, total_duration)
        ax.set_ylim(0, 1)
        ax.set_ylabel(title, fontsize=11)
        ax.set_yticks([])
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')


# ==================== 評估器 ====================

class Evaluator:
    """
    模型評估器
    ==========
    
    整合 DER 計算、混淆矩陣、甘特圖視覺化
    """
    
    def __init__(self, config: Config):
        """
        初始化評估器
        
        Args:
            config: 配置物件
        """
        self.config = config
        self.device = config.device
        
        self.der_calculator = DERCalculator(
            collar_ms=config.der_collar_ms,
            step_size_ms=config.step_size_ms
        )
        
        self.cm_visualizer = ConfusionMatrixVisualizer(config.class_names)
        self.gantt_visualizer = GanttChartVisualizer(config.step_size_ms)
    
    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        save_dir: Optional[str] = None,
        apply_median_filter: bool = False
    ) -> Dict:
        """
        評估模型
        
        Args:
            model: 模型
            data_loader: 資料載入器
            save_dir: 結果儲存目錄
            apply_median_filter: 是否應用中值濾波（預設停用）
            
        Returns:
            Dict: 評估結果
        """
        model.eval()
        
        all_predictions = []
        all_references = []
        
        print("\n進行推論...")
        print(f"使用 Median Filter: {apply_median_filter}")
        
        for batch in tqdm(data_loader, desc="推論中"):
            audio = batch["audio"].to(self.device)
            labels = batch["labels"]
            
            if self.config.use_amp:
                with autocast():
                    logits = model(audio)
            else:
                logits = model(audio)
            
            predictions = torch.argmax(logits, dim=-1)
            
            # 應用中值濾波（如果模型有 median_filter 方法且啟用）
            if apply_median_filter and hasattr(model, 'median_filter'):
                predictions = model.median_filter.filter_predictions(predictions)
            
            predictions = predictions.cpu().numpy()
            labels = labels.numpy()
            
            # 對齊長度
            for pred, ref in zip(predictions, labels):
                seq_len = min(len(pred), len(ref))
                all_predictions.append(pred[:seq_len])
                all_references.append(ref[:seq_len])
        
        # 合併所有預測和參考
        all_predictions = np.concatenate(all_predictions)
        all_references = np.concatenate(all_references)
        
        print(f"總 frames 數: {len(all_predictions)}")
        
        # 顯示標籤分佈統計（診斷用）
        print("\n【標籤分佈統計】")
        for i, name in enumerate(self.config.class_names):
            ref_count = np.sum(all_references == i)
            pred_count = np.sum(all_predictions == i)
            ref_pct = ref_count / len(all_references) * 100
            pred_pct = pred_count / len(all_predictions) * 100
            print(f"  {name}: 參考={ref_count} ({ref_pct:.1f}%), 預測={pred_count} ({pred_pct:.1f}%)")
        
        # 計算 DER (3 分類，無 Overlap)
        print("\n計算 DER...")
        der_result = self.der_calculator.calculate(all_predictions, all_references)
        
        # 計算混淆矩陣
        print("計算混淆矩陣...")
        cm = self.cm_visualizer.compute(all_predictions, all_references)
        
        # 計算分類報告
        report = classification_report(
            all_references,
            all_predictions,
            target_names=self.config.class_names,
            output_dict=True,
            zero_division=0
        )
        
        # 儲存結果
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # 儲存混淆矩陣圖
            self.cm_visualizer.plot(
                cm,
                save_path=save_path / "confusion_matrix.png",
                normalize=True
            )
            
            # 儲存甘特圖
            self.gantt_visualizer.plot(
                all_predictions,
                all_references,
                save_path=save_path / "gantt_chart.png",
                max_duration=120  # 顯示前 2 分鐘
            )
            
            # 儲存多窗口比較圖
            self.gantt_visualizer.plot_comparison(
                all_predictions,
                all_references,
                save_path=save_path / "gantt_comparison.png"
            )
            
            # 儲存完整時間軸甘特圖
            self.gantt_visualizer.plot_full_timeline(
                all_predictions,
                all_references,
                save_path=save_path / "gantt_full_timeline.png"
            )
            
            # 儲存評估結果 JSON
            results = {
                "der": {
                    "total": der_result.der,
                    "false_alarm": der_result.false_alarm,
                    "missed_detection": der_result.missed_detection,
                    "speaker_confusion": der_result.speaker_confusion,
                },
                "classification_report": report,
                "confusion_matrix": cm.tolist(),
                "total_frames": len(all_predictions),
                "total_duration_seconds": der_result.total_duration,
            }
            
            with open(save_path / "evaluation_results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # ===== 新增：輸出完整推論結果 =====
            self._save_detailed_results(
                all_predictions, 
                all_references, 
                save_path,
                der_result
            )
            
            print(f"\n評估結果已儲存至: {save_path}")
        
        # 輸出結果
        self._print_results(der_result, cm, report)
        
        return {
            "der": der_result,
            "confusion_matrix": cm,
            "classification_report": report,
            "predictions": all_predictions,
            "references": all_references,
        }
    
    def _save_detailed_results(
        self,
        predictions: np.ndarray,
        references: np.ndarray,
        save_path: Path,
        der_result: DERResult
    ):
        """
        儲存詳細的推論結果，用於分析模型弱點
        
        輸出檔案：
        1. frame_predictions.csv - 每個 frame 的預測結果
        2. segment_analysis.tsv - 片段級別的分析
        3. error_analysis.md - 錯誤分析報告
        """
        import csv
        
        class_names = self.config.class_names  # ["Child", "Adult", "Silence"]
        step_ms = self.config.step_size_ms
        
        # ===== 1. Frame-level 預測結果 (CSV) =====
        frame_csv_path = save_path / "frame_predictions.csv"
        with open(frame_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_idx", "time_start_ms", "time_end_ms", "time_str", 
                           "prediction", "pred_label", "reference", "ref_label", "correct"])
            
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                time_start_ms = i * step_ms
                time_end_ms = (i + 1) * step_ms
                time_str = self._ms_to_time_str(time_start_ms)
                pred_label = class_names[pred]
                ref_label = class_names[ref]
                correct = "Y" if pred == ref else "N"
                
                writer.writerow([
                    i, time_start_ms, time_end_ms, time_str,
                    pred, pred_label, ref, ref_label, correct
                ])
        
        print(f"  ✓ Frame 級預測結果: {frame_csv_path}")
        
        # ===== 2. Segment-level 分析 (TSV) =====
        segment_tsv_path = save_path / "segment_analysis.tsv"
        
        # 將連續相同標籤的 frame 合併成片段
        pred_segments = self._labels_to_segments_with_ref(predictions, references)
        
        with open(segment_tsv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([
                "segment_id", "start_time", "end_time", "duration_sec",
                "prediction", "reference", "status", "error_type", "notes"
            ])
            
            for idx, seg in enumerate(pred_segments):
                start_sec = seg["start_ms"] / 1000
                end_sec = seg["end_ms"] / 1000
                duration = end_sec - start_sec
                
                pred_label = class_names[seg["pred"]]
                ref_label = class_names[seg["ref_majority"]]
                
                # 判斷錯誤類型
                status, error_type, notes = self._classify_error(
                    seg["pred"], seg["ref_majority"], seg["accuracy"]
                )
                
                writer.writerow([
                    idx + 1,
                    f"{start_sec:.2f}",
                    f"{end_sec:.2f}",
                    f"{duration:.2f}",
                    pred_label,
                    ref_label,
                    status,
                    error_type,
                    notes
                ])
        
        print(f"  ✓ 片段級分析結果: {segment_tsv_path}")
        
        # ===== 3. 錯誤分析報告 (Markdown) =====
        error_report_path = save_path / "error_analysis.md"
        self._generate_error_report(
            predictions, references, pred_segments,
            der_result, error_report_path
        )
        
        print(f"  ✓ 錯誤分析報告: {error_report_path}")
    
    def _ms_to_time_str(self, ms: float) -> str:
        """將毫秒轉換為 HH:MM:SS.mmm 格式"""
        total_seconds = ms / 1000
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    
    def _labels_to_segments_with_ref(
        self,
        predictions: np.ndarray,
        references: np.ndarray
    ) -> List[Dict]:
        """
        將 frame-level 預測轉換為片段，並記錄對應的參考標籤
        """
        segments = []
        if len(predictions) == 0:
            return segments
        
        step_ms = self.config.step_size_ms
        current_pred = predictions[0]
        start_idx = 0
        
        for i in range(1, len(predictions) + 1):
            if i == len(predictions) or predictions[i] != current_pred:
                # 結束當前片段
                end_idx = i
                
                # 計算該片段內參考標籤的分佈
                ref_segment = references[start_idx:end_idx]
                ref_counts = np.bincount(ref_segment, minlength=3)
                ref_majority = np.argmax(ref_counts)
                accuracy = np.mean(predictions[start_idx:end_idx] == ref_segment)
                
                segments.append({
                    "pred": int(current_pred),
                    "start_ms": start_idx * step_ms,
                    "end_ms": end_idx * step_ms,
                    "ref_majority": int(ref_majority),
                    "ref_counts": ref_counts.tolist(),
                    "accuracy": float(accuracy),
                    "num_frames": end_idx - start_idx
                })
                
                if i < len(predictions):
                    current_pred = predictions[i]
                    start_idx = i
        
        return segments
    
    def _classify_error(
        self,
        pred: int,
        ref: int,
        accuracy: float
    ) -> Tuple[str, str, str]:
        """
        分類錯誤類型
        
        Returns:
            Tuple[status, error_type, notes]
        """
        class_names = self.config.class_names
        
        if pred == ref:
            if accuracy >= 0.9:
                return "正確", "-", ""
            else:
                return "部分正確", "邊界不準", f"準確率: {accuracy:.1%}"
        
        # 錯誤情況
        pred_is_speech = pred != 2
        ref_is_speech = ref != 2
        
        if pred_is_speech and not ref_is_speech:
            return "錯誤", "False Alarm", f"誤判 {class_names[pred]} 為語音"
        elif not pred_is_speech and ref_is_speech:
            return "錯誤", "Missed Detection", f"漏判 {class_names[ref]}"
        elif pred_is_speech and ref_is_speech and pred != ref:
            return "錯誤", "Speaker Confusion", f"混淆 {class_names[ref]} → {class_names[pred]}"
        else:
            return "錯誤", "其他", f"{class_names[ref]} → {class_names[pred]}"
    
    def _generate_error_report(
        self,
        predictions: np.ndarray,
        references: np.ndarray,
        segments: List[Dict],
        der_result: DERResult,
        output_path: Path
    ):
        """生成詳細的錯誤分析報告"""
        from datetime import datetime
        
        class_names = self.config.class_names
        step_ms = self.config.step_size_ms
        
        # 統計錯誤類型
        error_stats = defaultdict(list)
        
        for seg in segments:
            _, error_type, _ = self._classify_error(
                seg["pred"], seg["ref_majority"], seg["accuracy"]
            )
            if error_type != "-":
                error_stats[error_type].append(seg)
        
        # 找出高錯誤率的時間段
        window_size = 500  # 10秒窗口 (500 frames × 20ms)
        error_rates = []
        for i in range(0, len(predictions) - window_size, window_size // 2):
            window_pred = predictions[i:i+window_size]
            window_ref = references[i:i+window_size]
            error_rate = np.mean(window_pred != window_ref)
            start_time = i * step_ms / 1000
            end_time = (i + window_size) * step_ms / 1000
            error_rates.append((start_time, end_time, error_rate))
        
        # 排序找出最高錯誤率的時段
        error_rates.sort(key=lambda x: x[2], reverse=True)
        top_error_windows = error_rates[:10]  # 前10個高錯誤時段
        
        # 生成報告
        report = f"""# 模型錯誤分析報告 (Bi-LSTM)

**生成時間**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**音訊總時長**: {der_result.total_duration:.2f} 秒 ({der_result.total_duration/60:.1f} 分鐘)

---

## 1. 整體錯誤統計

| 錯誤類型 | 錯誤率 | 說明 |
|----------|--------|------|
| **DER (總錯誤率)** | **{der_result.der:.2f}%** | False Alarm + Missed + Confusion |
| False Alarm | {der_result.false_alarm:.2f}% | 將靜音誤判為語音 |
| Missed Detection | {der_result.missed_detection:.2f}% | 將語音誤判為靜音 |
| Speaker Confusion | {der_result.speaker_confusion:.2f}% | 語者辨識錯誤 |

---

## 2. 錯誤類型分佈

"""
        # 錯誤類型統計
        for error_type, segs in error_stats.items():
            total_duration = sum(
                (s["end_ms"] - s["start_ms"]) / 1000 for s in segs
            )
            report += f"### {error_type}\n\n"
            report += f"- **錯誤片段數**: {len(segs)}\n"
            report += f"- **累計時長**: {total_duration:.2f} 秒\n\n"
            
            # 顯示前5個典型錯誤
            report += "**典型錯誤片段**:\n\n"
            report += "| 時間範圍 | 預測 | 實際 | 時長 |\n"
            report += "|----------|------|------|------|\n"
            
            for seg in segs[:5]:
                start = seg["start_ms"] / 1000
                end = seg["end_ms"] / 1000
                duration = end - start
                pred_label = class_names[seg["pred"]]
                ref_label = class_names[seg["ref_majority"]]
                report += f"| {start:.2f}s - {end:.2f}s | {pred_label} | {ref_label} | {duration:.2f}s |\n"
            
            report += "\n"
        
        # 高錯誤率時段
        report += """---

## 3. 高錯誤率時段（建議重點分析）

以下時段的錯誤率較高，建議對照原始錄音進行詳細分析：

| 排名 | 時間範圍 | 錯誤率 | 建議動作 |
|------|----------|--------|----------|
"""
        for i, (start, end, rate) in enumerate(top_error_windows[:10]):
            action = "重點分析" if rate > 0.3 else "檢查邊界"
            report += f"| {i+1} | {start:.1f}s - {end:.1f}s | {rate:.1%} | {action} |\n"
        
        # 語者切換分析
        report += """
---

## 4. 語者切換分析

"""
        # 找出所有語者切換點
        switches = []
        for i in range(1, len(references)):
            if references[i] != references[i-1]:
                time_sec = i * step_ms / 1000
                pred_correct = predictions[i] == references[i]
                switches.append({
                    "time": time_sec,
                    "from": class_names[references[i-1]],
                    "to": class_names[references[i]],
                    "pred_correct": pred_correct
                })
        
        correct_switches = sum(1 for s in switches if s["pred_correct"])
        switch_accuracy = correct_switches / len(switches) * 100 if switches else 0
        
        report += f"- **語者切換總數**: {len(switches)}\n"
        report += f"- **切換點預測正確率**: {switch_accuracy:.1f}%\n\n"
        
        # 分析哪種切換容易出錯
        switch_errors = defaultdict(lambda: {"total": 0, "errors": 0})
        for s in switches:
            key = f"{s['from']} → {s['to']}"
            switch_errors[key]["total"] += 1
            if not s["pred_correct"]:
                switch_errors[key]["errors"] += 1
        
        report += "**各類切換錯誤率**:\n\n"
        report += "| 切換類型 | 總次數 | 錯誤次數 | 錯誤率 |\n"
        report += "|----------|--------|----------|--------|\n"
        
        for switch_type, counts in sorted(switch_errors.items(), 
                                         key=lambda x: x[1]["errors"], reverse=True):
            error_rate = counts["errors"] / counts["total"] * 100
            report += f"| {switch_type} | {counts['total']} | {counts['errors']} | {error_rate:.1f}% |\n"
        
        # 改善建議
        report += """
---

## 5. 改善建議

根據以上分析，建議從以下方向進行改善：

### 5.1 針對 False Alarm（靜音誤判為語音）
- 檢查靜音片段是否包含背景噪音
- 考慮調整靜音檢測閾值
- 可考慮加入後處理平滑

### 5.2 針對 Missed Detection（語音漏判）
- 檢查是否為低音量語音
- 分析是否為特定語者的語音特徵不明顯
- 考慮增加訓練資料中的邊緣案例

### 5.3 針對 Speaker Confusion（語者混淆）
- 分析混淆主要發生在哪兩個語者之間
- 檢查是否為語者聲音相似的片段
- 考慮增加語者特徵的區分度

### 5.4 針對邊界檢測
- 語者切換點附近通常錯誤率較高
- 考慮使用後處理平滑或邊界優化

---

## 6. 輸出檔案說明

| 檔案 | 說明 |
|------|------|
| `frame_predictions.csv` | 每個 20ms frame 的預測結果 |
| `segment_analysis.tsv` | 連續片段的分析，含錯誤分類 |
| `error_analysis.md` | 本報告 |
| `gantt_chart.png` | 前 2 分鐘預測與參考的甘特圖 |
| `gantt_comparison.png` | 多時段比較圖 |
| `gantt_full_timeline.png` | 完整錄音時間軸甘特圖 |
| `confusion_matrix.png` | 混淆矩陣 |

---

*報告由 Whisper + Bi-LSTM 評估系統自動生成*
"""
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
    
    def _print_results(
        self,
        der: DERResult,
        cm: np.ndarray,
        report: Dict
    ):
        """印出評估結果 (3 分類)"""
        print("\n" + "=" * 60)
        print("評估結果 (3 分類: Child, Adult, Silence)")
        print("=" * 60)
        
        print(f"\n【DER】")
        print(f"  Diarization Error Rate: {der.der:.2f}%")
        print(f"  False Alarm Rate: {der.false_alarm:.2f}%")
        print(f"  Missed Detection Rate: {der.missed_detection:.2f}%")
        print(f"  Speaker Confusion Rate: {der.speaker_confusion:.2f}%")
        
        print(f"\n【分類準確率】")
        for class_name in self.config.class_names:
            if class_name.lower() in report:
                metrics = report[class_name.lower()]
            elif class_name in report:
                metrics = report[class_name]
            else:
                continue
            
            print(f"  {class_name}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1-score: {metrics['f1-score']:.4f}")
        
        print(f"\n  Overall Accuracy: {report.get('accuracy', 0):.4f}")
        
        print("\n【混淆矩陣】")
        print("           ", end="")
        for name in self.config.class_names:
            print(f"{name[:7]:>8}", end="")
        print()
        
        for i, name in enumerate(self.config.class_names):
            print(f"  {name[:10]:<10}", end="")
            for j in range(len(self.config.class_names)):
                print(f"{cm[i, j]:>8}", end="")
            print()
        
        print("=" * 60)


# ==================== 推論函式 ====================

def inference_on_audio(
    model: nn.Module,
    audio_path: str,
    config: Config,
    srt_path: Optional[str] = None,
    save_dir: Optional[str] = None
) -> Dict:
    """
    對單一音訊檔案進行推論
    
    Args:
        model: 模型
        audio_path: 音訊檔案路徑
        config: 配置
        srt_path: 標註檔案路徑（可選，用於評估）
        save_dir: 結果儲存目錄
        
    Returns:
        Dict: 推論結果
    """
    from dataset import (
        AudioProcessor, SRTParser, FrameLabelGenerator,
        AudioSample, SpeakerDiarizationDataset
    )
    
    print(f"\n對音訊進行推論: {audio_path}")
    
    # 載入音訊
    audio_processor = AudioProcessor(config.sample_rate)
    waveform, duration = audio_processor.load_audio(audio_path)
    
    print(f"音訊時長: {duration:.2f} 秒 ({duration/60:.2f} 分鐘)")
    
    # 建立資料集
    sample = AudioSample(
        subject_id="inference",
        audio_path=audio_path,
        srt_path=srt_path if srt_path else "",
    )
    sample.waveform = waveform
    sample.duration = duration
    
    # 如果有標註檔案，載入標籤
    if srt_path and os.path.exists(srt_path):
        srt_parser = SRTParser()
        label_generator = FrameLabelGenerator(config.step_size_ms)
        segments = srt_parser.parse_file(srt_path)
        sample.labels = label_generator.generate_labels(segments, duration)
    
    dataset = SpeakerDiarizationDataset(
        samples=[sample],
        config=config,
        is_training=False,
        preload=False
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0  # 單一檔案推論不需要多進程
    )
    
    # 進行推論
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="推論中"):
            audio = batch["audio"].to(config.device)
            
            if config.use_amp:
                with autocast():
                    logits = model(audio)
            else:
                logits = model(audio)
            
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            all_predictions.extend(predictions)
    
    # 合併預測結果
    all_predictions = np.concatenate(all_predictions)
    
    # 如果有參考標籤，進行評估
    if sample.labels is not None:
        evaluator = Evaluator(config)
        
        # 對齊長度
        min_len = min(len(all_predictions), len(sample.labels))
        predictions = all_predictions[:min_len]
        references = sample.labels[:min_len]
        
        # 計算 DER
        der_result = evaluator.der_calculator.calculate(predictions, references)
        
        print(f"\nDER: {der_result.der:.2f}%")
        print(f"Speaker Confusion: {der_result.speaker_confusion:.2f}%")
        
        # 儲存結果
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # 繪製甘特圖
            evaluator.gantt_visualizer.plot(
                predictions,
                references,
                save_path=save_path / "inference_gantt.png"
            )
            
            # 繪製混淆矩陣
            cm = evaluator.cm_visualizer.compute(predictions, references)
            evaluator.cm_visualizer.plot(
                cm,
                save_path=save_path / "inference_confusion_matrix.png"
            )
        
        return {
            "predictions": all_predictions,
            "references": sample.labels,
            "der": der_result,
        }
    
    return {
        "predictions": all_predictions,
    }


def predictions_to_srt(
    predictions: np.ndarray,
    step_size_ms: float = 20.0,
    output_path: str = "output.srt"
) -> str:
    """
    將預測結果轉換為 SRT 格式 (3 分類)
    
    Args:
        predictions: 預測標籤 (0=Child, 1=Adult, 2=Silence)
        step_size_ms: Frame step size
        output_path: 輸出路徑
        
    Returns:
        str: SRT 檔案路徑
    """
    from dataset import FrameLabelGenerator
    
    # 轉換為片段
    gantt = GanttChartVisualizer(step_size_ms)
    segments = gantt._labels_to_segments(predictions)
    
    # 3 分類類別名稱
    class_names = ["小孩", "測試者", "靜音"]
    
    srt_content = []
    index = 1
    
    for label, start, end in segments:
        if label == 2:  # 跳過靜音 (Silence = 2)
            continue
        
        # 格式化時間
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            ms = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
        
        start_str = format_time(start)
        end_str = format_time(end)
        
        speaker = class_names[label]
        
        srt_content.append(f"{index}")
        srt_content.append(f"{start_str} --> {end_str}")
        srt_content.append(f"{speaker}: [語音]")
        srt_content.append("")
        
        index += 1
    
    # 寫入檔案
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_content))
    
    print(f"SRT 檔案已儲存至: {output_path}")
    
    return output_path


# ==================== 主程式入口 ====================

if __name__ == "__main__":
    import argparse
    from model import load_model
    from dataset import create_test_dataloader
    
    parser = argparse.ArgumentParser(description="評估 Whisper 語者分離模型")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="模型權重路徑"
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default=None,
        help="音訊檔案路徑（單一檔案推論）"
    )
    parser.add_argument(
        "--srt-path",
        type=str,
        default=None,
        help="SRT 標註檔案路徑"
    )
    
    args = parser.parse_args()
    
    # 載入配置
    config = Config()
    
    # 載入模型
    model = load_model(config, args.model_path)
    
    if args.audio_path:
        # 單一檔案推論
        results = inference_on_audio(
            model=model,
            audio_path=args.audio_path,
            config=config,
            srt_path=args.srt_path,
            save_dir=config.output_dir
        )
    else:
        # 使用預設測試集
        test_loader = create_test_dataloader(config)
        
        evaluator = Evaluator(config)
        results = evaluator.evaluate(
            model=model,
            data_loader=test_loader,
            save_dir=config.output_dir
        )
