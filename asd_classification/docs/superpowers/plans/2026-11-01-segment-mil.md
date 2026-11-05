# 對話段落注意力模型（segment-MIL）實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 依 `research_plan_1030.md` 蓋出「對答 → 段落 → 閘門＋注意力小網路」的分類管線，先在合成資料驗證，再用已轉錄的 10 位 ASD 對 19 位 TD 跑一次（初步結果）。

**Architecture:** 新增 `src/asdpipe/mil/` 套件。段落層級的 6 個量測只用逐字稿；孩子層級的 4 個聲學量測直接取現有 `extract_corpus` 的輸出。模型的參數帶一個「模型」維度，一次平行訓練幾百個小模型（不同種子、不同打亂標籤），1,000 次打亂才跑得完。

**Tech Stack:** Python（`D:\Anaconda\envs\paper\python.exe`）、torch 2.8（CUDA）、scikit-learn 1.7、librosa（pYIN）、parselmouth 0.4.7（Praat）、torchcrepe、CKIP 斷詞與 ECAPA（既有 backends）。

**Spec:** `asd_classification/research_plan_1030.md`（下稱「計畫」）。工程規範見 `asd_classification/CLAUDE.md`。

**範圍：** 只做計畫第三節（分類）。第四節的次要分析要等 ADOS 逐項分數到手，另開計畫處理；它的定義已經寫在計畫裡，跟著這次一起凍結。

## Global Constraints

- R2：一位孩子是一個單位；段落不是獨立樣本，標籤和切分都在孩子層級。
- R3：補缺值、標準化、閘門、注意力、模型，全部只用訓練折的孩子估計。
- R4：不新增任何預訓練模型；只用 torchcrepe、pYIN、Praat/parselmouth、CKIP、ECAPA。不安裝新套件。
- 真實資料（`D:/自閉症/data/Fine_tune`）只讀、不進 git；輸出只寫進已 gitignore 的 `outputs/`、`cache/`。
- 報告、測試、commit、對話裡都不出現孩子姓名或資料夾名；一律用 repo 根目錄 `_private_id_map.csv`（TD，欄位 `pid,fold,session`）和 `_private_id_map_asd.csv`（ASD，欄位 `uid,session`）的編號。
- 凍結 commit（Task 9）之前，只能算 TD；ASD 一律不碰。
- 搜尋 `outputs/` 一律排除 `_quarantine_asd_exposed/`，也不開啟它。
- GPU：`ASDPIPE_GPU_MEM_FRACTION=0.15`，模型分批（`mil.chunk_models`）。
- 程式識別字和註解用英文；文件用繁體中文。
- 只做本計畫列出的 commit，而且逐一指定路徑。`自閉症兒童語音-聲學特徵臨床評估表.docx` 沒有被 gitignore，絕對不能 stage。
- 測試指令：`cd D:\PycharmProjects\paper\asd_classification; D:\Anaconda\envs\paper\python.exe -m pytest tests/<file> -q`

## Review Focus

1. 對答不到 5 組、或整場孩子幾乎沒說話：整場只有一個段落，孩子的量測全是缺值。不能當掉，注意力只落在那一段（Task 2、4 各有測試）。
2. 重疊語句（孩子在醫師說完前就開口）：既不是回應，也不是孩子起頭（Task 2）。
3. 某個量測在訓練折裡全是缺值：中位數算不出來，標準化後要補 0，不能產生 NaN（Task 4）。
4. 分批大小改變結果：每個模型有自己的亂數產生器，`chunk_models` 不管設多少，結果都要一樣（Task 5）。
5. 編號表找不到某場：要報錯，但錯誤訊息只能寫是哪個 cohort、第幾筆，不能印出資料夾名（Task 7）。

---

### Task 1: 把預設值和實作細節寫進計畫與設定檔

**Files:**
- Modify: `research_plan_1030.md`（3.3、3.4、3.5、3.6、第七節；新增 3.8）
- Modify: `configs/pipeline.yaml`（新增 `mil:`）
- Modify: `configs/decision_rules.yaml`（新增 `mil_analysis:`；舊的 `primary_model`、`honesty_clause`、`comparison`、`criterion_validity.decision_table` 只加註「被 D17 取代」，不刪除）
- Modify: `DEVIATIONS.md`（新增 D17）

**Interfaces:**
- Produces: `cfg["mil"]`、`rules["mil_analysis"]`，鍵名如下，後面各 task 照用。

- [ ] **Step 1: 計畫文件的修改**（全部是凍結前的定義釐清，沒有用到任何 ASD 資料）
  - 第七節「待決定」改成「已決定（2026-11-01，全部採預設）」：A11、F4、門檻 0.70、第 5 步人工檢查依左欄、音高失敗率 AUC 0.30–0.70。
  - 3.3 F3：整場出現 ≥ 3 次的 2–5 詞詞串（CKIP 斷詞，排除只含虛詞的，沿用 `features.e_stream.ngram_range`、`ngram_min_repeats`）算固定詞串；段落值 ＝ 孩子在段落內的詞，有多少比例落在固定詞串裡。
  - 3.3 F4：改成「每 100 字有幾個你／妳」。理由：過去罕見事件的比例都沒通過一致性，這個比率版本 r = 0.72（`e_clinical.py` 文件）。
  - 3.3 F1：段落內每組「醫師起頭且孩子有回應」的對答，算字元雙字重疊比例後取平均。
  - 3.3 C9、C1、F4 都只數漢字。
  - 3.4 一致性：改成「單雙數句交錯分半」。把孩子的句子依時間排序，單數句一半、雙數句一半，醫師的句子也同樣分；B7 每半各自切聲紋視窗。統計量是兩半之間、跨孩子的 Pearson r，不做 Spearman-Brown 校正。過去那些 r 值的算法沒有留下程式，所以這裡採保守版。
  - 3.4 音高：三法是 CREPE、pYIN、Praat 自相關法。每場取孩子說話的前 60 秒；CREPE 跟 pYIN、跟 Praat 的一致率都要 > 0.90，八度錯誤都要 < 0.03，才算這場通過。沒通過的場次，該場的 A2、B2 當缺值；TD 超過一半場次沒通過，A2、B2 整欄留空。
  - 3.5 固定設定（數值同 `mil:` 設定）；95% 區間的算法是：5 次重複的折外分數取平均，再對孩子做分層 bootstrap 2,000 次。
  - 3.6 「拿掉曝光的 6 位」：要有 ≥ 10 位未曝光的 ASD 才跑，所以這次不跑。
  - 新增 3.8：這次是初步結果（10 位 ASD）。20 位轉錄齊之後，用同一套設定再跑一次，那次才是正式結果，中間不改任何設定。醫師回覆錄音條件之前，「只用逐字稿」的版本也一起跑、一起做打亂檢定；醫師說條件不同，就以它為主。
- [ ] **Step 2: `configs/pipeline.yaml` 新增**

```yaml
mil:
  max_gap_s: 10.0
  segment_length: 5
  sensitivity_lengths: [3, 7]
  model: {hidden: 8, epochs: 300, lr: 0.01, weight_decay: 1.0e-4,
          gate_sigma: 0.5, gate_init: 0.5, gate_lambda: 0.05}
  cv: {n_folds: 5, repeats: 5, seeds: 5}
  logreg_C: 1.0
  n_permutations: 1000
  permutation_seed: 20261101
  n_boot: 2000
  chunk_models: 250
  device: auto            # cuda when available
```

- [ ] **Step 3: `configs/decision_rules.yaml` 新增 `mil_analysis:`**，內容：`frozen_at: "2026-11-01"`、`separable_if: "permutation p <= 0.05"`、`split_half: {threshold: 0.70, statistic: pearson_odd_even_uncorrected}`、`f0_session: {min_agreement: 0.90, max_octave_error: 0.03, drop_if_td_fail_share_above: 0.5}`、`f0_group_rule: {feature: c_f0_tracking_failure_ratio, auc_keep_range: [0.30, 0.70]}`、`recording_condition: {pending: true, if_different: transcript_only}`、`unexposed_only_min_asd: 10`、`outcome: {}`（Task 8 填入）。
- [ ] **Step 4: DEVIATIONS D17**（白話）：研究設計改以計畫為準；R5 的主模型 M1 和誠實條款不再適用，改成段落注意力模型對照邏輯迴歸，不檢定兩者差異；這次先用 10 位 ASD 跑初步結果；列出 Step 1 的定義釐清。
- [ ] **Step 5: 確認設定讀得到**

Run: `D:\Anaconda\envs\paper\python.exe -c "import sys; sys.path.insert(0,'src'); from asdpipe.config import load_pipeline_config, load_decision_rules; print(load_pipeline_config()['mil']['segment_length'], load_decision_rules()['mil_analysis']['split_half']['threshold'])"`
Expected: `5 0.7`

---

### Task 2: 對答切分、段落、6 個段落量測

**Files:**
- Create: `src/asdpipe/mil/__init__.py`、`src/asdpipe/mil/segments.py`
- Test: `tests/test_mil_segments.py`

**Interfaces:**
- Consumes: `contracts.Turn/Speaker`、`features.e_text.pair_turns/_overlap_ratio/FUNCTION_WORDS`、`features.e_clinical._SECOND`
- Produces:
  - `Exchange(kind: str, examiner: Turn | None, child: Turn | None, t_start: float, t_end: float)`（frozen dataclass；kind 是 `"examiner"` 或 `"child"`）
  - `split_exchanges(turns, max_gap_s=10.0) -> list[Exchange]`
  - `segment_index(n_exchanges: int, length: int) -> list[tuple[int, int]]`
  - `fixed_phrases(word_lists, n_range, min_repeats) -> set[tuple[str, ...]]`
  - `segment_measures(turns, words: Mapping[int, Sequence[str]], cfg, length: int) -> np.ndarray`，形狀 `(n_segments, 6)`，欄位順序照 `SEGMENT_MEASURES`
  - 常數：
    - `SEGMENT_MEASURES = ("e_f3_fixed_phrase_cover", "e_f1_echo_bigram", "e_e5_talk_share", "e_f4_you_per_100c", "c_c9_chars_per_utt", "c_c1_chars_per_s")`
    - `CHILD_MEASURES = ("c_exaggerated_utterance_ratio", "b7_mean_pairwise_cosdist", "c_pitch_range_st_median", "b_level_rel_examiner_db")`
    - `PITCH_MEASURES = ("c_exaggerated_utterance_ratio", "c_pitch_range_st_median")`
    - `STREAMS = {"E": SEGMENT_MEASURES[:4], "C": ("c_c9_chars_per_utt", "c_c1_chars_per_s", "c_exaggerated_utterance_ratio"), "B": ("b7_mean_pairwise_cosdist", "c_pitch_range_st_median", "b_level_rel_examiner_db")}`

規則：
- 醫師的每一句都是一組「醫師起頭」，孩子的回應取 `pair_turns` 的配對（沒有回應就是 `None`），時間範圍是醫師開口到兩人中較晚結束的時間。
- 孩子的一句話如果在開口前 `max_gap_s` 秒內醫師都沒說話（沒有任何醫師句子滿足 `e.t_start < c.t_start` 且 `e.t_end > c.t_start - max_gap_s`），就是一組「孩子起頭」。
- 對答依開始時間排序。段落 ＝ 連續 `length` 組；對答不到 `length` 組時，整場當一段。
- 段落包含哪些句子：開始時間落在 [段落第一組的開始, 段落內最晚的結束) 之間的所有句子，兩人都算。
- 只有 `Speaker.CHILD` 和 `Speaker.EXAMINER` 參與，`OVERLAP` 忽略。
- 量測算不出來時給 NaN。例外是 E5：只要有人說話，就照算，孩子沒說話就是 0。

- [ ] **Step 1: 寫失敗的測試**（用 `conftest.make_turn` 手工造句子）

```python
def test_examiner_led_with_and_without_response():
    # E 0-2 "你叫什麼名字", C 3-4 "我叫小明", E 20-22 "好", (no reply)
    ex = split_exchanges(turns)
    assert [e.kind for e in ex] == ["examiner", "examiner"]
    assert ex[0].child.text == "我叫小明" and ex[1].child is None

def test_child_led_needs_ten_seconds_of_examiner_silence():
    # E 0-2, C 13-14 (11 s after) -> child-led ; C 5-6 after E 0-2 -> response, not child-led
    assert [e.kind for e in split_exchanges(turns)] == ["examiner", "child"]

def test_overlap_and_continuation_turns_are_not_exchanges():
    # child starts at 1.5 while examiner speaks 0-2; child's second turn after a reply
    assert sum(e.kind == "child" for e in split_exchanges(turns)) == 0

def test_segment_index():
    assert segment_index(7, 5) == [(0, 5), (1, 6), (2, 7)]
    assert segment_index(3, 5) == [(0, 3)] and segment_index(0, 5) == []

def test_measures_hand_computed():
    # one segment: E 0-7 "你好", C 7-10 "你好嗎" (3 s child, 7 s examiner)
    m = dict(zip(SEGMENT_MEASURES, segment_measures(turns, words, cfg, 5)[0]))
    assert m["e_e5_talk_share"] == pytest.approx(0.3)
    assert m["e_f1_echo_bigram"] == pytest.approx(0.5)          # {你好,好嗎} vs {你好}
    assert m["e_f4_you_per_100c"] == pytest.approx(100 / 3)
    assert m["c_c9_chars_per_utt"] == pytest.approx(3.0)
    assert m["c_c1_chars_per_s"] == pytest.approx(1.0)

def test_f3_cover_uses_session_level_phrases():
    # child says ["恐龍","好","厲害"] three times across the session -> fixed phrase
    # a segment holding one of them plus ["我","要"] -> cover = 3 / 5
def test_f3_ignores_function_word_only_ngrams():
    assert fixed_phrases([["然後", "就是"]] * 5, (2, 5), 3) == set()

def test_silent_child_gives_nan_but_e5_zero():
    row = segment_measures(examiner_only_turns, {}, cfg, 5)[0]
    assert np.isnan(row[[0, 1, 3, 4, 5]]).all() and row[2] == 0.0
```

- [ ] **Step 2: 跑測試確認失敗**　Run: `...pytest tests/test_mil_segments.py -q`　Expected: `ModuleNotFoundError: asdpipe.mil`
- [ ] **Step 3: 實作 `segments.py`**：F1 直接呼叫 `_overlap_ratio(child.text, examiner.text, 2)`；F3 涵蓋率用每句詞串的 n-gram 比對 `fixed_phrases` 的集合，標記被涵蓋的詞位置；漢字用 `re.compile(r"[一-鿿]")` 計數。
- [ ] **Step 4: 跑測試確認通過**　Expected: 8 passed

---

### Task 3: 資料容器 `Bags`（真實與合成共用）

**Files:**
- Create: `src/asdpipe/mil/bags.py`
- Test: `tests/test_mil_bags.py`

**Interfaces:**
- Produces:
  - `Bags`（frozen dataclass）：
    - `ids: tuple[str, ...]`
    - `y: np.ndarray (N,) int64`
    - `seg: np.ndarray (N, T, Ds) float32`：缺值是 NaN，補位列是 0
    - `mask: np.ndarray (N, T) bool`
    - `child: np.ndarray (N, Dc) float32`
    - `seg_names: tuple[str, ...]`
    - `child_names: tuple[str, ...]`
    - `exposed: np.ndarray (N,) bool`
  - `Bags.select(seg_names=None, child_names=None, rows=None) -> Bags`：照給定的名稱和列取子集，保留原順序；給空 tuple 代表那一層不用
  - `bags_from_arrays(ids, y, seg_list: Sequence[np.ndarray], child: np.ndarray, seg_names, child_names, exposed) -> Bags`
  - `synthetic_bags(n_pos, n_neg, *, signal: float, seed: int, n_seg=(20, 60), d_seg=6, d_child=4, informative_seg=(0,), informative_child=(0,), signal_fraction=0.3, nan_rate=0.05) -> Bags`

合成規則：所有值 ~ N(0, 1)。ASD 孩子隨機挑 `signal_fraction` 的段落，在 `informative_seg` 欄加上 `signal`；孩子層級的 `informative_child` 欄也加 `signal`。最後依 `nan_rate` 隨機挖空。

- [ ] **Step 1: 寫失敗的測試**

```python
def test_padding_and_mask():
    b = bags_from_arrays(("a", "b"), [0, 1], [np.ones((2, 6)), np.ones((5, 6))], np.zeros((2, 4)), SEG6, CH4, [False, False])
    assert b.seg.shape == (2, 5, 6) and b.mask.sum(1).tolist() == [2, 5] and (b.seg[0, 2:] == 0).all()

def test_select_keeps_order():
    s = b.select(seg_names=("e_e5_talk_share",), child_names=(), rows=np.array([1]))
    assert s.seg.shape[2] == 1 and s.child.shape == (1, 0) and s.ids == ("b",)

def test_synthetic_null_and_signal():
    null = synthetic_bags(200, 200, signal=0.0, seed=0)
    assert abs(np.nanmean(null.child[null.y == 1, 0]) - np.nanmean(null.child[null.y == 0, 0])) < 0.2
    sig = synthetic_bags(200, 200, signal=2.0, seed=0)
    assert np.nanmean(sig.child[sig.y == 1, 0]) - np.nanmean(sig.child[sig.y == 0, 0]) > 1.5
```

- [ ] **Step 2: 確認失敗**　Expected: `ImportError`
- [ ] **Step 3: 實作 `bags.py`**
- [ ] **Step 4: 確認通過**　Expected: 3 passed

---

### Task 4: 閘門＋段落注意力模型（批次平行訓練）

**Files:**
- Create: `src/asdpipe/mil/model.py`
- Test: `tests/test_mil_model.py`

**Interfaces:**
- Consumes: `Bags`
- Produces:
  - `MILConfig(hidden=8, epochs=300, lr=0.01, weight_decay=1e-4, gate_sigma=0.5, gate_init=0.5, gate_lambda=0.05)`，搭配 `MILConfig.from_cfg(cfg)`（讀 `cfg["mil"]["model"]`）
  - `n_parameters(d_seg, d_child, hidden) -> int`
  - `prepare(bags, train_mask: np.ndarray (M, N) bool) -> tuple[np.ndarray, np.ndarray]`：回傳 `(M, N, T, Ds)` 和 `(M, N, Dc)` 兩個陣列。每個模型只用自己的訓練孩子算中位數補缺值，再標準化（sd 下限 1e-6）。訓練折整欄缺值時補 0。
  - `MILFit(scores: (M, N) logits, gates: (M, Ds + Dc), attention: (M, N, T))`
  - `fit_predict(bags, train_mask, labels: (M, N) int, seeds: Sequence[int], cfg: MILConfig, device="cpu") -> MILFit`

演算法（signature 決定不了的部分）：

```
gates: 用 stochastic gates（Yamada et al., ICML 2020）
  訓練時 z = clamp(mu + sigma * eps, 0, 1)，eps 每步每個模型重抽；預測時 z = clamp(mu, 0, 1)
h      = tanh(einsum("mntd,mdh->mnth", x * z_seg, W1) + b1)
s      = einsum("mnth,mh->mnt", h, w2) + b2            # 每段落像 ASD 的程度
a      = masked_softmax(einsum("mnth,mh->mnt", h, q))  # 注意力，只在有效段落上
logit  = sum_t(a * s) + einsum("mnd,md->mn", c * z_child, wc) + b0
loss_m = 加權 BCE（訓練孩子平均；pos_weight = 訓練折 TD 數 / ASD 數）
       + gate_lambda * sum_d Phi(mu_d / sigma)
       + weight_decay * (|W1|² + |w2|² + |q|² + |wc|²)
總 loss = sum_m loss_m；Adam(lr)，全批次，固定 epochs，不提早停止
初始化：每個模型用自己的 torch.Generator(seed)：
  W1 ~ N(0, 1/Ds)；w2、q ~ N(0, 1/H)；wc ~ N(0, 0.01)；bias 0；mu = gate_init
  eps 也從這個 generator 抽
```

- [ ] **Step 1: 寫失敗的測試**

```python
def test_parameter_count_is_about_100():
    assert n_parameters(6, 4, 8) == 88

def test_models_in_a_batch_are_independent():
    alone = fit_predict(b, tm[:1], lab[:1], [7], cfg)
    pair = fit_predict(b, tm[[0, 0]], lab[[0, 0]], [7, 8], cfg)
    np.testing.assert_allclose(alone.scores[0], pair.scores[0], atol=1e-5)

def test_learns_planted_signal_on_held_out_children():
    b = synthetic_bags(40, 40, signal=1.5, seed=0); train = rows % 2 == 0
    f = fit_predict(b, train[None], b.y[None], [0], cfg)
    assert safe_auc(b.y[~train], f.scores[0, ~train]) > 0.8

def test_gates_keep_signal_and_close_noise():
    f = fit_predict(b, all_rows, b.y[None].repeat(5, 0), range(5), cfg)
    g = f.gates.mean(0)
    assert g[0] > 0.5 and g[Ds] > 0.5                      # informative seg / child gate open
    assert np.delete(g, [0, Ds]).mean() < 0.5               # noise gates mostly closed

def test_prepare_uses_training_rows_only():
    x1, _ = prepare(b, tm); b2 = b with a test child's seg *= 1000; x2, _ = prepare(b2, tm)
    np.testing.assert_allclose(x1[0][tm[0]], x2[0][tm[0]])

def test_all_nan_column_and_single_segment_bag():
    f = fit_predict(bag_with_T1_and_an_all_nan_column, ...)
    assert np.isfinite(f.scores).all()
    assert np.allclose(f.attention[0, 0, 0], 1.0)
```

- [ ] **Step 2: 確認失敗**
- [ ] **Step 3: 實作 `model.py`**
- [ ] **Step 4: 確認通過。** 只有 `test_gates_keep_signal_and_close_noise` 失敗時，才在合成資料上依序試 `gate_lambda` ∈ {0.02, 0.05, 0.1, 0.2}，取能通過的最小值；`epochs` 不夠時改成 500。改動寫回 `pipeline.yaml`，並記進 D17。這些調整只看合成資料，所以合規。Expected: 6 passed

---

### Task 5: 交叉驗證、邏輯迴歸基線、bootstrap、打亂檢定

**Files:**
- Create: `src/asdpipe/mil/evaluate.py`
- Test: `tests/test_mil_evaluate.py`

**Interfaces:**
- Consumes: `fit_predict`、`MILConfig`、`Bags`、`cv.splits.subject_folds(y, n_folds, seed)`、`eval.metrics.safe_auc`
- Produces:
  - `CVSettings(n_folds=5, repeats=5, seeds=5, n_boot=2000, chunk_models=250, device="cpu")`，搭配 `CVSettings.from_cfg(cfg)`
  - `label_sets(y, n_perm, seed) -> np.ndarray (1 + n_perm, N)`：第 0 列是真實標籤
  - `oof_mil(bags, labels: (P, N), mcfg, cv) -> tuple[np.ndarray (P, repeats, N), np.ndarray (P, repeats, n_folds, Ds + Dc)]`：回傳折外分數（5 個種子平均）和各折閘門值
  - `child_means(bags) -> np.ndarray (N, Ds + Dc)`：每個段落量測對有效段落取 nanmean，再接上孩子層級量測
  - `oof_logreg(bags, labels, cv, C) -> np.ndarray (P, repeats, N)`：每折一條 `SimpleImputer(median) → StandardScaler → LogisticRegression(C, class_weight="balanced")` pipeline
  - `mean_auc(y, oof (repeats, N)) -> float`
  - `bootstrap_ci(y, score (N,), n_boot, seed) -> tuple[float, float]`：依組別分層重抽孩子
  - `permutation_p(observed, null) -> float`，算式 `(1 + #{null >= observed}) / (1 + len(null))`
  - `evaluate(bags, mcfg, cv, n_perm, perm_seed, C) -> dict`：回傳鍵 `mil` 和 `logreg`，各自含 `auc`、`ci`、`p`、`null`、`oof_mean`；`mil` 另含 `gates_by_fold`

批次方式：每組標籤 p、每次重複 r，用 `subject_folds(labels[p], n_folds, seed=r)` 依「這組標籤」重新分層切折，每折每種子一個模型，排成 `(M, N)` 的訓練遮罩和標籤，每 `chunk_models` 個模型丟一次 `fit_predict`。模型的種子 ＝ `perm_seed + 100_000 * p + 1_000 * r + 10 * fold + s`，跟分批大小無關。

- [ ] **Step 1: 寫失敗的測試**

```python
def test_null_auc_is_near_half():
    aucs = [evaluate(synthetic_bags(20, 19, signal=0.0, seed=k), mcfg, cv, 0, 0, 1.0)["mil"]["auc"] for k in range(6)]
    assert abs(np.mean(aucs) - 0.5) < 0.08

def test_signal_auc_is_high_for_both_models():
    r = evaluate(synthetic_bags(20, 19, signal=1.5, seed=1), mcfg, cv, 0, 0, 1.0)
    assert r["mil"]["auc"] > 0.8 and r["logreg"]["auc"] > 0.8

def test_every_child_is_tested_once_per_repeat():   # R2/R3: built from Fold, asserted on the masks

def test_chunk_size_does_not_change_results():
    a = oof_mil(b, labels, mcfg, cv_chunk_25)[0]; c = oof_mil(b, labels, mcfg, cv_chunk_125)[0]
    np.testing.assert_allclose(a, c, atol=1e-5)

def test_permutation_p():
    assert permutation_p(0.9, np.full(99, 0.5)) == pytest.approx(0.01)
    assert permutation_p(0.5, np.full(99, 0.5)) == pytest.approx(1.0)

def test_strong_signal_gives_smallest_p():
    r = evaluate(synthetic_bags(20, 19, signal=2.5, seed=2), mcfg, cv, 19, 0, 1.0)
    assert r["mil"]["p"] == pytest.approx(1 / 20)

def test_bootstrap_ci_brackets_point():
    lo, hi = bootstrap_ci(y, s, 2000, 0); assert lo <= safe_auc(y, s) <= hi
```

- [ ] **Step 2: 確認失敗**
- [ ] **Step 3: 實作 `evaluate.py`**
- [ ] **Step 4: 確認通過**　Expected: 7 passed（CPU 上整檔約 1–2 分鐘）

---

### Task 6: 建模前的自動檢查（只用 TD）

**Files:**
- Create: `src/asdpipe/mil/checks.py`
- Test: `tests/test_mil_checks.py`

**Interfaces:**
- Consumes:
  - `b_voice.ecapa_windows`、`b_voice.BStreamFeatures(...).b7()`、`b_voice._level_rel_examiner`
  - `features.common.rms_db/slice_turn`
  - `backends.track_f0/embed_windows/segment`
  - `eval.convention_gate.TRANSFORMS/apply_transform/gate_shifts`、`normalize.normalize_corpus`
  - `segments.split_exchanges/segment_measures`
- Produces:
  - `split_half_r(odd: np.ndarray, even: np.ndarray) -> float`：跨孩子的 Pearson r，只用兩半都有值的孩子；少於 5 位時回傳 NaN
  - `b7_a11_halves(subject: SubjectRecord, cfg, backends) -> dict[str, tuple[float, float]]`：鍵是 `"b7_mean_pairwise_cosdist"`、`"b_level_rel_examiner_db"`
  - `f0_three_way(audio, sr, child_turns, cfg, backends, max_seconds=60.0) -> dict`：鍵是 `agree_pyin`、`agree_praat`、`octave_pyin`、`octave_praat`、`n_frames`
  - `f0_session_pass(check, min_agreement=0.90, max_octave=0.03) -> bool`
  - `f0_group_rule(failure_ratio, y, lo=0.30, hi=0.70) -> tuple[float, bool]`：回傳 `(auc, keep_pitch_measures)`
  - `segment_gate(raw_td: Corpus, cfg, rules, words_of: Callable[[list[str]], list[list[str]]]) -> list[GateResult]`：各孩子對段落取平均後比較；基準是正規化後的原始資料，變體是「先套轉換、再正規化」，做法同 `run_gate`

細節：
- Praat 用 `parselmouth.Sound(x, sr).to_pitch_ac(time_step=hop, pitch_floor=fmin, pitch_ceiling=fmax)`。
- pYIN 和 Praat 都內插到 CREPE 的幀時間上；一致和八度的判定沿用 `backends.f0_cross_check` 的比值區間。
- A11 分半：兩半各自算孩子句子的 `rms_db`（長度 ≥ 0.1 秒），再交給 `_level_rel_examiner`，醫師那邊用同一半的醫師句子。

- [ ] **Step 1: 寫失敗的測試**

```python
def test_split_half_r():
    assert split_half_r(np.arange(10.), np.arange(10.) * 2) == pytest.approx(1.0)
    assert np.isnan(split_half_r(np.array([1., np.nan, 2, 3]), np.array([1., 2, np.nan, 3])))

def test_b7_a11_halves_split_turns_alternately(fake_backends):   # embed_windows -> seeded random vectors
    h = b7_a11_halves(synthetic_subject, cfg, fake_backends)
    assert all(np.isfinite(v).all() for v in map(np.array, h.values()))

@pytest.mark.slow
def test_f0_three_way_agrees_on_a_harmonic_tone():   # 220 Hz, 5 s, 16 kHz
    c = f0_three_way(tone, 16000, [one_child_turn], cfg, real_backends)
    assert f0_session_pass(c)

def test_f0_group_rule():
    assert f0_group_rule(np.full(10, .1), np.r_[np.ones(5), np.zeros(5)]) == (0.5, True)
    assert f0_group_rule(np.r_[np.ones(5), np.zeros(5)], np.r_[np.ones(5), np.zeros(5)]) == (1.0, False)

def test_segment_gate_strip_is_a_no_op_after_normalization(synthetic_corpus):
    res = segment_gate(synthetic_corpus, cfg, rules, words_of=lambda ts: [list(t) for t in ts])
    assert all(r.shifts.get("strip_punctuation", 0.0) == pytest.approx(0.0) for r in res)
```

- [ ] **Step 2: 確認失敗**
- [ ] **Step 3: 實作 `checks.py`**
- [ ] **Step 4: 確認通過**（slow 那條要跑：`-m "slow or not slow"`）　Expected: 5 passed

---

### Task 7: 真實資料串接、報告、CLI

**Files:**
- Create: `src/asdpipe/mil/run.py`
- Modify: `src/asdpipe/cli/main.py`：新增 `mil-checks`、`mil-run` 兩個 handler；`make.cmd` 不用改
- Test: `tests/test_mil_run.py`

**Interfaces:**
- Consumes: Task 2–6 全部；`realdata.discover_cases/load_cohort`、`normalize.normalize_corpus`、`extract.extract_corpus`、`assemble.build_table`、`eval.auto_checks.SEEN_BEFORE`
- Produces:
  - `pseudonyms(repo_root: Path) -> dict[str, str]`：session → pid/uid
  - `lookup(mapping, session: str, cohort: str, index: int) -> str`：查不到時丟 `KeyError("TD case #3 has no pseudonym")`，只寫 cohort 和序號
  - `freeze_guard(repo_root: Path) -> str`：`research_plan_1030.md`、`configs/`、`src/asdpipe/mil/` 只要有未提交的修改就 raise；回傳 HEAD commit hash
  - `analyse(bags_by_len: Mapping[int, Bags], cfg, rules, log) -> dict`
  - `write_mil_report(path, results) -> Path`：只放編號和數字，不放任何原文
  - `write_attention_excerpts(path, ...) -> Path`：每位孩子權重最高的段落，含時間和原文，寫到 `outputs/`，不在對話中顯示
  - `cmd_mil_checks(cfg, rules)`：只讀 TD → 寫出 `outputs/mil_checks.md` 和 `.json`
  - `cmd_mil_run(cfg, rules)`：先過 `freeze_guard` → 寫出 `outputs/mil_results_<標籤>.json`、`mil_report_<標籤>.md`、`mil_attention_<標籤>.md` 和 `mil_attention_transcript_only_<標籤>.md`；標籤是初步跑的 `prelim_asd10`（ASD 10 位）或正式跑的 `final`（20 位），所以正式跑蓋不掉初步跑的檔案，同名檔案已存在就拒絕（`output_files`）

`analyse` 跑的版本：

| 版本 | 段落量測 | 孩子量測 | 打亂檢定 |
|---|---|---|---|
| primary | 通過檢查的段落量測 | 通過檢查的孩子量測 | 有，模型和 LR 都做 |
| transcript_only | 同上 | 無 | 有，模型和 LR 都做 |
| seg_len_3、seg_len_7 | 同 primary，換段落長度 | 同 primary | 無 |
| no_E5、E5_only | 拿掉 E5；只留 E5 | primary；無 | 無 |
| drop_E、drop_C、drop_B | 依 `STREAMS` 拿掉一整流 | 同左 | 無 |
| unexposed_only | 只用 TD 加未曝光 ASD，且需 ≥ `unexposed_only_min_asd` 位 | 同 primary | 無 |

`analyse` 最後用全部孩子、5 個種子訓練最終模型，報告：
- 閘門平均值。
- 穩定度：25 次折內擬合（先對種子平均）裡，閘門 > 0.5 的比例。
- 每位孩子注意力最高的段落。

`cmd_mil_run` 的處理順序：
1. 讀兩組資料：TD 標籤設為 0；ASD 只取有逐字稿的，標籤設為 1。
2. 正規化。
3. 兩組各自跑 `extract_corpus`（有快取），取出 4 個孩子量測和 `c_f0_tracking_failure_ratio`。
4. 每場做音高三法檢查，沒通過的場次 A2、B2 設成 NaN。
5. 套用 `mil_checks.json` 的 TD 結果：一致性沒過、或慣例關卡沒過的量測整欄拿掉。
6. 套 `f0_group_rule`。
7. 用 CKIP 斷孩子的句子，建立長度 3、5、7 的 bags。
8. 跑 `analyse`，寫出報告。
9. 結果標示「初步（ASD n）」或「正式」：ASD 等於 20 位才算正式。

- [ ] **Step 1: 寫失敗的測試**

```python
def test_analyse_on_synthetic(small_cfg):          # repeats=2, seeds=2, n_permutations=9, epochs=100
    res = analyse({3: b3, 5: b5, 7: b7}, small_cfg, rules, log=lambda m: None)
    assert {"primary", "transcript_only", "seg_len_3", "E5_only", "drop_B"} <= set(res["variants"])
    assert res["variants"]["unexposed_only"]["status"].startswith("not run")
    assert 0.0 <= res["variants"]["primary"]["mil"]["auc"] <= 1.0

def test_pseudonym_error_never_names_the_folder(tmp_path):
    with pytest.raises(KeyError) as e: lookup(pseudonyms(tmp_path), "某某某", cohort="TD", index=3)
    assert "某某某" not in str(e.value)

def test_report_holds_ids_and_numbers_only(tmp_path):
    p = write_mil_report(tmp_path / "r.md", res_with_ids)
    assert "某某某" not in p.read_text(encoding="utf-8")

def test_freeze_guard_refuses_dirty_tree(tmp_git_repo):
    (tmp_git_repo / "configs/x.yaml").write_text("a: 2"); pytest.raises(RuntimeError, freeze_guard, tmp_git_repo)
```

- [ ] **Step 2: 確認失敗**
- [ ] **Step 3: 實作 `run.py` 和兩個 CLI handler**
- [ ] **Step 4: 確認通過，並跑全部測試**　Run: `.\make.cmd test`　Expected: 全部 passed

---

### Task 8: 跑 TD 自動檢查（不碰 ASD）

- [ ] **Step 1:** Run（只有一次完成通知，不輪詢）：`$env:ASDPIPE_GPU_MEM_FRACTION='0.15'; .\make.cmd mil-checks`　Expected: 寫出 `outputs/mil_checks.md`，內含 B7、A11 的 r；19 場的音高三法通過數；6 個段落量測在 3 種轉換下的移動量。
- [ ] **Step 2:** 把 `outputs/mil_checks.md` 末段建議的 `outcome` 區塊（等於 `mil_checks.json` 的 `proposed_outcome`）抄進 `decision_rules.yaml: mil_analysis.outcome`，共四個鍵 `{excluded, source, date, code_hash}`：`excluded` 是被拿掉的量測（十個都通過就寫 `[]`），`source` 是 `outputs/mil_checks.md`，`date`（`"YYYY-MM-DD"`）和 `code_hash`（64 位十六進位）都要加引號，不加引號 YAML 會把日期讀成日期物件，`mil-run` 會拒絕。一致性 ≥ 0.70；音高 TD 通過場次 ≥ 一半；慣例關卡 < 0.20 SD。一律照規則判定，不調門檻。`code_hash` 是檢查當時程式和設定的雜湊值，之後程式或設定有任何改動，`mil-run` 都會拒絕，要重跑 `mil-checks`。
- [ ] **Step 3:** 用白話向使用者回報檢查結果。如果有量測被拿掉，照計畫「那一格留空」，不找替代品。

---

### Task 9: 凍結 commit → 正式跑一次 → 報告

- [ ] **Step 1: 檢查要 commit 的東西**　Run: `git -C D:\PycharmProjects\paper status --short asd_classification`，只 stage 以下路徑：
  - `asd_classification/research_plan_1030.md`
  - `asd_classification/DEVIATIONS.md`
  - `asd_classification/configs/`
  - `asd_classification/src/`
  - `asd_classification/tests/`
  - `asd_classification/docs/superpowers/plans/2026-11-01-segment-mil.md`
  - `asd_classification/docs/superpowers/calibration/`（含 `mil_calibrate.log`；`*.log` 被 gitignore，要用 `git add -f` 才加得進去）
  - repo 根目錄的 `.gitignore`（只多了 `_private_id_map*.csv` 這條規則，R38）

  接著跑 `git diff --cached --stat`，確認沒有 docx、`outputs/`、`cache/`，而且 log 只有 `mil_calibrate.log`。
- [ ] **Step 2: Commit**（使用者核准本計畫，即代表同意這一次 commit）

```bash
git commit -m "Freeze research plan 1030 and the segment-MIL pipeline before the first labelled run (D17)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 3: 跑**　Run（背景執行，完成時只通知一次）：`$env:ASDPIPE_GPU_MEM_FRACTION='1.0'; .\make.cmd mil-run`（使用者 2026-11-02 同意可用 100%）　Expected: 寫出 `outputs/mil_report_prelim_asd10.md`（正式跑是 `mil_report_final.md`；數字在 `mil_results_prelim_asd10.json`，原文節錄在 `mil_attention_prelim_asd10.md` 和 `mil_attention_transcript_only_prelim_asd10.md`），內含 primary／transcript_only 的 AUC、95% 區間、打亂 p，LR 的同一組數字，以及各附加版本的 AUC。
- [ ] **Step 4: 寫 `progress_report_11XX.md`**：比較表格加上「相對 LR 的差」，文字精簡、白話；只放編號，不放原文。必須附上計畫第五節第 4 點的揭露事項，以及 D16。
- [ ] **Step 5: 在 D17 補一行**：跑的日期和 commit hash。
