# Lab 21 — Evaluation Report

**Học viên**: Phan Văn Hiếu — 2A202600732
**Ngày nộp**: 2026-06-25
**Submission option**: A (lightweight ZIP)
**Notebook**: `notebooks/Lab21_LoRA_Finetuning_T4.ipynb`

---

## 1. Setup

| Mục | Giá trị |
|-----|---------|
| **Base model** | `unsloth/Qwen2.5-3B-bnb-4bit` (Qwen2.5-3B, pre-quantized 4-bit NF4 / QLoRA) |
| **Dataset** | `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` — 200 samples (180 train + 20 eval) |
| **Format** | Alpaca (`instruction_vi` / `input_vi` / `output_vi`), 90/10 split, seed = 42 |
| **max_seq_length** | 1024 (token analysis: p50 = 227, **p95 = 562**, p99 = 704, max = 738 → làm tròn lên nhưng cap T4 = 1024) |
| **GPU** | Tesla T4, 15.6 GB VRAM (CUDA 12.8, PyTorch 2.11.0, Unsloth 2026.6.9) |
| **Training config** | 3 epochs (69 steps), cosine LR, lr = 2e-4, warmup ratio = 0.10, effective batch = 8 (1 × grad_accum 8), optim = `adamw_8bit`, gradient checkpointing = unsloth, `packing=False` |
| **LoRA target** | `["q_proj", "v_proj"]` (lab spec) |
| **Training cost** | **~$0.07** total (11.4 phút train × 3 ranks @ $0.35/hr T4) |

> 3 adapter checkpoints (`r8/`, `r16/`, `r64/`) đã được lưu vào `OUTPUT_DIR` trước khi eval để tránh mất checkpoint nếu eval OOM.

---

## 2. Rank Experiment Results

Cùng dataset, cùng hyperparameters — chỉ thay đổi `rank` / `alpha` (giữ tỉ lệ `alpha/r = 2`).

| Rank | Alpha | Trainable Params | % Trained | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-------|------------------|-----------|-----------|-----------|-----------|------------|
| 8    | 16    | 1,843,200        | 0.06%     | 3.77 min  | 7.22 GB   | 1.5577    | **4.748**  |
| 16   | 32    | 3,686,400        | 0.12%     | 3.96 min  | 6.62 GB   | 1.5161    | **4.554**  |
| 64   | 128   | 14,745,600       | 0.48%     | 3.72 min  | 8.00 GB   | 1.4768    | **4.379**  |

> **Lưu ý về perplexity của base model**: phần qualitative dùng base model nhưng eval-loss/perplexity của base model *không* được tính riêng trong lần chạy này (chỉ compute cho 3 fine-tuned adapter). Để có đủ 4 con số theo rubric, cần chạy thêm một lượt `safe_evaluate()` trên base model chưa wrap LoRA — đây là điểm cần bổ sung nếu chạy lại.

**Quan sát chính**:
- **Trainable params** tăng tuyến tính theo rank (r=64 gấp ~8× r=8) đúng như công thức `params ∝ r × (d_in + d_out) × n_modules`.
- **Perplexity giảm đều** khi tăng rank: 4.748 → 4.554 → 4.379. Tức là rank lớn hơn fit dataset tốt hơn một chút.
- **Train time gần như bằng nhau** (~3.7–4.0 min) — trên T4 với q+v target nhỏ, bottleneck là I/O và overhead chứ không phải số LoRA params, nên rank lớn hơn *không* đắt thêm về thời gian.
- **Peak VRAM không đơn điệu** (7.22 → 6.62 → 8.00 GB): r=16 thấp nhất, có thể do biến động allocator / fragmentation giữa các lần reload model. r=64 cao nhất như kỳ vọng do thêm optimizer state cho nhiều params hơn. Tất cả đều nằm thoải mái trong 16 GB của T4.

---

## 3. Loss Curve Analysis

Xem `results/loss_curve.png` (r=16).

- **Eval-during-training bị tắt** (`eval_strategy="no"`) để tiết kiệm VRAM trên T4 — vì vậy chỉ có **train loss curve**, không có eval curve song song để vẽ overfitting trực tiếp.
- Train loss giảm đều và hội tụ qua 3 epochs (69 steps) — không có dấu hiệu loss bật lên hay phân kỳ.
- **Đánh giá overfitting gián tiếp**: eval loss cuối cùng (1.5161 cho r=16) vẫn ở mức hợp lý và *thấp hơn* khi tăng rank, không cho thấy overfitting nghiêm trọng. Với dataset chỉ 180 train samples × 3 epochs, rủi ro overfit có tồn tại nhưng perplexity vẫn cải thiện theo rank → chưa chạm ngưỡng memorize.
- **Cách kiểm tra tốt hơn**: bật eval mỗi epoch (trên GPU lớn hơn) để theo dõi gap train/eval loss — nếu eval loss tăng trong khi train loss giảm thì mới kết luận overfit chắc chắn.

---

## 4. Qualitative Comparison (Base vs Fine-tuned r=16)

5 prompt tiếng Việt, so sánh base model gốc vs adapter r=16. (Đầy đủ trong `results/qualitative_comparison.csv`.)

### Example 1 — "Giải thích khái niệm machine learning cho người mới bắt đầu."
- **Base**: "Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu…"
- **Fine-tuned**: "Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp…"
- **Nhận xét**: ~Tương đương. Cả hai đều đúng; FT diễn đạt mạch lạc, gần văn phong dataset hơn một chút.

### Example 2 — "Viết đoạn code Python tính số Fibonacci thứ n."
- **Base**: định nghĩa `fibonacci(n)` xử lý `n <= 0` trả về string.
- **Fine-tuned**: `raise ValueError("Input phải là một số nguyên dương.")` + xử lý `n == 0` rõ ràng.
- **Nhận xét**: **Cải thiện**. FT dùng error-handling chuẩn (raise exception) thay vì trả string — code production-quality hơn.

### Example 3 — "Liệt kê 5 nguyên tắc thiết kế UI/UX."
- **Base**: liệt kê kiểu mô tả dài dòng, lặp từ ("thân thiện… thân thiện…").
- **Fine-tuned**: bullet list ngắn gọn — "1. Chuyển đổi… 2. Thích ứng… 3. Đơn giản…".
- **Nhận xét**: **Cải thiện về format**. FT bám yêu cầu "liệt kê" tốt hơn, output structured và súc tích hơn.

### Example 4 — "Tóm tắt sự khác biệt giữa LoRA và QLoRA."
- **Base**: "LoRA (Low-Rank Adaptation)…" — *đúng* tên viết tắt.
- **Fine-tuned**: "LoRA (Layer-wise Adaptive Regularization Optimization)…" — **sai** tên viết tắt.
- **Nhận xét**: **Degraded (case loss)**. Đây là ví dụ cho thấy fine-tune trên dataset general-domain *không* thêm kiến thức kỹ thuật mới — thậm chí có thể làm nhiễu kiến thức factual có sẵn của base. Đúng với quy tắc "fine-tune cho style/format, RAG cho knowledge".

### Example 5 — "Phân biệt prompt engineering, RAG, và fine-tuning."
- **Base**: giải thích 3 khái niệm, tương đối tổng quát.
- **Fine-tuned**: cấu trúc rõ hơn ("…là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI…"), giải thích từng kỹ thuật.
- **Nhận xét**: ~Tương đương / hơi tốt hơn về tổ chức ý.

**Tổng kết qualitative**: 3/5 cải thiện (chủ yếu về **format & văn phong**), 1/5 tương đương, **1/5 degraded** (factual về tên viết tắt LoRA). Không cherry-pick — case loss được giữ lại để phản ánh trung thực giới hạn của SFT general-domain.

---

## 5. Conclusion về Rank Trade-off

Trên dataset Vietnamese-Alpaca 180 samples này, **r=16 cho ROI tốt nhất**. Lý do: khi đi từ r=8 → r=16, perplexity giảm 4.748 → 4.554 (~4.1%) với chi phí gấp đôi params nhưng VRAM và thời gian *không* tăng (thực tế r=16 còn dùng ít VRAM nhất, 6.62 GB). Đi tiếp r=16 → r=64, perplexity chỉ giảm thêm 4.554 → 4.379 (~3.8%) nhưng phải trả gấp 4× params (3.7M → 14.7M) và VRAM cao nhất (8.00 GB). Đây chính là vùng **diminishing returns**: mỗi đơn vị params bổ sung mua được ít cải thiện perplexity hơn — gain tuyệt đối từ r=16→r=64 (-0.175 ppl) còn nhỏ hơn gain r=8→r=16 (-0.194 ppl) dù tốn nhiều params hơn nhiều. Với một dataset nhỏ và target chỉ q+v, capacity của r=16 đã đủ để học style/format mục tiêu; rank cao hơn chủ yếu thêm khả năng *overfit* hơn là khái quát hóa.

**Khi nào tăng rank hết tác dụng**: ngay sau r=16 trong setup này — đường cong perplexity-vs-rank đã phẳng dần, và lợi ích r=64 không bù đắp được chi phí VRAM + nguy cơ overfit trên 180 mẫu.

**Recommendation cho production**: chọn **r=16**. Nó nằm ở "sweet spot" — perplexity gần bằng r=64 (4.55 vs 4.38, chênh ~3.8%) nhưng adapter nhỏ hơn 4×, nhẹ để serve multi-tenant và ít rủi ro overfit. Chỉ cân nhắc r=64 nếu dataset lớn hơn nhiều (10k+ samples) hoặc task phức tạp cần thêm capacity; còn r=8 phù hợp khi cần adapter siêu nhẹ và chấp nhận chất lượng thấp hơn một chút.

---

## 6. What I Learned

- **Rank ≠ thời gian train trên adapter nhỏ**: với target q+v trên T4, r=8/16/64 train gần như bằng nhau (~3.7–4.0 min) vì overhead (load model, tokenize, I/O) lấn át chi phí tính LoRA params — bài học là đừng giả định "rank cao = chậm" mà phải đo.
- **Fine-tune cải thiện format/style chứ không phải knowledge**: ví dụ FT dịch sai "LoRA" thành "Layer-wise Adaptive Regularization Optimization" chứng minh trực tiếp quy tắc vàng — muốn đúng factual thì cần RAG, SFT general-domain có thể còn làm nhiễu kiến thức base.
- **Diminishing returns là thật và đo được**: perplexity-vs-rank phẳng dần sau r=16, giúp mình hiểu vì sao r=16 là "standard choice" trong thực hành thay vì cứ chọn rank cao nhất.

---

## Phụ lục — Files đính kèm (Option A)

```
results/
├── rank_experiment_summary.csv     # metrics đầy đủ 3 ranks (verify được)
├── qualitative_comparison.csv      # 5 before/after examples
└── loss_curve.png                  # train loss r=16
adapters/
└── r16/                            # best-rank adapter (adapter_model.safetensors + adapter_config.json)
```
