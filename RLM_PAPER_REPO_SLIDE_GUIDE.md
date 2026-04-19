# Hướng Dẫn Làm Slide: Paper + Repo RLM

## 0) Thông điệp một câu cho bài nói
Recursive Language Models (RLMs) dịch chuyển long-context reasoning từ token-space processing thuần túy sang REPL environment bên ngoài, kết hợp recursive sub-calls để mở rộng năng lực inference-time scaling trên information-dense tasks.

---

## 1) Tóm tắt điều hành 30 giây (mở bài)
- Bài toán: chất lượng của vanilla LLM giảm khi input length và task complexity tăng ("context rot").
- Ý tưởng cốt lõi: coi prompt là external state trong REPL, để model programmatically inspect, decompose và recursively sub-query.
- Kết quả chính: tăng đáng kể trên long-context benchmarks (bao gồm thiết lập 10M+ token trong paper) với median API cost cùng bậc.
- Kết quả kỹ thuật: repo cung cấp inference engine chạy được (RLM loop + LMHandler + REPL), không chỉ là ý tưởng lý thuyết.

---

## 2) Cấu trúc deck gợi ý (12 slide)

## Slide 1 - Tiêu đề + Động lực
- Tiêu đề: Recursive Language Models (RLMs): Góc nhìn Paper và System
- Phụ đề: Từ failure modes của long-context đến recursive inference-time scaling
- Hình minh họa: sơ đồ trước/sau (vanilla LM vs RLM loop)

## Slide 2 - Đặt vấn đề
- Effective context window phụ thuộc task, không chỉ phụ thuộc model.
- Task khó hơn sẽ degrade sớm hơn task dễ.
- Quản lý long-context bằng naive compaction có thể làm mất chi tiết quan trọng.

## Slide 3 - Khái niệm cốt lõi của RLM
- Prompt được offload vào environment state (biến `context`).
- Model viết code trong REPL để inspect và transform context.
- Model có thể gọi `llm_query` (single-shot) và `rlm_query` (recursive).

## Slide 4 - Khác gì so với agent phổ biến
- Không chỉ tool-use trên tài liệu ngoài; chính prompt trở thành manipulable state.
- Không chỉ decomposition tĩnh theo workflow cố định; decomposition được model quyết định tại runtime.
- Bổ sung symbolic control flow + recursion trên các prompt slices.

## Slide 5 - Benchmark và complexity scaling
- S-NIAH (xấp xỉ O(1) retrieval demand)
- OOLONG (xấp xỉ O(n) aggregation)
- OOLONG-Pairs (xấp xỉ O(n^2) pairwise aggregation)
- BrowseComp-Plus, LongBench-v2 CodeQA

## Slide 6 - Quan sát thực nghiệm chính
- RLM scale tốt hơn khi input tăng và task dày thông tin hơn.
- Chỉ riêng REPL offloading đã giúp vượt context-limit behavior.
- Recursive sub-calls là yếu tố then chốt với information-dense tasks.
- Cost: cùng bậc ở median nhưng tail variance cao do trajectory length.

## Slide 7 - Tín hiệu huấn luyện (RLM-Qwen3-8B)
- Paper báo cáo post-training dựa trên trajectory để học hành vi recursive-native.
- Mức tăng báo cáo: +28.3% trung bình so với baseline Qwen3-8B trong setup đánh giá của paper.
- Thông điệp: "RLM behavior" có thể huấn luyện được, không chỉ prompt-engineering.

## Slide 8 - Kiến trúc repo (paper -> code)
- `rlm/core/rlm.py`: orchestration loop, limits, recursion, lifecycle.
- `rlm/core/lm_handler.py`: TCP routing layer cho LM calls.
- `rlm/environments/local_repl.py`: stateful REPL execution và helper APIs.
- `rlm/utils/prompts.py`: RLM system prompt contract.
- `rlm/utils/parsing.py`: parse `repl` blocks và `FINAL`/`FINAL_VAR`.

## Slide 9 - Runtime lifecycle
- Start: spawn LMHandler + environment.
- Iterate: model response -> parse code blocks -> execute -> append outputs.
- Stop: `FINAL`/`FINAL_VAR` hoặc khi chạm limit triggers.
- Cleanup: stop handler, cleanup environment.

## Slide 10 - Điểm mạnh kỹ thuật và rủi ro
- Strengths: model-agnostic, environment-agnostic, extensible, reproducible.
- Risks: local `exec` là soft sandbox; cost/time variance theo trajectory; nhạy với prompt policy.

## Slide 11 - Kịch bản demo
- Chạy một ví dụ trong `examples/`.
- Nhấn mạnh nơi xuất hiện sub-calls trong trajectory.
- Chỉ ra điểm chốt final answer (`FINAL_VAR` path).

## Slide 12 - Kết luận chính
- RLM là một inference paradigm, không phải base architecture mới.
- Nó mở khóa long-context problem solving bằng symbolic programming + recursive LM calls.
- Repo chứng minh cách làm này triển khai được như một engine tổng quát.

---

## 3) Bảng mapping paper -> repo (dùng trực tiếp lên slide)

| Claim trong paper | Cơ chế runtime | Code anchor |
|---|---|---|
| Offload prompt vào environment | Context được nạp thành REPL variable(s) | `rlm/environments/local_repl.py` |
| Iterative reasoning với execution feedback | Completion loop + code execution theo từng iteration | `rlm/core/rlm.py` |
| Recursive sub-querying | `rlm_query` -> `_subcall` -> child RLM hoặc leaf LM | `rlm/environments/local_repl.py`, `rlm/core/rlm.py` |
| Multi-model routing cho subcalls | Chọn client theo model/depth | `rlm/core/lm_handler.py` |
| Finalization protocol tường minh | Parse `FINAL` / `FINAL_VAR` rồi return | `rlm/utils/parsing.py`, `rlm/environments/local_repl.py` |
| Operational guardrails | Timeout/budget/token/error limits | `rlm/core/rlm.py` |

---

## 4) Speaker notes gợi ý (ngắn, dễ nói)

## Cho Slide 3 (Core concept)
"Trong RLM, prompt không còn là một chuỗi khổng lồ nạp một lần vào attention. Nó được externalize thành mutable state trong REPL. Model có thể viết code để inspect, branch, aggregate và gọi đệ quy sub-models. Điều này chuyển long-context handling từ thụ động sang programmatic control chủ động."

## Cho Slide 6 (Results)
"Điểm mạnh nhất không chỉ là score cao hơn, mà là scaling behavior tốt hơn khi context dài và dày thông tin. Ablation cho thấy REPL offloading quan trọng, và recursive sub-calls còn quan trọng hơn ở các task như OOLONG-Pairs."

## Cho Slide 8 (Code architecture)
"Repo phản ánh đúng architecture của paper: một RLM loop chính, một request router (LMHandler), và một execution environment (REPL). Sự tách lớp này giúp framework portable giữa nhiều model providers và sandbox options."

## Cho Slide 10 (Risks)
"RLM mạnh nhưng không miễn phí: trajectory variance tạo long-tail cho cost/runtime, và local exec chỉ là soft sandbox. Đi production cần isolation mạnh hơn, policy guardrails và monitoring."

---

## 5) Demo script (5 phút)
1. Bắt đầu từ một long-context prompt.
2. Cho xem iteration đầu: model inspect `context` và lên kế hoạch decomposition.
3. Cho xem một lần gọi `llm_query` và một lần gọi `rlm_query`.
4. Cho xem intermediate variables trong REPL.
5. Cho xem `FINAL_VAR` trả về final answer.
6. Chốt bằng trajectory metadata và thảo luận cost.

---

## 6) Q&A prep (câu hỏi hay gặp)

## Q1: RLM có phải chỉ là ReAct + tools không?
Không. Khác biệt cốt lõi là prompt được externalize thành manipulable environment state, và recursion là first-class inference primitive.

## Q2: Vì sao không chỉ tăng context window?
Context window lớn hơn có ích, nhưng không giải quyết triệt để task-complexity scaling. RLM bổ sung algorithmic decomposition và selective access tại inference time.

## Q3: Cost có luôn thấp hơn không?
Không phải lúc nào cũng thấp hơn. Median có thể cạnh tranh hoặc thấp hơn, nhưng tail có thể cao do trajectory dài.

## Q4: Mặc định đã an toàn cho production chưa?
Chưa, nếu dùng local exec. Production nên dùng isolated sandbox, policy constraints và monitoring.

---

## 7) Glossary (giữ thuật ngữ chuyên ngành)
- Inference-time scaling: tăng năng lực nhờ thêm computation/workflow lúc suy luận, không chỉ nhờ pretraining lớn hơn.
- REPL: Read-Eval-Print Loop execution environment có persistent state.
- Sub-call: LM call hoặc child RLM call được ủy quyền cho một subproblem.
- Trajectory: chuỗi đầy đủ các bước model outputs, code executions và subcalls.
- Compaction: nén ngữ cảnh bằng summarization để ở trong context window.
- Information-dense task: bài toán cần truy cập/tổng hợp rộng trên nhiều phần của ngữ cảnh.

---

## 8) Tài liệu tham chiếu nên trích trong deck
- Paper PDF: `2512.24601v2.pdf`
- Repo overview: `rlm/README.md`
- Architecture doc: `rlm/docs/architecture.md`
- Core runtime: `rlm/core/rlm.py`
- Handler: `rlm/core/lm_handler.py`
- Local environment: `rlm/environments/local_repl.py`
