---
name: ais-bench-eval
description: Run dataset accuracy benchmarks with the ais_bench (AISBench) CLI for ceval, MMLU, CMMLU, GSM8K, and other datasets. Use when the user requests testing, evaluating, or benchmarking a dataset or model accuracy, or when troubleshooting evaluation results such as a zero score or a failed run. Provides a complete from-scratch workflow covering installation from the repository, dataset download, configuration, execution, and result interpretation.
---

# ais_bench accuracy evaluation

`ais_bench` (AISBench) submits a dataset (ceval, MMLU, CMMLU, GSM8K, and others) to an OpenAI-compatible inference endpoint and computes the accuracy of the model under test. This document describes a complete from-scratch workflow and contains no machine-specific assumptions.

---

## 1. Prerequisites

- **Python 3.10, 3.11, or 3.12.** Versions 3.9 and earlier, and 3.13 and later, are not supported. A Conda environment is recommended to avoid dependency conflicts.
- System utilities required by the dataset download script: `git`, `wget`, and `unzip`.
- A running **OpenAI-compatible inference endpoint** that exposes `/v1/models` and `/v1/chat/completions` for the model under evaluation (for example, a `vllm serve` process). ais_bench acts only as the client; it does not serve the model.

## 2. Installation (from the repository)

```shell
git clone https://github.com/AISBench/benchmark.git
cd benchmark

conda create --name ais_bench python=3.10 -y
conda activate ais_bench

pip3 install -e ./ --use-pep517            # core dependencies (requirements/runtime.txt)
pip3 install -r requirements/api.txt       # required for the vllm_api served-model configuration
pip3 install -r requirements/extra.txt     # required for the vllm_api served-model configuration
# pip3 install -r requirements/hf_vl_dependency.txt   # only for HuggingFace or vision-language offline models
```

Alternative (no clone required): `pip3 install ais_bench_benchmark` for the basic distribution, or `pip3 install "ais_bench_benchmark[full]"` for the full distribution. The source checkout above is recommended because it allows the configuration files to be edited in place.

Verify the installation by running `ais_bench -h`, which prints the CLI help. All commands in the remainder of this document are intended to be run from the repository root (`benchmark/`).

## 3. Download datasets

```shell
bash download_datasets.sh
```

The script downloads 19 text benchmarks into `ais_bench/datasets/` from opencompass OSS, ModelScope, Google, and GitHub: MMLU, CMMLU, C-Eval, GSM8K, MATH, AIME 2024/2025, BBH, GPQA, HellaSwag, WinoGrande, PIQA, ARC, HumanEval, MBPP, AGIEval, TriviaQA, DROP, and MMLU-Pro. The `synthetic` dataset is built in and requires no download.

- Where an archive name differs from the expected directory name, the script creates lowercase symlinks: `BBH→bbh`, `ARC→arc`, `drop_simple_eval→drop`, `physicaliqa-train-dev→piqa`, `ceval-exam→ceval`, and `aime→aime2024`.
- Verify that a dataset is ready:
  ```bash
  ls ais_bench/datasets/mmlu/test/ | wc -l      # approximately 57 subjects
  ls ais_bench/datasets/cmmlu/test/ | wc -l     # 67
  wc -l ais_bench/datasets/gsm8k/test.jsonl     # 1319
  ```
- To obtain a single dataset, copy the corresponding `wget … && unzip …` block from `download_datasets.sh`.

## 4. Configuration

| Purpose | Path (repository-relative) |
|---|---|
| Model configuration | `ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py` |
| Dataset configurations | `ais_bench/benchmark/configs/datasets/<dataset>/*.py` |
| Dataset data | `ais_bench/datasets/<dataset>/` |

### Model configuration fields

| Field | Description |
|---|---|
| `model` | The model under evaluation, specified as a local path or a HuggingFace identifier (for example, `meta-llama/Llama-3.1-8B-Instruct`). This value must match the `--served-model-name` reported by the endpoint. |
| `host_ip` / `host_port` | The address and port of the inference endpoint. These values must match the endpoint exactly; a mismatch is the most common cause of every subject failing. |
| `batch_size` | Request concurrency. A value of 1 runs requests serially; for API-based models this field controls the number of concurrent requests. |
| `generation_kwargs` | Sampling parameters, including `temperature`, `top_p`, and `chat_template_kwargs(enable_thinking=False)`. |
| `max_out_len` | The maximum number of tokens generated per request. Increase this value for chain-of-thought datasets such as GSM8K and MATH. |
| `retry` | The number of retries per request on transient errors. |

### Dataset configuration structure

Each dataset configuration file (`configs/datasets/<dataset>/*.py`) defines the following components: a **prompt template** that includes an `</E>` token marking the in-context examples used for few-shot prompting; a **retriever** (for example, `FixKRetriever`, which selects the first K development-set examples as few-shot demonstrations); an **inferencer** (`GenInferencer` for generation); an **evaluator** (`AccEvaluator`); the **answer postprocessor**, which extracts the answer letter or number from the raw model output; and the dataset **path together with the train and test splits**.

Common configurations, passed to `--datasets` by filename:
- ceval: `ceval_gen_5_shot_str.py` (5-shot, validation split, approximately 1.3k questions)
- MMLU: `mmlu_gen_5_shot_chat_prompt.py` (5-shot, test split, approximately 14k questions)
- CMMLU: `cmmlu_gen_5_shot_cot_chat_prompt.py` (5-shot, test split, approximately 11.6k questions; the filename contains "cot" although the prompt requests a direct answer)
- GSM8K: `gsm8k_gen_4_shot_cot_chat_prompt.py` (4-shot, chain-of-thought), `gsm8k_gen_0_shot_cot_chat_prompt.py`, and `gsm8k_gen_0_shot_noncot_chat_prompt.py`

> The `--models` and `--datasets` arguments are matched **by filename** recursively under the `configs/` directory; full paths are not accepted. A configuration file placed outside `configs/` will not be found and raises `FileMatchError`. Place custom configurations under `configs/datasets/<dataset>/`.

### Confirm that the endpoint can serve requests

ais_bench sends chat requests to `host_ip`:`host_port`. Verify both that the model is listed and that it can produce a completion; a successful `/v1/models` response alone does not confirm that `/v1/chat/completions` is functional:

```bash
PORT=<your-port>      # must match host_port in the model configuration
MODEL=<your-model-id> # must match the endpoint served-model-name
curl -sf http://localhost:$PORT/v1/models | grep -q "$MODEL" && echo "models OK"
curl -sf http://localhost:$PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}" \
  >/dev/null && echo "chat OK"
```

The `-sf` flags cause curl to exit with a nonzero status on connection errors and on HTTP 4xx and 5xx responses, so that a faulty endpoint is not reported as healthy.

## 5. Execution

```bash
ais_bench --models vllm_api_general_chat.py \
          --datasets <dataset_config>.py \
          -w outputs/<run_name> &
```

Notable flags:
- `-w <dir>` — the output root directory; each run creates a `YYYYMMDD_HHMMSS/` subdirectory.
- `-m <mode>` — `all` (default), `infer` (generation only), `eval` (score existing predictions), or `viz`.
- `-r [timestamp]` — reuse prior outputs and run only the missing jobs; omitting the value reuses the latest run.
- `--debug` — run in a single process with output printed to the console and no log redirection; intended for quick debugging only.
- `--summarizer <name>.py` — override the default results summarizer.

Run long jobs in the background with `&`. Large test splits (MMLU and CMMLU contain thousands of questions) require several hours; ceval and GSM8K are smaller.

## 6. Monitoring and result interpretation

```bash
# number of questions generated so far
find outputs/<run_name>/*/predictions -name "*.jsonl" | xargs cat 2>/dev/null | wc -l
# connection and task failures recorded in the detail logs
grep -rE "Connect call failed|warmup failed|RUNNER-TASK-001" outputs/<run_name>/*/logs/ | tail -3
# follow a single task detail log in real time
tail -f outputs/<run_name>/*/logs/infer/*/*.out
```

If stdout has been redirected to a file, `grep -oE "Monitoring tasks progress:[^]]*\]" <file>` displays the overall progress bar.

Output layout (`<ts>` denotes the timestamp subdirectory):
```text
outputs/<run_name>/<ts>/
├── predictions/<model>/<dataset>.jsonl   # per-question raw prediction and gold answer
├── results/<model>/<dataset>.json        # single-dataset evaluation result
├── summary/summary_<ts>.{md,csv,txt}     # final summary; read this file
└── logs/infer/<model>/<dataset>.out      # per-task detail log
```

The `summary_*.md` file contains per-subject rows, category aggregates (such as `mmlu-stem` and `ceval-humanities`), overall rows (`mmlu`, `ceval`, `cmmlu`), and weighted rows (`*-weighted`). The reported metric is **accuracy**, expressed as a percentage. Use the plain `ceval`, `mmlu`, or `cmmlu` row as the overall score; the `*-weighted` row accounts for differences in per-subject question counts.

## 7. Troubleshooting

**a) The endpoint cannot be reached, or every subject fails.** The log contains `RUNNER-TASK-001 task failed` together with `ClientConnectorError: Cannot connect to host localhost:<port>`, which indicates that the endpoint is unreachable or that `host_port` does not match. The `Failed to read status file … .json` warnings at startup are expected; they originate from the reuse check and can be ignored.

**b) Warmup hangs and then fails, or no predictions are produced.** The log shows `Warmup: 1/1 [600s+…]` followed by `Connect call failed`, and the summary contains only dashes. This indicates that the endpoint stopped responding during the run.

**c) All questions ran, but the accuracy is 0.00.** Inspect the raw output to distinguish an extraction mismatch from garbled output:
```bash
F=$(ls outputs/<run>/<ts>/predictions/*/<dataset>.jsonl | head -1)
python3 -c "
import json
for i,l in enumerate(open('$F')):
    if i>=3: break
    d=json.loads(l)
    print('GOLD     :', repr(str(d.get('gold'))[:50]))
    print('PRED tail:', repr(d.get('prediction','')[-120:]))
"
```
- Extraction mismatch: the model produced a valid answer that the postprocessor regular expression did not capture. Resolve this by adjusting the postprocessor or by selecting a different dataset configuration.
- Garbled output: the model emitted incorrect or corrupted tokens (for example, because of numerical instability in the endpoint). This cannot be resolved through ais_bench configuration.

**d) The summary contains only dashes.** No score was produced. Check the prediction count first: a count of zero indicates a crash (see items a and b); a positive count indicates an extraction or output problem (see item c).

## 8. Checklist

1. Install: run `git clone`, `cd benchmark`, create the Conda environment, then run `pip3 install -e ./` together with `requirements/api.txt` and `requirements/extra.txt`.
2. Run `bash download_datasets.sh` and verify that the split files exist.
3. Configure the model: set `model`, `host_ip` and `host_port`, `generation_kwargs`, and `max_out_len`.
4. Confirm the endpoint by obtaining `models OK` and `chat OK` from curl.
5. Run `ais_bench --models … --datasets … -w outputs/<name>` in the background.
6. Monitor progress and connection errors in the logs.
7. Read `summary_*.md` for the accuracy. If the accuracy is `0.00` or the summary contains only dashes, follow section 7 against the raw predictions.
