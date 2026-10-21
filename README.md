# Navi Agent PoC — B disclosure + slow ReAct

Current approved path: all-domain JEV B disclosure → DeepSeek ReAct → shared
mock runtime. The reference navigation repository stays read-only.

## 这是什么 / What this repository is

上游依赖仓库：车载语音 Agent 的 B 披露（每个域一次 Noul 三问披露）+ DeepSeek ReAct
慢执行 + 共享 mock 运行时。它同时被 `poc-streaming-eot` 当作**只读上游**使用（不复制文件、
不用 submodule），后者通过 `POC_UPSTREAM` 指向本仓库根目录。

### 分支 / Branches

| 分支 | 内容 | 谁用 |
|---|---|---|
| `main` | 早期基线快照（B-only 慢执行服务 + 仅有的 21 个文件） | 独立跑 `app.py` 服务 |
| `poc-streaming-eot` | 在基线上继续演进的分支：`eval/`（KEV 两阶段选工具、EOT 判停打分、推流式 ASR、前缀回放评测）、扩写的 `model_clients.py`、`turn_session.py` 等 | **`poc-streaming-eot` 仓库必须用这个分支** |

`poc-streaming-eot` 仓库 import 的具体文件：`model_clients.py`、`registry.py`、
`decisions.py`，以及 `eval/fixtures/core-pipeline-corpus.jsonl`（前缀回放语料）。
`main` 的 `model_clients.py` 没有 `MULTIDOMAIN_CONTROLS`，用 `main` 会在 import 阶段失败。

```sh
git clone https://github.com/songyuyao110-tech/navi-dual-system.git
cd navi-dual-system
git checkout poc-streaming-eot          # PoC 依赖这个分支，不是 main
```

运行时产物 `eval/evidence/`、`eval/results/` 是本地评测输出，不入库。

### 安装 / Install

需要 Python 3.10+（在 3.13 上验证）。

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

### 凭据 / Credentials（自备，仓库里没有任何密钥）

只从**进程环境变量**读取，本仓库不含 `.env`、密钥或证书：

- `TYPESAFE_API_KEY` — JEV/TypeSafe（B 披露）凭据，`jev_smoke.py` 与披露链路使用。
- `LITELLM_BASE_URL` / `LITELLM_API_KEY` / `SLOW_MODEL` — LiteLLM 慢执行凭据与模型 ID。
  三个都必须设，缺一个会显式失败，不会回落到默认模型。

没有这些凭据时：单元测试（`python -m unittest discover -s tests`，用 mock transport）可以跑，
真实 JEV 披露 / LiteLLM 决策跑不了。详见下面各节。

## Status

The source now uses **B-only slow execution**. No A task/target/route decision,
fast-model call, or low-confidence task-review call occurs in the HTTP entry.
Legacy A/fast helpers and their isolated tests remain for reference, but are not
part of this active path. Fast-protocol optimization is paused.

- Each registered domain makes one concurrent B request with one Noul question
  per tool. Disclosure is the union at threshold 0.5: 10 domains × 20 tools means
  **10 requests**, not 11 or 200. No embedding, group filter, or A domain filter.
- New tasks go straight to LiteLLM `deepseek-v4.1-flash` ReAct (at most 10 rounds
  per execution invocation); `ask_user`, `request_tools`, and `finish` remain.
- Only one active task is allowed. While disclosure/execution is processing,
  another chat returns HTTP 409 without recording its message or creating a job.
- Once the task is waiting for input, the next chat is its answer. Refill keeps
  the task ID, original registry snapshot, disclosed tools, and completed facts;
  it skips initial B and resumes ReAct. Additional disclosure reruns every domain
  B against that same original snapshot. Successful non-idempotent actions are
  not repeated.
- Cancellation, revision, switching topics during a clarification, and multiple
  active tasks are not supported by this simplified interaction. If old state
  contains multiple pending tasks, input is rejected rather than guessing.
- Registry reload still validates all definitions before atomic publication.
  New tasks use the new snapshot; existing tasks and additional B keep the old one.
- Navigation remains **mock only**. Android/Omni voice, the `/agent` vehicle
  protocol, real vehicle execution, and final V1-V12 evaluation are not connected.

**Source versus deployment:** this B-only change has not restarted the existing
8260 process. An existing service may still run the earlier A/B dual-chain code;
a newly started B-only process reports `execution_policy: b_only_slow` in health.
Do not restart without approval: it loses the in-memory ledger, conversation,
and mock state. This is a local single-user PoC, not a production service.

The LiteLLM migration and real DeepSeek probes on 2026-10-10 are historical
connectivity evidence, not live verification of this new B-only path. Configuration
is already confirmed below; no Claude fallback, new credentials, or vendor switch
is needed. Current change verification uses isolated mocked provider transports.

## Setup

Use Python 3.10 or newer. The macOS system Python 3.9 is not supported.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run

The current B-only slow path does **not** need Ollama or a fast-model startup.
Leave any existing Ollama service unchanged; do not start or stop it for this change.

After confirming no existing PoC service owns the port, and approving a restart
if needed:

```sh
.venv/bin/python app.py --env-file /path/to/private.env
```

Open <http://127.0.0.1:8260>. Enter an utterance and optional JSON context, run
the mock task or only B disclosure, then inspect decisions, task facts, and runtime calls.
For a synthetic navigation-home test, context can contain
`{"home_address_configured": true, "synthetic": true}`.

The env-file loader only reads TypeSafe variables; it does not execute the file.
Slow execution requires all three variables already in the process environment:

- `LITELLM_BASE_URL`: the confirmed LiteLLM API root or `/v1` base URL.
- `LITELLM_API_KEY`: credentials explicitly bound to that LiteLLM endpoint.
- `SLOW_MODEL`: the exact deployed DeepSeek model ID from that account; no
  guessed model name or default Claude model is used.

The TypeSafe env-file loader does not load these variables. Set them in your
private launch configuration before starting the service. Credentials stay
outside this project; do not reuse Bedrock or direct DeepSeek credentials at a
different endpoint. Missing settings fail explicitly without a model request.
TLS certificate verification remains enabled; non-loopback HTTP is rejected.

Optional active process variables: `PORT` (8260), `TOOLS_FILE` (tools.yaml).
The required slow settings are above. Legacy `FAST_MODEL` / `FAST_MODEL_URL`
only apply to isolated fast-model experiments, not the HTTP execution path.

Task text/context and registered tool descriptions are sent to the configured
JEV endpoint; slow execution also sends task briefs and disclosed schemas to
the configured LiteLLM endpoint. Do not enter secrets or sensitive test data.

LiteLLM requests use `/v1/chat/completions`, function tools, and tool-result
messages. Responses are normalized to the existing controller contract.
DeepSeek reasoning metadata is retained for subsequent tool-call rounds when
the provider supplies it. Invalid JSON, malformed calls, truncation, and
transport errors remain explicit failures.

### Confirmed DSH configuration

The 2026-07-23 email **Your LiteLLM API Key** binds the key to
`https://litellm.swfcn.cn.bg.corpintra.net`. The DSH provider
`swf-litellm` in `~/.dsh/profiles/desktop/cordis.patch.yml` and Desktop's
`~/Library/Application Support/dsh-desktop/harness/profiles/web/cordis.patch.yml`
uses:

- `api: openai-completions`, not the Responses or Anthropic API.
- `baseURL: https://litellm.swfcn.cn.bg.corpintra.net/v1`.
- `apiKeyEnv: SWF_LITELLM_API_KEY`; the value is stored under
  `refs.SWF_LITELLM_API_KEY` in `~/.dsh/.credentials.yaml`.
- Confirmed account model ID: `deepseek-v4.1-flash`.

The email key and DSH reference match. Do not paste the key into this project,
modify DSH's settings, or log in to the LiteLLM platform UI. The email grants API
access only. To start the PoC with that read-only reference, after approving the
loss of the existing in-memory ledger and mock state:

```sh
.venv/bin/python - --env-file /path/to/private-typesafe.env <<'PY'
import os
from pathlib import Path
import runpy
import yaml

try:
    credentials = yaml.safe_load((Path.home() / ".dsh/.credentials.yaml").read_text())
except (OSError, yaml.YAMLError):
    raise SystemExit("DSH credential file could not be read") from None
refs = credentials.get("refs") if isinstance(credentials, dict) else None
key = refs.get("SWF_LITELLM_API_KEY") if isinstance(refs, dict) else None
if not isinstance(key, str) or not key.strip():
    raise SystemExit("DSH LiteLLM credential reference is missing")
os.environ.update({
    "LITELLM_BASE_URL": "https://litellm.swfcn.cn.bg.corpintra.net/v1",
    "LITELLM_API_KEY": key,
    "SLOW_MODEL": "deepseek-v4.1-flash",
})
runpy.run_path("app.py", run_name="__main__")
PY
```

This reads the key only into the new process and does not create a copied
credential file. It is not a hot switch of an already running Python process.
Do not start a second service on port 8260 or stop the existing service without
approval. The approved restart cleared the previous five completed records,
conversation, and mock navigation state. Health and empty-ledger/state checks
passed before new synthetic tests were submitted.

## Tool updates

Edit `tools.yaml`, then click **重新加载 tools.yaml** or POST
`/api/registry/reload`. The endpoint loads only the configured file, not a path
supplied by a request.

Domains, descriptions, schemas, and bindings are published together. Duplicate
identifiers, invalid schemas, unknown bindings, and reserved control names
reject the whole reload and preserve the previous version. Available execution
bindings are `mock` and `mock_error`; arbitrary Python handler hot replacement is
not implemented. A newly registered domain participates automatically in B.

API: `/health`, `/api/registry`, `/api/registry/reload`, `/api/decide`,
`/api/chat`, `/api/jobs/<job_id>`, `/api/tasks`, `/api/runtime`.
Chat/decision requests return HTTP 202 and a job ID; poll for the result.
Task outcomes (`done`, `waiting_input`, `failed`, `cancelled`) are distinct from
the job having finished processing.

## Real JEV probe

Set `TYPESAFE_API_KEY` in the process environment, or pass an existing private env
file. Only TypeSafe credential/model/URL assignments are read. The file is not
copied and its values are not printed.

```sh
.venv/bin/python jev_smoke.py \
  --env-file /path/to/private.env \
  --output eval/results/jev-smoke.json
```

The probe sends only a synthetic navigation request. It lists account models,
then calls one B per registered probe domain concurrently, with retries disabled.
No A is requested. The report's `request_count` counts B calls only; model discovery
is a separate request. B uses Noul questions for every supplied tool. Incomplete
answers and invalid probabilities fail explicitly.

Nothing is executed on a vehicle. A passing probe proves API connectivity and
response shape, not semantic accuracy or a latency target.

## B-only verification recorded on 2026-10-12

- All **67 tests pass** with the project Python 3.13 environment; Python compile
  checks and `git diff --check` pass. Existing legacy helper tests are retained.
- Active-path tests cover no A/fast/review calls, concurrent per-domain B union,
  10 domains × 20 tools as exactly 10 requests, same-ID clarification refill,
  original snapshots, completed-fact preservation, busy-state rejection, true
  simultaneous HTTP submissions, the question-publication window, and failure
  release (initial B, additional B, model/control failure, worker startup).
- A Flask test-client loop exercises the actual TypeSafe SDK and LiteLLM adapter
  with **mock HTTP transports**: one B → `ask_user` → same-task refill → navigation
  mock fact → `finish`. This is not a live model semantic or latency evaluation.
- No live request, existing-service restart, ledger reset, Ollama stop, or real
  vehicle operation was performed. Existing 8260 process deployment is unverified.
- The original 53-test baseline required removing inherited `NO_PROXY/no_proxy`
  only in its test subprocess because `httpx2` rejected an IPv6 proxy-exclusion
  entry. No proxy/TLS/service setting was changed. Current isolated tests pass
  in the ordinary test environment.

## Historical verification recorded on 2026-10-10

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
```

53 tests pass after the LiteLLM migration, including two-round tool execution,
additional disclosure, task review, error-result conversion, reasoning metadata,
and refusal to reuse Bedrock credentials. Python compile checks and editor
diagnostics pass. Earlier dependency and debug-page JavaScript syntax checks
passed; browser interaction/rendering has not been verified.

Live calls used synthetic inputs only:

| Probe | Observed result |
|---|---|
| JEV minimal A + one B | `jev-1.13.0`, about 393 ms; discovered aliases `jev-latest`, `jev-preview` |
| Actual navigation registry B | One B with all 14 tool questions; A + B about 418 ms |
| Initial complete fast chain | One Qwen3-4B plan and one mock home action; backend about 7.06 s |
| Latest consecutive home/company commands | Two separate completed tasks; A + B about 469/439 ms; backend about 2.16/4.18 s |
| Historical slow execution before migration | `claude-sonnet-4-6`, two rounds, one shared-runtime mock status call; about 16.95 s |
| New LiteLLM slow adapter | Mocked HTTP/protocol tests pass; authenticated `/v1/models` returns 200 and lists `deepseek-v4.1-flash` |
| Actual DeepSeek slow execution | `deepseek-v4.1-flash`, two rounds, one shared-runtime mock status call then `finish`; about 5.30 s in an isolated ledger, without live JEV |
| Restarted service, live JEV and DeepSeek | Explicit new conditional navigation-check task: slow route, two `deepseek-v4.1-flash` rounds, one mock status call, task `done`; backend about 4.86 s |
| Unchanged local fast client after restart | Real `qwen3:4b` one-shot home plan and isolated mock execution pass; not a live HTTP fast-route test |
| Registry reload HTTP endpoint | Successful load of the active registry; changed/invalid registry and in-flight version isolation validated by tests |

The 10-domain x 20-tool, 11-request concurrency test uses an HTTP mock barrier;
real 10-domain service quota and latency are **not** verified. Latencies above
are individual backend samples, not P50/P95, voice latency, or a demonstrated
0.5-second fast-chain target. That target has not been reached.

Generated synthetic probe results are under ignored `eval/results/`. The
original navigation repository was not modified.

The initial live conditional-cancellation input returned `ModelError` before
task creation; its original diagnostic is retained in
`deepseek-v41-live-service-smoke.json`. An isolated real task review returned
a clarification. The explicit new-task test passed
(`deepseek-v41-live-service-check.json`), and replaying the initial input also
passed, with no task review requested on that replay and backend about 4.68 s
(`deepseek-v41-live-review-replay.json`). The initial failure's root cause is not
confirmed; these few probes do not establish reliable low-confidence review or
overall semantic accuracy.

A live home request also completed through DeepSeek because JEV selected the
slow route; `fast-after-deepseek-switch.json` must not be counted as a fast-chain
pass. A separate real local fast-client test passed
(`qwen4b-after-switch-local.json`). No fast prompt, routing threshold, Ollama
process, or DSH settings were changed during the switch. The restarted PoC is
the attached `poc-server-deepseek` process, not a detached permanent service.
