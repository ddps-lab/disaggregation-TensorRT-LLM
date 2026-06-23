# LEARNING_NOTES — 공부 노트 (개념 중심)

> **역할**: 이 실험을 하면서 부딪힌 개념을 *이해 가능하게* 정리하는 학습 노트.
> `SETUP_LOG.md`(무엇을 했나, 시간순)·`CLAUDE.md`(결정/규칙)와 **역할 분리** — 여기는 "그게 무슨 뜻이고 왜 그런가".
> 살아있는 문서: 공부하며 계속 덧붙일 것. (아래는 2026-06-10 fork 셋업에서 나온 개념 씨앗)

---

## A. Git / GitHub 워크플로우

### Fork vs Clone vs Branch vs Tag (헷갈리기 쉬움)
- **Fork**: 남의 GitHub repo를 *서버 측에서* 내 계정/org로 통째 복사. 원본 수정권이 없을 때 내 복사본을 만드는 것. (예: NVIDIA/TensorRT-LLM → ddps-lab/disaggregation-TensorRT-LLM)
  - ⚠️ **fork엔 "특정 버전만 가져오기" 기능이 없다.** 항상 전체를 복사한 뒤, 그 안에서 원하는 버전 지점을 골라낸다.
- **Clone**: GitHub(클라우드)에 있는 repo를 *내 컴퓨터로* 내려받기. 편집은 clone한 로컬에서 함.
- **Branch(브랜치)**: 커밋을 쌓으면 *따라 움직이는* 포인터. 작업 줄기. (예: `disagg-exp/trtllm-v1.2.1`)
- **Tag(태그)**: 특정 커밋에 *고정으로* 붙인 이름표. 보통 릴리스 시점 표시. 움직이지 않음. (예: `v1.2.1`)
- **관계**: "특정 버전(태그)에서 새 작업 줄기(브랜치)를 딴다" = `git checkout -b <새브랜치> <태그>`. 그래서 우리는 `v1.2.1` 태그 위에 `disagg-exp/trtllm-v1.2.1` 브랜치를 만들었다.

### Submodule (서브모듈)
- repo 안에 *다른 git repo*를 특정 커밋에 고정해 끼워 넣는 것. 보통 빌드 의존 라이브러리(예: cutlass).
- **로컬 빌드 안 하면 불필요** → `--no-recurse-submodules`로 빼서 용량/시간 절약. 우리는 컨테이너로 실행하므로 뺐다.

### git-lfs (Large File Storage)
- Git은 원래 텍스트(코드)용. 큰 바이너리(이미지·가중치)를 직접 담으면 repo가 무거워짐.
- LFS = 큰 파일을 별도 저장소에 두고, repo엔 **"이 파일은 저기 있음"이라는 텍스트 포인터만** 넣는 방식.
- 포인터를 실제 파일로 바꾸려면 `git-lfs` 프로그램이 필요. **그게 안 깔려 있으면 체크아웃이 멈춘다** (우리가 겪은 exit 128).
- 해결: (a) git-lfs 설치, 또는 (b) **LFS 콘텐츠가 필요 없으면 필터를 우회**해서 포인터인 채로 두고 코드만 펼침. 우리는 (b) 선택(LFS는 docs용이라 불필요).

### 버전 핀(pin)은 **두 층위**다 — 중요
1. **소스 핀**: 어떤 코드 시점인가 → `v1.2.1` 태그/브랜치.
2. **런타임 핀**: 실제 실행 환경 → 컨테이너 이미지 `nvcr.io/nvidia/tensorrt-llm/release:1.2.1`.
- 진짜 재현성은 **런타임(컨테이너)** 이 보장: CUDA·드라이버 유저스페이스·파이썬·라이브러리·**ABI**까지 통째 고정. 소스만 같고 환경이 다르면 결과가 안 맞을 수 있다.

### 소스(clone) ≠ 컨테이너(runtime) — 헷갈리기 쉬운 핵심
- 둘 다 "1.2.1"이라 같은 것처럼 보이지만 **완전히 다른 산출물**이다.

| | 소스 코드 (git clone) | 컨테이너 이미지 |
|---|---|---|
| 정체 | 읽고·수정하고·실험파일 추가하는 **텍스트** | 미리 빌드된 **실행 환경 통조림** (OS라이브러리+CUDA+컴파일된 엔진+파이썬 의존성) |
| 위치 | 로컬 Mac + GitHub fork | 원격 GPU에서 `pull` |
| 핀 | tag `v1.2.1` | `release:1.2.1` |
| 역할 | 우리가 **건드리는** 것 | 실제로 **도는** 것 |

- **원격 GPU 실행 그림**: ① 컨테이너 `release:1.2.1` pull(엔진+환경 통째) → ② 우리 실험파일(`launch_trtllm.sh`·YAML·`sweep.py`)을 컨테이너 안으로 마운트/복사 → ③ 컨테이너 안에서 `trtllm-serve` 실행.
- **왜 둘 다 1.2.1?** 짝을 맞춰야 *소스에서 본* 예제 config·API가 *실제 도는* 엔진과 일치한다.
- **왜 소스도 clone하나?** 컨테이너 안 엔진은 읽기/수정이 불편하고, 우리 스크립트·config는 버전관리(fork)돼야 하니까. → 소스="읽기/수정/스크립트 작성"용, 컨테이너="빌드+실행"용.

> **"컨테이너로 돌린다"는 한 사실 → 두 결론**: ① 빌드 안 함 → 서브모듈 뺐다, ② 환경 통째 고정 → 재현성 보장. 두 문장은 *같은 말이 아니라* 같은 사실의 다른 결론이다.

---

## B. 왜 버전을 고정하나 (재현성)
- TRT-LLM의 **disaggregation은 아직 EXPERIMENTAL** → 버전마다 동작/버그가 바뀜. "최신"을 따라가면 어느 날 결과가 흔들린다.
- 한 버전에 고정 → "v1.2.1 + 이 컨테이너"면 누구나 같은 실험 재현 가능. **논문의 생명은 재현성.**
- 결과(숫자) 하나엔 항상 **버전·하드웨어·config**가 따라붙어야 한다. 안 그러면 "고아 숫자"(orphan number).

---

## C. 연구 방법론 개념 (research-rigor)
- **"주장 ≠ 증거"**: 릴리스 노트/블로그가 "Qwen3 됨, disagg 됨"이라 *말하는 것*과, 그 버전 *코드에 실제로 있는지*는 다름. 그래서 소스를 직접 grep해 `serve.py:646`의 disaggregated 커맨드, `modeling_qwen3.py`를 눈으로 확인했다.
- **나쁜 소식 먼저(front-load bad news)**: 빌드/실험 시작 전에 "막힐 수 있는 것"(T4 미지원, hang 버그, 무RDMA)부터 찾는다. 1시간 읽어 dealbreaker 찾는 게 1주 컴퓨트 날리는 것보다 싸다.
- **변인통제(hold everything fixed but the variable)**: 여러 config를 비교할 때 *독립변수 하나만* 다르고 나머지는 전부 고정. 캐시·dtype·배치·시드 등을 고정 안 하면 숫자가 오염돼 비교 불가.
- **스파이크 먼저(spike before scale)**: 최소 버전이 실제 환경에서 *정확히* 도는지 먼저 확인(게이트). 통과한 조합만 대량 스윕. trial #1 실패는 싸고, 풀 스윕 후 발견은 비싸다.
- **음성 결과도 결과**: "안 된다"를 근거(코드/이슈)와 함께 아는 것도 유효한 발견.

---

## D. PD Disaggregation 도메인 개념

### PD 분리가 뭔가
- LLM 추론은 두 단계: **Prefill**(프롬프트 전체를 한 번에 처리, compute-bound, KV cache 생성) + **Decode**(토큰을 하나씩 생성, memory-bandwidth-bound, KV cache 사용).
- 성격이 정반대라 한 GPU에 묶으면 서로 방해. **PD 분리** = prefill 전용 인스턴스(P)와 decode 전용 인스턴스(D)를 따로 두고, **prefill이 만든 KV cache를 D로 전송**.
- 그래서 P와 D에 **서로 다른 병렬화/자원**을 줄 수 있다 = 이 실험의 핵심.

### 병렬화 축 (TP / PP / DP)
- **TP (Tensor Parallel)**: 한 레이어의 행렬을 여러 GPU에 *쪼갬*. GPU 간 통신 잦음(빠른 인터커넥트 선호).
- **PP (Pipeline Parallel)**: 레이어들을 *단계(stage)로 나눠* 여러 GPU에 배치. 통신 적지만 파이프 버블 존재.
- **DP (Data Parallel)**: 모델을 *복제*해 데이터를 나눠 처리. **dense Qwen3-4B에선 독립 모델-병렬 축이 아님** — "DP"는 attention-DP(MoE/MLA 전용) 또는 xPyD의 복제수(D 개수)일 뿐. → 우리 실험의 모델-병렬 축 = **TP, PP 둘.**

### 대칭 / 비대칭
- **대칭**: prefill의 (TP,PP) == decode의 (TP,PP).
- **비대칭 TP**: prefill TP ≠ decode TP (KV 전송 시 head 재매핑 필요). → TRT-LLM에서 **견고**.
- **비대칭 PP**: prefill PP ≠ decode PP (레이어 분할이 단계마다 다름). → **미보증, 검증 필요.**
- (vLLM은 PD에서 PP 자체가 안 됐다 → 그래서 TRT-LLM으로 넘어옴.)

### xPyD 토폴로지
- **X개의 prefill 인스턴스 + Y개의 decode 인스턴스**를 라우터/오케스트레이터 뒤에 둔 구성. (1P1D → 1P3D → D 확장)
- decode가 throughput 병목이라 보통 **D를 늘려** 처리량을 키운다.

### KV 전송 (cache_transceiver)
- P가 만든 KV cache를 D로 실어 나르는 컴포넌트. 백엔드 = **UCX**(기본), RDMA 없으면 **TCP로 자동 폴백**.
- AWS xlarge/12xlarge는 보통 **EFA(RDMA) 없음** → inter-node는 TCP → 느림(TTFT 급증). → inter-node는 구조/correctness 비교용, 깨끗한 성능은 **intra-node(PCIe)** 중심.

### disagg config(orchestrator YAML) 스키마 읽는 법 (우리 핵심 발견)
> 이 disagg config는 정적 파일이 아니라 `launch_trtllm.sh`가 런타임 생성한다(`write_disagg_yaml()` → `disagg_<LABEL>.yaml`). 스키마 원형 = 업스트림 `examples/disaggregated/disagg_config.yaml`.
```yaml
context_servers:        # = Prefill (P)
  num_instances: 1      #   P 인스턴스 수 (xPyD의 X)
  tensor_parallel_size  #   P의 TP   ← 비대칭축
  pipeline_parallel_size#   P의 PP   ← 비대칭축
generation_servers:     # = Decode (D)
  num_instances: 3      #   ★ D 인스턴스 수 = xPyD의 Y (1P3D면 3)
  tensor_parallel_size  #   D의 TP   ← 비대칭축
  pipeline_parallel_size#   D의 PP   ← 비대칭축
```
→ P/D 각각 독립 TP·PP = **대칭/비대칭 자유**, `generation_servers.num_instances` = **D 스케일 노브**. 이 한 config로 실험 두 축이 다 표현된다.

### 알려진 함정 (왜 g5/g6/g6e, 왜 조심)
- **T4 탈락**: disagg 빌드는 **SM80 이상** 필요. T4는 SM75 → 하드 abort. → A10G(g5)/L4(g6)/L40S(g6e)만.
- **ctx-PP → gen-TP hang (#14020)**: "context에 PP + generation에 TP" 조합에서 멈추는 알려진 버그. Phase 0에서 반드시 실측.
- **무RDMA inter-node**: 위 KV 전송 항목 참고.

---

## E. 빠른 용어 사전 (측정 지표)
- **TTFT** (Time To First Token): 요청~첫 토큰. PD에선 prefill + KV전송 지연 포함. 클라이언트에서 측정(서버 메트릭은 파이프 전체를 못 봄).
- **TPOT** (Time Per Output Token): `(e2e − ttft) / (생성토큰수 − 1)`. decode 속도.
- **Throughput 2종**: service window `tok/(max(recv)−min(send))` vs arrival window `tok/(max(send)−min(send))`. 부하 높으면 둘이 벌어짐 → 둘 다 보고.
- **achieved_rate**: 성공요청수/도착윈도우. 제공 rate와 비교해 **포화(saturation)** 감지.
- **$/Mtok**: `시간당단가 / (throughput·3600/1e6)`. 다른 하드웨어/가격 비교의 정답 축(비용 정규화).

---

## F. 웜업·cold-start (벤치마크 정확성)
- **두 층위**: (1) **서버측 웜업 = TRT-LLM 자동** — `trtllm-serve` 기동 시 torch.compile + autotuner + CUDA graph 사전캡처(`py_executor.py:276-287`). 우리가 안 함. (2) **클라이언트 2-phase 웜업 = 우리 sweep** — 요청 `WARMUP_N`개 버린 뒤 measured 측정.
- **왜 클라 웜업이 필요(특히 disagg)**: disagg UCX 연결이 **첫 KV전송 때 lazy 수립**(핸드셰이크 10–200ms, `kv_cache_transceiver.py`) → 서버 웜업이 못 덮음. 스케줄러 ramp도. → 첫 요청들 버려야 cold-start 오염 제거.
- **WARMUP_N=20**: disagg UCX 흡수용 넉넉히. smoke 땐 3. **`/health` OK ≠ UCX ready** 주의.
- **autotuner 끄지 말 것**(`enable_autotuner` 기본 on; 끄면 성능 저하, 기동 시 1회만 캐시).
- 우리 실험은 **CUDA graph OFF**(균일 eager) → graph-capture cold-start 자체가 없음(변인통제).

## 더 공부할 것 (TODO — 채워가기)
- [ ] UCX가 TCP 폴백할 때 정확히 어떤 경로(`UCX_TLS`)로 가는지, shm/cuda_copy 차이
- [ ] TP head 재매핑이 비대칭 TP에서 실제로 어떻게 동작하는지 (cacheFormatter)
- [ ] PP에서 KV가 단계별로 어떻게 분할·전송되는지 (왜 비대칭 PP가 미보증인지)
- [ ] disable_overlap_scheduler가 정확히 무엇을 끄는지
- [ ] CUDA Graph가 decode throughput을 왜 크게 바꾸는지

---

## 〔심화 조사〕 병렬화 × KV전송 × per-side 측정

> **상태: TRT-LLM 적응본.** 이 문서는 vLLM 1세대 조사노트(`../../vllm-disaggregation/disagg-exp/병렬화-커넥터.md`)의
> **방법론·구조만 차용**하고, 내용은 전부 **TensorRT-LLM v1.2.1 소스 기준**으로 다시 썼다 (vLLM 복붙 아님).
> *(파일명 `병렬화-커넥터`→`병렬화-KV전송-측정`: TRT-LLM엔 vLLM 같은 "커넥터" 개념이 없고 cache_transceiver를 쓰므로 이름을 내용에 맞춤.)*
> vLLM 문서가 "어떤 *커넥터*가 어떤 병렬화를 지원하나"였다면, TRT-LLM은 커넥터 플러그인 개념이 없고
> `cache_transceiver`로 KV를 옮기며 **임의 비대칭 PP+TP를 1급 지원**한다 → 표·결론이 근본적으로 다르다.
> 대상: `ddps-lab/disaggregation-TensorRT-LLM` @ `disagg-exp/trtllm-v1.2.1`. file:line은 v1.2.1 기준(드리프트 시 재grep).

---

### 0. 핵심 용어 — cache_transceiver ≠ transport (vLLM의 커넥터≠트랜스포트에 대응)

| | 정체 | 결정 | TRT-LLM 값 |
|---|---|---|---|
| **cache_transceiver** | KV를 누구→누구로 언제 옮길지 + ctx/gen 레이아웃 매핑. `CacheTransceiverConfig` | **호환성**(어떤 KV를 보낼 수 있나) | `backend: DEFAULT\|UCX\|NIXL\|MOONCAKE\|MPI` |
| **transport** | 바이트를 옮기는 배관 | **속도/경로** | UCX TLS (tcp, cuda_copy, sm, cuda_ipc, rdma…) |

**vLLM과의 결정적 차이**: vLLM은 커넥터(P2pNccl/LMCache/NIXL)마다 PP·비대칭TP 지원이 갈렸다. **TRT-LLM은 cache_transceiver 백엔드가 무엇이든 비대칭 PP·TP를 막지 않는다** — 호환성 게이트는 *병렬화 조합*이 아니라 *KV 동질성*(dtype·head수·layer수·non-MLA·beam=1)만 본다(`cacheFormatter.cpp::inquireSupport`). 그래서 dense GQA Qwen3-4B는 비대칭 TP·PP를 1급으로 통과한다. (이게 vLLM에서 TRT-LLM으로 넘어온 직접 이유 — `LEARNING_NOTES.md §프레임워크 판정·배경`.)

- 우리 설정: `cache_transceiver_config.backend: UCX`, Ethernet 노드 `UCX_TLS=tcp,cuda_copy,sm,self`.
- `DEFAULT == NIXL` (examples README). 우리는 명시적으로 UCX 고정.
- 근거: `tensorrt_llm/llmapi/llm_args.py:1807-1843` (CacheTransceiverConfig), `examples/disaggregated/README.md`.

---

### 1. KV 전송 백엔드 (vLLM의 "커넥터 종류 factory.py"에 대응)

vLLM처럼 커넥터 클래스를 고르는 게 아니라, **disagg YAML / 워커 extra YAML에 backend 문자열 하나**로 정한다.

| `cache_transceiver_config.backend` | 비고 |
|---|---|
| `UCX` | 우리 기본. UCX TLS로 transport 경로 제어. 컨테이너 사전설치 |
| `NIXL` | `DEFAULT`의 실체. 별도 플러그인 빌드 필요할 수 있음 |
| `MOONCAKE` | 외부 KV store. 본 실험 범위 밖 |
| `MPI` | 레거시. DEPRECATED 경로 |
| `DEFAULT` | =NIXL |

- ctx/gen **양쪽 동일 backend**여야 KV 전송 호환 (우리 ctx/gen extra YAML 둘 다 UCX).
- vLLM의 `--kv-transfer-config` JSON·`kv_producer/kv_consumer`·`kv_port`·`send_type=PUT_ASYNC`·`nccl_num_channels` 같은 P2pNccl 잡설정은 **TRT-LLM엔 없음** (orchestrator가 알아서 ctx→gen 라우팅).

---

### 2. "decode를 어떻게 스케일하나" — 병렬화 3분류 (개념은 공통, KV 전송 관점은 TRT-LLM)

| 방식 | 정체 | KV 전송 관점 | TRT-LLM 표현 |
|---|---|---|---|
| **DP**(복제) | full-model decode 인스턴스 N개, 라우터 분배 | KV 레이아웃 변환 불필요 | `generation_servers.num_instances: N` (xPyD) |
| **TP**(분할) | decode 1인스턴스를 N GPU에 head 분할 | 비대칭 TP면 head 재매핑 | `generation_servers.tensor_parallel_size: N` |
| **PP**(파이프라인) | decode 1인스턴스 layer를 stage 분할 | layer→stage KV 라우팅 | `generation_servers.pipeline_parallel_size: N` |

**dense 모델 주의(중요)**: Qwen3-4B(dense GQA)엔 **독립 DP 모델-병렬 축이 없다.** "DP"는 (1) xPyD 복제수(`num_instances`)이거나 (2) attention-DP(MoE/MLA 전용, GQA 무의미)일 뿐. → 우리 실제 모델-병렬 축 = **TP·PP 둘**, 토폴로지 축 = **xPyD(P·D 인스턴스 수)**. (vLLM 문서의 "DP=복제" 부분만 유효, attention-DP 칸은 dense엔 해당 없음.)

---

### 3. 지원 매트릭스 — TRT-LLM (vLLM과 정반대로 대부분 ✅)

범례: ✅ 1급 지원 / ⚠️ 지원하나 런타임 검증 필요 / ❌

| 병렬화 (P=prefill, D=decode) | vLLM 1세대(참고) | **TRT-LLM v1.2.1** | 근거 |
|---|---|---|---|
| DP / xPyD (복제 N) | ✅ | ✅ | `num_instances` |
| 대칭 TP (P=D) | ✅ | ✅ | 견고 |
| **비대칭 TP** (P tp1 → D tp4) | NixlConnector만 ✅ | **✅** | ctx/gen 독립 `tensor_parallel_size`, head 재매핑 |
| 대칭 PP (P=D) | ❌(WIP) | **⚠️** | 독립 `pipeline_parallel_size` 허용. 런타임 확인 |
| **비대칭 PP** (P pp1 → D pp4) | ❌ 전 커넥터 | **⚠️** | 막는 assert 없음. inquireSupport는 KV 동질성만 검사 |
| ⚠️ **ctx-PP → gen-TP** (P pp2 → D tp2) | — | **⚠️ hang 위험** | **알려진 #14020 hang** |

> 🔑 **핵심 결론 (vLLM과 반대)**:
> - vLLM은 "**PP-disagg를 아무 커넥터도 안 짜서**(WIP #40674) 전부 ❌, 비대칭 TP는 NIXL만"이었다.
> - **TRT-LLM은 ctx/gen이 각각 독립 `tensor_parallel_size`/`pipeline_parallel_size`를 받고, parse-time에 비대칭 PP/TP를 막는 assert가 없다.** 호환성 검사는 KV 동질성(dtype·head·layer·non-MLA·beam=1)만 → dense GQA Qwen3-4B 충족.
> - **단 "코드가 안 막음 ≠ 실제로 안 깨짐"** — 특히 **ctx-PP→gen-TP는 알려진 hang(#14020)**. → 어떤 (TP,PP) 조합이 실제로 도는지는 **Phase 0 스파이크 게이트**(`SETUP_LOG.md §Phase 0 지원 매트릭스`)로 실측 확정. 비대칭 TP 먼저(견고) → 비대칭 PP(미보증) 순.

#### 근거 (코드/메모리)
- ctx/gen 독립 TP·PP: `tensorrt_llm/llmapi/disagg_utils.py:171-211` (`extract_ctx_gen_cfgs`, instance_num_ranks=TP×PP×CP).
- 비대칭 막는 assert 없음 + KV 동질성만 검사: `LEARNING_NOTES.md §프레임워크 판정·배경` (`cacheFormatter.cpp::inquireSupport`).
- #14020 ctx-PP→gen-TP hang: release note KNOWN ISSUE (수정중 #15136). FlashInfer 금지(비대칭 TP 오출력 #6507) → `attn_backend: TRTLLM`.

---

### 4. 같은 노드 vs 다른 노드 — transport(UCX) + 멀티노드 분산

- **capability 게이트 아님.** same/cross 둘 다 UCX가 transport로 처리, 비대칭 여부와 무관.
  - same-node: UCX `cuda_copy/sm`(SHM). g5/g6/g6e는 **NVLink 없음** → cuda_ipc는 Nitro에서 막힐 수 있어 shm/cuda_copy로.
  - cross-node: UCX `tcp`. AWS xlarge/12xlarge는 **EFA(RDMA) 거의 없음** → TCP → 느림(TTFT 지배). → inter-node는 구조/correctness용, 깨끗한 성능은 intra-node(PCIe) 중심.
- **멀티노드 분산엔 Ray 아님 — TRT-LLM은 MPI.** 단, **독립 `trtllm-serve` 워커(우리 P1D1 방식)는 MPI도 불필요** (각 워커가 독립 프로세스, orchestrator가 HTTP/UCX로 조율). 한 인스턴스를 노드에 걸쳐 TP/PP로 펼치면 그때 MPI(`trtllm-llmapi-launch`/slurm)가 필요해짐. vLLM "PD엔 Ray 불필요"와 같은 결론, 메커니즘만 Ray→MPI.

---

### 5. 시나리오별 결론 (1P + "decode를 크게") — TRT-LLM은 PP도 후보

전제: prefill = 1 워커(TP1·PP1). "D를 어떻게 키우나"가 대칭/비대칭을 정함.

| 시나리오 | 분류 | 노드 | TRT-LLM 지금? | 방법/주의 |
|---|---|---|---|---|
| 다른노드 **DP** (D 복제 N) | DP, 대칭 | xPyD | ✅ | `num_instances:N` + orchestrator 라우팅 |
| 다른노드 **비대칭 TP** (D tp4) | 비대칭 TP | 2노드 | ✅ | ctx tp1 → gen tp4, UCX-TCP |
| 같은노드 **비대칭 PP** (D pp4) | 비대칭 PP | 1대(4GPU) | ⚠️ 실측 | assert 없음, Phase 0 확인 |
| **ctx-PP → gen-TP** | 비대칭 TP+PP | — | ⚠️ **hang** | #14020 — 300s 감시 |

**읽는 법**: vLLM에선 "PP 끼면 다 ❌"였지만, **TRT-LLM은 비대칭 TP·PP 둘 다 후보**다. 막히는 건 원리가 아니라 *특정 조합의 버그(#14020)* → Phase 0 게이트로 통과 조합만 본 스윕. 우리 목표("decode를 prefill보다 크게")는 **비대칭 TP(확실) 또는 PP(실측)** 로 달성.

> xPyD 라우팅: vLLM은 공식 proxy가 1P1D 한계였지만, **TRT-LLM `trtllm-serve disaggregated`는 `generation_servers.num_instances`>1 fan-out을 내장** → 별도 라우터 불필요.

---

### 6. per-side 측정 — prefill RPS/TPS vs decode RPS/TPS (소스 전수조사 + 적대적 검증 완료 2026-06-10)

vLLM은 각 노드 `/metrics`의 누적 토큰 카운터를 1초 차분해 per-side를 구했다. **TRT-LLM은 메커니즘이 다르다 — 그대로 이식하면 깨진다.** 9-에이전트 워크플로우로 v1.2.1 소스를 전수조사하고 핵심 3주장을 적대적으로 검증한 결과:

| 지표 | 공식 가능? | **TRT-LLM 실제 소스** |
|---|---|---|
| per-side **RPS** | 🟡 부분공식 | **orchestrator :8000 `/prometheus/metrics`의 `ctx_completed_requests_total` / `gen_completed_requests_total`** (단일 엔드포인트, role 접두사) 윈도우 차분. 대안=워커별 `trtllm_request_success_total` |
| per-side **TPS** | ❌ 공식불가(검증 confirmed) | **누적 토큰 카운터가 어디에도 없음** → **RPS × 고정 ISL/OSL**(그리드가 길이 고정, 우리가 강제) |

**핵심 발견 (vLLM 대비 더 깔끔):**
- **orchestrator(:8000)가 per-side 카운터를 단일 엔드포인트로 노출**한다. `instance_metric()`이 role별 접두사(`ctx`/`gen`)를 붙여 `completed_requests` Counter를 만들고(perf_metrics.py:93-111), prometheus_client이 `_total`을 자동 부착 → `ctx_completed_requests_total`/`gen_completed_requests_total`. ctx=prefill 완료, gen=decode 완료를 각각 셈(openai_client.py:267 `.inc()`). **워커별로 돌아다닐 필요 없음.**
- per-side **TPS는 진짜 공식 불가**: `/prometheus/metrics`엔 `request_success_total` + 4개 latency 히스토그램뿐, **토큰 카운터 0개**(collector.py:27-68). 워커 `/perf_metrics` per-request 레코드에도 토큰 필드 없음(timing/kv만, openai_server.py:414-452). → 토큰수는 **우리가 강제한 길이**(token-id prompt=prefill_len, max_tokens+ignore_eos=decode_len)로 곱해 파생, `usage.prompt/completion_tokens`로 검증.

**함정(정정 포함):**
1. orchestrator(:8000)엔 `/metrics`가 **없음(404)** 이지만 **`/prometheus/metrics`는 있음**(openai_disagg_server.py:156-158) — 여기에 ctx_/gen_ 카운터 족(族)이 노출됨. *(과거 메모: "orchestrator는 워커 포트로만" → 정정: per-side RPS는 orchestrator 단일 엔드포인트가 1순위.)*
2. 워커 `/metrics`(JSON iteration stats)와 워커 `/prometheus/metrics`(진짜 Prometheus, `return_perf_metrics:true`일 때만)는 다름 — 후자만 스크레이프. `numCtxTokens`/`numGenTokens`는 per-iteration 순간값이라 차분 금지.
3. 카운터는 워커가 `return_perf_metrics`로 떠야 생김(openai_server.py:138). ctx/gen extra YAML에 이미 켜둠.

**측정 방식(채택):**
- per-side **RPS** = orchestrator `/prometheus/metrics`를 **measured 윈도우 시작/끝에 1회씩 스냅샷** → `(end−start)/window_s`. (warmup 제외 = 변인통제). 1순위 키 `ctx_/gen_completed_requests_total`, 폴백 = 워커 `trtllm_request_success_total` 합.
- per-side **TPS** = `prefill_tps = prefill_rps × mean(prompt_tokens)`, `decode_tps = decode_rps × mean(completion_tokens)` (analyze.py, ~5줄).
- **전체 TTFT/TPOT/throughput** = 우리 sweep.py가 이미 측정(공식 benchmark_serving과 정의 동일). **전체 latency(E2EL)**도 sweep `e2e_s`로 직접 측정 — *공식 benchmark_serving은 E2EL이 기본 출력에서 빠짐*(§7).
- **KV 전송시간**(side 분해 보조) = orchestrator `/perf_metrics`의 `gen_perf_metrics[.perf_metrics].timing_metrics.kv_cache_transfer_end−start` (현 sweep.py 수집).
- 근거: perf_metrics.py:71,92-113(role 접두 카운터), openai_client.py:267(.inc), collector.py:27-68(워커 메트릭=토큰카운터 0), openai_server.py:138,414-452, openai_disagg_server.py:156-158.

---

### 7. 부하/측정 도구 — 공식 최대 + per-side만 커스텀 (vLLM 공식 벤치 브랜치 철학)

- **공식 부하 도구**: `python -m tensorrt_llm.serve.scripts.benchmark_serving` — vLLM `benchmark_serving.py`의 fork, **공식 disagg slurm 벤치(`examples/disaggregated/slurm/benchmark/run_benchmark.sh`)가 호출**. orchestrator :8000 OpenAI를 침. token-id ISL 고정(`--random-ids --tokenize-on-client --random-range-ratio 0`)·`--ignore-eos`·OSL(`--random-output-len`)·Poisson(`--request-rate --burstiness 1.0`)·`--max-concurrency` 지원.
- ⚠️ **검증으로 잡은 함정 — E2EL은 기본 출력에서 빠짐**: 결과 JSON에 throughput(무조건)·TTFT·TPOT는 기본(`--percentile-metrics`=`ttft,tpot,itl`)으로 나오지만 **E2EL(전체 latency)는 안 나옴**. → sweep가 **`--percentile-metrics ttft,tpot,itl,e2el`을 항상 명시**(benchmark_serving.py:537-539, 공식 run_benchmark.sh:71과 동일).
- ⚠️ **내장 warmup 없음**: benchmark_serving은 `--num-prompts`를 전부 측정(burn-in 없음) → sweep가 measured 전에 **소량 `--non-streaming` 호출로 warmup**(disagg UCX cold-start 흡수, run_benchmark.sh 패턴).
- **단일 엔드포인트 한계**: benchmark_serving은 한 base-url만 침(benchmark_serving.py:703-708) → **per-side를 못 냄**(전체만). → per-side는 별도 `prom_scrape.py`가 measured 윈도우 전/후 스냅샷(서브프로세스를 bracket).
- `trtllm-bench`는 `--engine_dir` in-process라 serving 엔드포인트 못 침 → 부하 도구 아님.
- **채택 구조 (구현됨 2026-06-11)**: **부하코어 = 공식 `benchmark_serving`** (sweep가 포인트마다 서브프로세스로 호출 → `bench_<point>.json`). sweep.py는 **오케스트레이션만**(그리드·resume·S3·metadata·warmup·per-side 스냅샷). analyze는 `bench_<point>.json`의 공식 집계(TTFT/TPOT/ITL/E2EL/throughput)를 읽고 + per-side(prom) + KV(perf) 병합. **커스텀 = 오케스트레이션 + `prom_scrape.py`(per-side) + `analyze.py`** — 전부 공식이 못 주는 것뿐. 손수 짠 aiohttp 부하는 제거됨.
- **정확한 측정 호출**(sweep `build_bench_args`): `--model $MODEL --backend openai --host H --port P --dataset-name random --random-ids --tokenize-on-client --random-input-len ISL --random-output-len OSL --random-range-ratio 0.0 --ignore-eos --num-prompts N --request-rate R --burstiness 1.0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 --save-result --result-dir DIR --result-filename bench_<point>.json --save-detailed`.

---

### 8. 소스 인덱스 (TRT-LLM v1.2.1, 박아두기)
- cache_transceiver 설정: `tensorrt_llm/llmapi/llm_args.py:1807-1843` (CacheTransceiverConfig: backend Literal DEFAULT/UCX/NIXL/MOONCAKE/MPI)
- disagg ctx/gen 독립 TP·PP 파서: `tensorrt_llm/llmapi/disagg_utils.py:115-211`
- per-side 메트릭: `tensorrt_llm/metrics/collector.py`(Prometheus), `tensorrt_llm/serve/openai_server.py:259-318`(/metrics·/prometheus/metrics·/perf_metrics 마운트), `:386-453`
- 공식 부하: `tensorrt_llm/serve/scripts/benchmark_serving.py`; 호출 예 `examples/disaggregated/slurm/benchmark/run_benchmark.sh`
- 프레임워크 판정·#14020·dense DP축 없음: `LEARNING_NOTES.md §프레임워크 판정·배경` (구 메모리 통합본)

---

### 9. 검증 방법 / TODO ("소스가 정의함 ≠ 런타임에 노출됨" → GPU 스모크로 확정)
소스는 확정(고신뢰). 남은 건 **런타임 노출**뿐 — P1D1 스모크에서 `curl :8000/prometheus/metrics`로 실측:
- [ ] **orchestrator per-side 카운터 실재 확인**: `ctx_completed_requests_total`·`gen_completed_requests_total`이 **non-zero로 노출**되나(role 접두 `ctx`/`gen` 적용, 단일 uvicorn에서 멀티프로세스 분실 없는지). 빠지면 폴백=워커 `trtllm_request_success_total`.
- [ ] **global vs per-worker**: ctx_/gen_completed가 role 전체 집계인지 워커별인지 → **1P3D**에서 decode RPS를 인스턴스별로 봐야 하면 워커별 `trtllm_request_success_total`로.
- [ ] **ctx↔gen 1:1**: 분리에선 요청 1개=ctx 1회+gen 1회 → 정상상태에서 `ctx_completed≈gen_completed`인지(재시도/다중응답 인플레 없는지) → `TPS=RPS×len` 성립 확인.
- [ ] **per-side TPS 정합**: `decode_rps×decode_len`이 전체 output throughput과 noise 내 일치하는지 한 점 대조.
- [ ] **윈도우 정렬**: 카운터 스냅샷 wall-clock 간격 vs sweep send-타임스탬프 윈도우가 achieved_rate와 noise 내 정합.
- [ ] **(선택) ctx_/gen_ latency 히스토그램**(`gen_first_token_latency_seconds` 등 openai_client.py:186,237,241,251 `.observe()`)이 non-zero `_count/_sum`이면 per-side TTFT/TPOT 공식 교차검증 가능.
- [ ] **Phase 0 게이트**(병행): TP·PP 조합 실측 → `SETUP_LOG.md §Phase 0 지원 매트릭스` (ctx-PP→gen-TP #14020 hang 300s 감시).

---

## 〔결정 기록〕 프레임워크 판정·배경 (구 CLAUDE에서 이전)

### 프레임워크 판정 (왜 TRT-LLM, 나머지 배제 — 공식 코드/이슈/PR로 검증 2026-06)
- **vLLM**: 비대칭 TP는 NixlConnector만 가능. **PP-in-PD 불가**(KV 전송 프로토콜에 PP 필드 없음, #40674). LMCache는 둘 다 ❌. → **배제.** (← 이 실험이 vLLM에서 넘어온 직접 이유)
- **SGLang**: PD KV 전송이 PP-aware(`base/conn.py`의 `KVArgs`에 `pp_rank`/`prefill_start_layer`/`prefill_end_layer`)하지만, **임의 비대칭 PP는 하드 assert로 차단** — `common/conn.py`의 `_resolve_rank_mapping`에 `assert pp_size == info.pp_size or pp_size == 1` ("Decode pp size should be equal to prefill pp size or 1"). 즉 **"prefill PP=N → decode PP=1" gather만** 가능. **대칭 PP-in-PD조차 크래시/출력손상 버그**(#15571 `PPMissingLayer.quant_method`, #16246; 수정 #19804는 #21189로 리버트). 비대칭 TP는 MLA 견고/비-MLA는 부하버그 #15674 OPEN. 무RDMA는 `mooncake_tcp` 또는 NIXL/UCX-TCP. ⚠️ **`--disaggregation-decode-tp` 플래그는 없음** — 비대칭 TP는 각 서버에 다른 `--tp-size`. → **배제.**
- **TensorRT-LLM**: **임의 비대칭 PP+TP를 1급으로 지원하는 유일한 유지보수 프로덕션 엔진**(context/generation 서버별 독립 tp/pp, UCX-over-TCP로 무EFA 가능). C++ `cacheFormatter.cpp::inquireSupport`는 PP 조합이 아니라 KV 동질성(dtype·head수·layer수·non-MLA·beam=1)만 검사 → dense GQA 충족. → **채택.**
- **Dynamo / llm-d**: vLLM 백엔드의 PP 한계를 그대로 상속 → 비대칭 PP 불가.
- **DistServe**(연구용): 비대칭 TP+PP는 깔끔하나 **per-phase DP 없음** + **CUDA IPC라 멀티노드 KV 전송 불가** → 단일노드 소형모델 연구용.

### DP 교차사실 (dense 모델 한정 — 중요)
"DP 안 됨"은 TRT-LLM 한정이 아니라 **dense GQA에선 어떤 엔진에서도 DP가 독립 모델-병렬 축이 아님**. DP의 3가지 의미: (1) 독립 샤딩 knob — TRT-LLM엔 아예 없음, vLLM/SGLang `--data-parallel-size`는 dense를 통째 복제, (2) attention-DP — MLA/MoE 전용(GQA 무의미), (3) 복제본 수/xPyD — 이것만 해당(리소스 배분). → **실험의 진짜 독립 축 = TP·PP 둘. DP는 인스턴스 수(xPyD)로만.** DP/EP를 진짜 축으로 보려면 MLA/MoE 모델(DeepSeek) 필요.

### T4(SM75) 교차사실 (중요)
T4 미지원은 **TRT-LLM·Dynamo만의 문제(최소 SM80)**. vLLM=T4 지원(SM75), SGLang=되나 과도기(prebuilt 휠에서 sm75 제거 #9207 → 소스빌드/구버전 핀 + Triton 3.2.0 다운). → 우리는 **T4 불필요 확정** → g5/g6/g6e(전부 SM80+).

### 1세대(vLLM) 배경 — 방법론의 출처
vLLM v0.21 fork + LMCache/NIXL, **Llama-3.1-8B**, AWS. 7개 config — monolithic(A1 TP2PP2 / A2 TP4PP1 / A3 TP1PP4, 4×T4) vs single big GPU(B, L40S) vs same-node PD(C, 4×T4 shm) vs cross-node PD(D, 2×L4 TCP). 측정 TTFT/TPOT/$per-Mtoken. vLLM이 **PD에서 PP 불가**라 2세대(TRT-LLM+Qwen3-4B)로 이전하며 sweep.py·analyze.py·setup.sh·2-phase·변인통제 **계승**. 원본 = `../vllm-disaggregation/disagg-exp/`.

### 전략 메모
Qwen3-4B(및 Llama-8B)는 GPU 1장에 올라가므로 **PP는 "크로스노드 메모리 분산" 용도** — 그게 바로 대부분 프레임워크에서 막힌 케이스(= 이 연구가 파고드는 지점). PP를 축으로 보는 연구적 의미가 약하면 더 큰 모델도 고려 가능.
