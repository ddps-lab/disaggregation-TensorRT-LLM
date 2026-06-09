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

### disagg_config.yaml 스키마 읽는 법 (우리 핵심 발견)
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
→ P/D 각각 독립 TP·PP = **대칭/비대칭 자유**, `generation_servers.num_instances` = **D 스케일 노브**. 이 한 파일로 실험 두 축이 다 표현된다.

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

## 더 공부할 것 (TODO — 채워가기)
- [ ] UCX가 TCP 폴백할 때 정확히 어떤 경로(`UCX_TLS`)로 가는지, shm/cuda_copy 차이
- [ ] TP head 재매핑이 비대칭 TP에서 실제로 어떻게 동작하는지 (cacheFormatter)
- [ ] PP에서 KV가 단계별로 어떻게 분할·전송되는지 (왜 비대칭 PP가 미보증인지)
- [ ] disable_overlap_scheduler가 정확히 무엇을 끄는지
- [ ] CUDA Graph가 decode throughput을 왜 크게 바꾸는지
