# SETUP_LOG — fork 셋업 작업 로그 (나중에 정리/methods용)

> 시간순 행동 + 검증 증거 기록. 결정/규칙은 `CLAUDE.md`, 설계는 `EXPERIMENT_PLAN.md` 참조.
> 작업 위치: 로컬 Mac(파일·git). 런타임(서버/추론)은 아직 안 함 — 다음 단계(원격 GPU).

## 2026-06-10 — TRT-LLM fork 클론 + 브랜치 + 핀 검증 + 문서 스캐폴딩

### 1. Fork 생성 (GitHub UI)
- `NVIDIA/TensorRT-LLM` → fork **`ddps-lab/disaggregation-TensorRT-LLM`**.
- "Copy the main branch only" **체크 해제**(전체 브랜치+태그 가져옴). GitHub fork엔 "특정 버전만 fork" 기능 없음 → 전체 fork 후 태그에서 브랜치 따는 방식.

### 2. 로컬 클론
```bash
git clone --no-recurse-submodules https://github.com/ddps-lab/disaggregation-TensorRT-LLM.git
```
- 서브모듈 제외: 로컬 빌드 안 하고 **컨테이너로 실행**할 거라 소스 트리만 필요.
- 결과: 2.0G (서브모듈 포함 시 훨씬 큼).

### 3. git-lfs 문제 + 우회 (재현 시 참고)
- 증상: clone 중 `git-lfs: command not found` → `checkout failed` (exit 128). git-lfs 미설치 환경.
- LFS 콘텐츠(docs 이미지/바이너리)는 실험에 불필요 → 설치 대신 **필터 우회**:
```bash
git config --local filter.lfs.process ""
git config --local filter.lfs.smudge cat
git config --local filter.lfs.clean cat
git config --local filter.lfs.required false
git checkout -f HEAD          # LFS 파일은 포인터로 남고 워킹트리 완성 → status clean
```

### 4. 버전 핀 브랜치
```bash
git checkout -b disagg-exp/trtllm-v1.2.1 v1.2.1   # 태그 v1.2.1 기준
git push -u origin disagg-exp/trtllm-v1.2.1        # push 완료
```
- `git describe --tags` → `v1.2.1` (정확히 일치).
- `v1.2.1` 태그가 upstream 실재함 확인 (GitHub API: `repos/NVIDIA/TensorRT-LLM/tags` 에 `v1.2.1` 존재).

### 5. v1.2.1 소스 코드 검증 (릴리즈 노트 주장 ≠ 코드 증거 — research-rigor #3)
| 확인 항목 | 증거 (파일:라인) | 결과 |
|---|---|---|
| 버전 | `tensorrt_llm/version.py` → `__version__ = "1.2.1"` | ✅ |
| disagg 진입점 | `tensorrt_llm/commands/serve.py:646` `@click.command("disaggregated")`, `:679 def disaggregated`, `:962` 커맨드 매핑 | ✅ |
| Qwen3 dense | `tensorrt_llm/_torch/models/modeling_qwen3.py` (MoE 아닌 dense — Qwen3-4B용) | ✅ |
| KV 전송 | `cache_transceiver` 참조 `_torch/pyexecutor/py_executor.py` 등 | ✅ |
| disagg config 스키마 | `examples/disaggregated/disagg_config.yaml` | ✅ |
| Qwen3 disagg 예시 | `examples/configs/curated/qwen3-disagg-prefill.yaml` (단 MoE용: `enable_attention_dp`/`moe_expert_parallel_size`) | ✅ |
| 컨테이너 ref 패턴 | `nvcr.io/nvidia/tensorrt-llm/release:x.y.z` → 핀 `:1.2.1` | ✅ |

**핵심 발견**: `disagg_config.yaml`에서 `context_servers`/`generation_servers` 각각 독립 `tensor_parallel_size`·`pipeline_parallel_size` → 대칭/비대칭 자유. **`generation_servers.num_instances` = xPyD의 D 개수** (1P3D → `num_instances: 3`).

### 6. 문서 스캐폴딩 (코드 아님 — 사용자 방침 준수)
- `disagg-exp/CLAUDE.md` 신규 — 단일 진실원(목표·핀·dealbreaker·변인통제·스키마).
- `disagg-exp/EXPERIMENT_PLAN.md` — 기존 vLLM `../vllm-disaggregation/disagg-exp/`에서 복사.
- `disagg-exp/SETUP_LOG.md` — 이 파일.
- commit `96942817c1` (docs) → push 완료. (커밋은 `disagg-exp/trtllm-v1.2.1` 브랜치, default main 아님)

### 7. 글로벌 메모리 갱신
- `disagg-exp-experiment-overview.md`: "fork 이전 예정" → "셋업 완료"(repo 경로·브랜치·핀·검증사실).

## 2026-06-10 (2) — 하네스 코드 작성 + 소스 대조 검증 (방침 변경: Claude가 코드 작성)

### 8. 흐름: 소스 분석 → 작성 → 검증
- **(a) 리서치 워크플로우** (6 병렬 리더 → 빌드스펙): v1.2.1 소스에서 serve CLI·disagg 스키마·LLM args·Qwen3·예제·하네스 이식점을 코드레벨 확정. 핵심: 모델 positional(`--model` 없음), `--pp_size` 존재, `--dtype` CLI 없음, `enable_block_reuse`는 `kv_cache_config` 하위, orchestrator host/port는 disagg YAML에서만, sweep payload(token-id/ignore_eos/min_tokens) 전부 유효.
- **(b) 작성한 파일 8개** (모두 검증된 사실 기반):
  - 신규: `launch_trtllm.sh`(role별 trtllm-serve + disagg YAML 런타임 생성, env 파라미터화), `ctx_extra_llm_api_options.yaml`, `gen_extra_llm_api_options.yaml`, `disagg_config.yaml`(1P1D 정적), `trtllm_support_matrix.md`(Phase0 게이트), `README.md`(실행 가이드+vLLM대비 역할).
  - 이식: `sweep.py`(4곳), `analyze.py`(2곳), `setup.sh`(컨테이너 모델 재작성, 수집기 보존).
- **(c) 검증 워크플로우** (3 적대적 체커 vs 소스): **blocker 0, major 1, minor 11**(7개는 "정합 확인").
  - major: `launch_trtllm.sh` 빈배열 확장(`"${role_flag[@]}"`/`${PIDS[@]}`)이 bash<4.4(macOS)서 `unbound variable` → 가드(`[@]+...`/`[*]:-`/count check)로 수정. **로컬 bash 3.2.57서 안 깨짐 실증.**
  - minor 픽스: build_urls 공백트림, URL 개수 사전검증, CACHE_BACKEND/TP·PP가 URL모드선 inert(문서화), cuda_graph/server_role 주석 정확성.
- **로컬 정적검사 통과**: `bash -n`(launch/setup), `py_compile`(sweep/analyze), yaml.safe_load(3개).

## 현재 상태
- ✅ fork·브랜치(v1.2.1)·핀·문서 = 완료, origin 동기화.
- ✅ **하네스 코드 8개 작성 + 소스 대조 검증 완료** (blocker 0, 로컬 정적검사 통과).
- ⬜ 원격 GPU(g6e.12xlarge 등)에서 **Phase 0 스파이크 = 미시작** = 런타임 검증. (어떤 TP/PP 조합 OK인지, ctx-PP→gen-TP hang #14020 실측, max_batch_size/free_gpu_mem OOM 한계, server_role 필요 여부)

## 아직 안 한 것 / 주의 (코드에서 확정 못 함 → GPU에서)
- 로컬에서 TRT-LLM **빌드/실행 안 함** (컨테이너로 원격에서). 코드 정확성은 정적, **동작 검증은 GPU**.
- 컨테이너 `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` 실제 pull·기동, Qwen3-4B 로드, KV전송 동작, 출력정확성(비분리 비교) = 전부 Phase 0.
- COST_PER_HR(analyze.py) 단가는 최종 인스턴스/리전 확정 후 채울 것 (현재 placeholder).
