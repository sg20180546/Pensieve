# Debug Logging Control

모든 debug 문이 환경 변수 `PENSIEVE_DEBUG`로 제어됩니다.

## 사용 방법

### 1. Debug 출력 활성화 (상세 로그)
```bash
PENSIEVE_DEBUG=1 python test_multiturn_debug.py
```

또는 쉘 스크립트 사용:
```bash
bash run_debug.sh
```

**출력 예시:**
```
[DEBUG] _custom_generate] Packed session_id=session_1, req_idx=0, kv[0].shape=torch.Size([1, 8, 8, 128])
[DEBUG] _process_outputs] session_id=session_1, req_idx=0, kv_data is tuple
[DEBUG] _process_outputs] After extraction: first_k.shape=torch.Size([1, 8, 8, 128]), first_v.shape=torch.Size([1, 8, 8, 128])
[DEBUG] Layer 0: k.shape=torch.Size([1, 8, 8, 128]), v.shape=torch.Size([1, 8, 8, 128])
[DEBUG] num_generated=32, fill_last=0
```

### 2. Debug 출력 비활성화 (프로덕션 모드)
```bash
python test_multiturn_debug.py
```

또는 명시적으로:
```bash
PENSIEVE_DEBUG=0 python test_multiturn_debug.py
```

**특징:**
- Debug 로그가 출력되지 않음
- 최소 오버헤드
- 프로덕션 실험에 적합

### 3. 일반 데모 실행
```bash
# Debug 로그 없음
python main.py --mode pensieve --model gpt2 --max-new-tokens 20

# Debug 로그 포함
PENSIEVE_DEBUG=1 python main.py --mode pensieve --model gpt2 --max-new-tokens 20
```

### 4. 비교 벤치마크 (Pensieve vs vLLM)
```bash
# Debug 로그 없음 (권장 - 정확한 성능 측정)
python main.py --mode compare --model gpt2 --num-concurrent-users 3

# Debug 로그 포함
PENSIEVE_DEBUG=1 python main.py --mode compare --model gpt2 --num-concurrent-users 3
```

## 어디서 Debug 로그가 나오는가?

현재 다음 위치에서 debug 로그가 출력됩니다:

| 위치 | 목적 | 로그 내용 |
|------|------|---------|
| `_custom_generate()` | Tuple 패킹 확인 | session_id, req_idx, KV 형태 |
| `_process_outputs()` | Batch 추출 확인 | tuple 여부, 추출 후 shape |
| `_store_new_kv_chunks()` | 청크 저장 확인 | Layer 0 shape, fill_last 값, last_chunk shape |

## 성능 영향

**Debug 비활성화 시:**
- 오버헤드: **거의 없음** (Logger 호출 자체가 최소)
- Python 최적화: 상수 조건 확인이므로 JIT 최적화 가능

**Debug 활성화 시:**
- 오버헤드: Layer 0만 로그하므로 약간의 I/O 비용
- 프로파일링/디버깅 용도에 적합

## 원리

```python
# worker.py 시작 부분
_debug_enabled = os.getenv("PENSIEVE_DEBUG", "0") == "1"
if _debug_enabled:
    logger.setLevel(logging.DEBUG)
    # ...
else:
    logger.setLevel(logging.WARNING)

# 코드에서 사용
logger.debug(f"Layer 0: k.shape={k.shape}")  # PENSIEVE_DEBUG=1일 때만 출력
```

## 로그 수집

대량의 실험 결과를 파일에 저장하려면:

```bash
# 파일로 저장
PENSIEVE_DEBUG=1 python test_multiturn_debug.py > debug_output.log 2>&1

# 나중에 분석
grep "Layer 0:" debug_output.log
grep "ERROR\|Warning" debug_output.log
```

## 권장 사항

| 상황 | 설정 | 이유 |
|------|------|------|
| 버그 디버깅 | `PENSIEVE_DEBUG=1` | 상세 정보 필요 |
| 성능 벤치마크 | `PENSIEVE_DEBUG=0` | 정확한 측정 |
| 프로덕션 배포 | `PENSIEVE_DEBUG=0` | 최소 오버헤드 |
| 초기 테스트 | `PENSIEVE_DEBUG=1` | 동작 확인용 |

## 추가 커스터마이징

필요하면 더 많은 계층에서 debug 로그를 추가할 수 있습니다:

```python
# 모든 layer에서 로그
if True:  # 또는 _debug_enabled 체크
    logger.debug(f"Layer {layer_idx}: k.shape={k.shape}")
```

또는 선택적으로:

```python
# 특정 layer만
if layer_idx < 5:
    logger.debug(f"Layer {layer_idx}: ...")
```
