# Quick Start - Multi-Turn KV Cache Testing

## 🚀 한 줄 시작 (Debug 포함)

```bash
cd /Users/sj/pensieve && PENSIEVE_DEBUG=1 python test_multiturn_debug.py
```

## 🚀 한 줄 시작 (Debug 제외)

```bash
cd /Users/sj/pensieve && python test_multiturn_debug.py
```

## 각 상황별 명령어

### 1️⃣ 버그 찾기 (상세 로그)
```bash
PENSIEVE_DEBUG=1 python test_multiturn_debug.py
```

### 2️⃣ 성능 측정 (로그 없음)
```bash
python test_multiturn_debug.py
```

### 3️⃣ 간단한 데모 (로그 없음)
```bash
python main.py --mode pensieve --model gpt2
```

### 4️⃣ Pensieve vs vLLM 비교 (로그 없음)
```bash
python main.py --mode compare --model gpt2 --num-concurrent-users 3
```

### 5️⃣ Llama 모델 테스트 (다운로드 필요)
```bash
PENSIEVE_DEBUG=1 python test_multiturn_debug.py
# 수동으로 test_multiturn_debug.py 수정: model="meta-llama/Meta-Llama-3-8B"
```

## 예상 결과

### 성공 (Debug 활성화 시)
```
[DEBUG] _custom_generate] Packed session_id=session_1, req_idx=0, kv[0].shape=torch.Size([1, 8, 8, 128])
[DEBUG] _process_outputs] After extraction: first_k.shape=torch.Size([1, 8, 8, 128])
[DEBUG] Layer 0: k.shape=torch.Size([1, 8, 8, 128])
[DEBUG] num_generated=32, fill_last=0
✓ Multi-turn test PASSED
```

### 실패 (shape mismatch)
```
[DEBUG] Layer 0: k.shape=torch.Size([1, 8, 81, 128])
[DEBUG] num_generated=32, fill_last=25
[DEBUG] last_chunk.key_tensor.shape=torch.Size([1, 8, 38, 128])
ERROR: Sizes of tensors must match except in dimension 2...
```

## 파일 목록

| 파일 | 용도 |
|------|------|
| `test_multiturn_debug.py` | 빠른 테스트 스크립트 |
| `run_debug.sh` | Debug 자동 실행 스크립트 |
| `main.py` | 전체 데모/벤치마크 |
| `DEBUG_CONTROL.md` | 상세 로그 제어 가이드 |
| `RUN_DEBUG.txt` | 이전 debug 가이드 (참고용) |

## 로그 저장

```bash
# 파일에 저장
PENSIEVE_DEBUG=1 python test_multiturn_debug.py > test.log 2>&1

# 로그 확인
grep DEBUG test.log
```

## 환경 변수

| 변수 | 값 | 효과 |
|------|-----|------|
| `PENSIEVE_DEBUG` | `1` | Debug 로그 활성화 |
| `PENSIEVE_DEBUG` | `0` | Debug 로그 비활성화 (기본값) |

## 다음 단계

1. ✅ Test 실행: `PENSIEVE_DEBUG=1 python test_multiturn_debug.py`
2. ✅ 결과 확인: "PASSED" 또는 오류 확인
3. ✅ 성능 측정: Debug 끄고 다시 실행
4. ✅ 전체 비교: `python main.py --mode compare --model gpt2 --num-concurrent-users 3`

---

**궁금한 점?** [DEBUG_CONTROL.md](DEBUG_CONTROL.md) 참고
