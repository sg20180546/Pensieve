# Recent Changes Summary

## 1️⃣ Interactive Mode - 모델 미리 로드 (main.py:222-226)
```python
# ✅ Pre-load model before accepting user input
print("Loading model... (this may take a minute)")
print()
server._get_worker()  # Trigger model loading now
print()
```

**목적**: 사용자가 첫 입력을 하기 전에 모델을 미리 다운로드/로드
**효과**: 첫 "You: " 프롬프트에서 기다리지 않음

---

## 2️⃣ vLLM 기본 모드 - 샘플링 추가 (server.py:350-362)

### Before:
```python
outputs = self.model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=max_new_tokens,
    do_sample=False,  # ❌ Greedy decoding
    return_dict_in_generate=True,
    output_attentions=False,
)
```

### After:
```python
outputs = self.model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=max_new_tokens,
    do_sample=True,  # ✅ Enable sampling
    top_p=0.9,  # Nucleus sampling
    temperature=0.7,  # Temperature scaling
    return_dict_in_generate=True,
    output_attentions=False,
)
```

**목적**: OPT-125m 같은 작은 모델의 반복 문제 해결
**효과**: "I'm a guy, I'm a guy..." 반복 감소

---

## 3️⃣ Pensieve 모드 - 샘플링 추가 (worker.py:310-321)

### Before:
```python
# Greedy decoding
next_token_ids = torch.argmax(next_token_logits, dim=-1)  # ❌
```

### After:
```python
# ✅ Use sampling instead of greedy to avoid repetition loops
temperature = 0.7

# Apply temperature scaling to logits
scaled_logits = next_token_logits / temperature

# Convert to probabilities
probs = torch.softmax(scaled_logits, dim=-1)

# Sample from distribution
next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
```

**목적**: Pensieve 모드에서도 샘플링으로 다양한 생성
**효과**: 일관된 생성 품질 (vLLM과 동일)

---

## 설정값 설명

| 파라미터 | 값 | 효과 |
|---------|-----|------|
| `do_sample` | `True` | 샘플링 활성화 (greedy 비활성화) |
| `temperature` | `0.7` | 낮을수록 더 보수적 (높을수록 더 랜덤) |
| `top_p` | `0.9` | Nucleus sampling - 상위 90% 범위에서만 선택 |

---

## 영향받는 파일
- ✅ `main.py` (interactive mode pre-loading)
- ✅ `src/pensieve/server/server.py` (vLLM sampling)
- ✅ `src/pensieve/worker/worker.py` (Pensieve sampling)
