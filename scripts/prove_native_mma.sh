#!/usr/bin/env bash
# Requirement 5: tie the *executed* kernel to *its own* OMMA instructions.
# Not "the library contains OMMA somewhere" - we disassemble exactly one function
# and additionally run a negative control to prove the filter really filters.
set -uo pipefail
cd "$HOME/fp4_bf16_study"
export PATH=/usr/local/cuda/bin:$PATH
LOG=raw/native_mma_proof.txt
: > "$LOG"
{
echo "=== step 1: which CUDA kernel executes for torch._scaled_mm(float4)? ==="
.venv/bin/python scripts/prove_native_mma.py 2>&1 || { echo "profiler step failed"; exit 1; }

python3 - > /tmp/kname.txt <<'PY'
import json
d = json.load(open('raw/native_mma_kernels.json'))
import re
cand = [e for e in d if re.search(r'fp4|nvf4|blockscal|blocked|scaled|cutlass|nvjet', e['kernel'], re.I)]
e = cand[0] if cand else (d[0] if d else None)
print(e['kernel'] if e else '')
PY
K=$(cat /tmp/kname.txt)
echo
echo "=== step 2: executed kernel selected ==="
echo "kernel: $K"
if [ -z "$K" ]; then echo "no kernel captured"; exit 1; fi
case "$K" in
  *bf16*|*bfloat16*) echo "WARNING: kernel name mentions bf16 - verify it is not a dequant path";;
esac

echo
echo "=== step 3: locate the owning library and disassemble ONE function ==="
FOUND=0
for LIB in "$HOME/fp4_bf16_study/.venv/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so" \
           /usr/local/cuda/lib64/libcublasLt.so.13 \
           /usr/local/cuda/lib64/libcublas.so.13 ; do
  [ -f "$LIB" ] || continue
  echo "--- probing $(basename "$LIB") ---"
  # (a) try the raw name, (b) fall back to a mangled symbol containing a distinctive token
  TOK=$(printf '%s' "$K" | tr -d ' ' | cut -c1-24)
  MANGLED=$(timeout 900 cuobjdump -symbols "$LIB" 2>/dev/null | grep -m1 -F "$TOK" | awk '{print $NF}')
  for ARG in "$K" "$MANGLED"; do
    [ -n "${ARG:-}" ] || continue
    N=$(timeout 900 cuobjdump -sass -fun "$ARG" "$LIB" 2>/dev/null | grep -c "OMMA")
    H=$(timeout 900 cuobjdump -sass -fun "$ARG" "$LIB" 2>/dev/null | grep -c "HMMA")
    if [ "${N:-0}" -gt 0 ] || [ "${H:-0}" -gt 0 ]; then
      echo "function matched: ${ARG:0:80}"
      echo "  OMMA(block-scaled FP4) = $N"
      echo "  HMMA(BF16/FP16)        = $H"
      timeout 900 cuobjdump -sass -fun "$ARG" "$LIB" 2>/dev/null \
        | grep -oE "OMMA[A-Z0-9._]*|HMMA[A-Z0-9._]*|QMMA[A-Z0-9._]*" | sort | uniq -c | sort -rn | head -6
      FOUND=1
      SEL="$ARG"; SELLIB="$LIB"
      break 2
    fi
  done
done

echo
echo "=== step 4: negative control (must be 0) ==="
NEG=$(cuobjdump -sass -fun "definitely_not_a_kernel_$RANDOM" "${SELLIB:-$HOME/fp4_bf16_study/bin/79a_v420}" 2>/dev/null | grep -c "OMMA")
echo "negative-control OMMA count = $NEG (expected 0)"

echo
echo "=== verdict ==="
if [ "$FOUND" -eq 1 ] && [ "${NEG:-1}" -eq 0 ] && [ "${N:-0}" -gt 0 ]; then
  echo "NATIVE_FP4_MMA_CONFIRMED: the executed kernel's own SASS contains $N OMMA instructions"
  echo "                          and a bogus function name yields 0 (filter is real)"
else
  echo "NOT CONFIRMED: found=$FOUND omma=${N:-0} negative_control=$NEG"
fi
} 2>&1 | tee "$LOG"
echo "log -> $LOG"
