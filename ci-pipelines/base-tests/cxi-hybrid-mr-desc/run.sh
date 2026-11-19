#!/bin/bash
# CI test for the libfabric CXI patch prov_cxi_hybrid_mr_desc_copy.
#
# Builds hyb_desc_test.c against the image's libfabric and CUDA, then runs:
#   1. correctness: all AMO/RMA ops, host and GPU sources, hybrid on/off,
#      valid/NULL/garbage/other-MR descs, IDC and DMA paths;
#   2. FI_INJECT correctness: IDC and DMA bounce paths, with and without
#      FI_COMPLETION, source poisoned right after the post;
#   3. expected failures (XFAIL) of upstream v2.6.0;
#   4. MR cache searches per post: 0 with a valid desc proves the patch is
#      active. Without the patch these cases do 1-2 searches per post.
# Needs one GH200 node with a CXI NIC and a GPU.
set -uo pipefail

here=$(dirname "$(realpath "$0")")
work=$(mktemp -d /tmp/cxi-hybrid-mr-desc.XXXXXX)
B=$work/hyb_desc_test
N=50
fails=0

export FI_PROVIDER=cxi
cuda=${CUDA_HOME:-/usr/local/cuda}
gcc -O2 -Wall -o "$B" "$here/hyb_desc_test.c" ${CXI_TEST_CFLAGS:-} \
  -I"$cuda/include" -L"$cuda/lib64" -lfabric -lcudart || exit 1
echo "libfabric: $(ldd "$B" | awk '/libfabric/ {print $3}')"

fail() { echo "FAIL  $*"; fails=$((fails + 1)); }

# 1 and 2: correctness. Each run prints FAIL lines and a SUMMARY.
run_mode() {  # mode, env...
  local mode=$1 h out r; shift
  for h in 1 0; do
    out=$(env "$@" timeout 300 "$B" "$mode" "$h" 2>&1); r=$?
    printf '%s\n' "$out" | grep -E '^(FAIL|SUMMARY)' |
      sed "s|^|[$mode ${*:-default}] |"
    [[ $r -eq 0 ]] || fail "$mode hybrid=$h env=[${*:-default}] exit=$r"
  done
}
echo "== correctness"
run_mode check
run_mode check FI_CXI_DISABLE_NON_INJECT_AMO_IDC=1 FI_CXI_DISABLE_NON_INJECT_RMA_IDC=1
run_mode check FI_CXI_DISABLE_HMEM_DEV_REGISTER=1
run_mode check CXI_FORK_SAFE=1
echo "== FI_INJECT correctness"
run_mode inject
run_mode inject FI_CXI_DISABLE_HMEM_DEV_REGISTER=1

# 3: expected failures. XPASS means the behavior changed; review and update.
echo "== expected failures"
xfail() {  # name, expected exit, env..., --, args...
  local name=$1 want=$2 envs=() r; shift 2
  while [[ $1 != -- ]]; do envs+=("$1"); shift; done; shift
  # The explicit exit keeps bash from exec'ing the command, so the subshell
  # (with stderr discarded) reports the "Segmentation fault" notice.
  (env "${envs[@]}" timeout 120 "$B" "$@"; exit $?) >/dev/null 2>&1; r=$?
  if [[ $r -eq $want ]]; then echo "XFAIL $name (exit=$r)"
  else fail "$name: exit=$r, expected $want (XPASS if 0)"; fi
}
for h in 1 0; do
  # fetching_amo_flush path dereferences a NULL txc->read_cntr (SIGSEGV).
  xfail "fetching AMO + FI_DELIVERY_COMPLETE, no counter, hybrid=$h" 139 -- famo-dc $h
  # fi_domain() rejects a zero-size MR cache with FI_HMEM (-FI_EINVAL).
  xfail "FI_MR_CACHE_MAX_COUNT=0, hybrid=$h" 2 FI_MR_CACHE_MAX_COUNT=0 -- check $h
done

# 4: MR cache searches per post, from "MR cache stats: searches X" that
# libfabric logs at domain close (FI_LOG_LEVEL=info).
echo "== MR cache searches per post (n=0 vs n=$N)"
searches() {  # env..., -- args...
  local envs=() out
  while [[ $1 != -- ]]; do envs+=("$1"); shift; done; shift
  out=$(env "${envs[@]}" FI_LOG_LEVEL=info FI_LOG_SUBSYS=mr timeout 120 \
        "$B" mrcache "$@" 2>&1) || { echo error; return; }
  grep -q 'result_ok=1' <<<"$out" || { echo error; return; }
  grep -o 'MR cache stats: searches [0-9]*' <<<"$out" |
    awk '{ s += $NF } END { if (NR) print s; else print "nolog" }'
}
# name | env | hybrid op mem desc key | expected searches per post
while IFS='|' read -r name envs args want; do
  [[ -z $name ]] && continue
  [[ $envs == - ]] && envs=
  read -r h op mem desc key <<<"$args"
  s0=$(searches $envs -- "$h" "$op" "$mem" 0 "$desc" "$key")
  s1=$(searches $envs -- "$h" "$op" "$mem" $N "$desc" "$key")
  if [[ $s0 =~ ^[0-9]+$ && $s1 =~ ^[0-9]+$ ]]; then
    d=$((s1 - s0))
    if (( d % N == 0 )); then got=$((d / N)); else got="~$((d / N))"; fi
  else
    got="error($s0,$s1)"
  fi
  if [[ $got == "$want" ]]; then printf 'PASS  %-34s %s/post\n' "$name" "$got"
  else fail "$(printf '%-34s %s/post, want %s' "$name" "$got" "$want")"; fi
done <<'EOF'
amo-sum host|-|1 sum host valid opt|0
amo-sum cuda|-|1 sum cuda valid opt|0
amo-write host|-|1 write host valid opt|0
famo-sum host|-|1 famo host valid opt|0
cswap host|-|1 cswap host valid opt|0
rma-8B host|-|1 rma8 host valid opt|0
rma-8B cuda|-|1 rma8 cuda valid opt|0
rma-192B host|-|1 rma192 host valid opt|0
DMA amo-write host|FI_CXI_DISABLE_NON_INJECT_AMO_IDC=1|1 write host valid opt|0
inject IDC amo-sum host|-|1 inj-sum host valid opt|0
inject IDC amo-write host|-|1 inj-write host valid opt|0
inject IDC cswap host|-|1 inj-cswap host valid opt|0
inject IDC rma-8B host|-|1 inj-rma8 host valid opt|0
inject DMA amo-sum host|-|1 inj-sum host valid std|0
inject DMA amo-sum cuda|-|1 inj-sum cuda valid std|0
inject DMA cswap host|-|1 inj-cswap host valid std|0
inject DMA rma-8B host|-|1 inj-rma8 host valid std|0
inject DMA rma-8B cuda|-|1 inj-rma8 cuda valid std|0
inject DMA rma-192B host|-|1 inj-rma192 host valid std|0
inject DMA amo-write (bounce buf)|-|1 inj-write host valid std|1
control: NULL desc|-|1 sum host null opt|1
control: hybrid off|-|0 sum host valid opt|1
control: DMA amo-sum|FI_CXI_DISABLE_NON_INJECT_AMO_IDC=1|1 sum host valid opt|0
EOF

rm -rf "$work"
echo "SUMMARY cxi-hybrid-mr-desc fails=$fails"
exit $((fails > 0))
