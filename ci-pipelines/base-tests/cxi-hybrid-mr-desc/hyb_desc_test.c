/*
 * Test for the libfabric CXI patch prov_cxi_hybrid_mr_desc_copy: in hybrid
 * MR desc mode, CPU copies of operands use the caller's MR desc instead of
 * an internal MR cache lookup.
 *
 * One process, one domain, two RDM endpoints (initiator and target) on the
 * same NIC. Targets are host memory: key 7 is an optimized MR (IDC capable),
 * key 1000 a standard MR (never IDC, so FI_INJECT uses the DMA bounce path).
 * Sources are host or cudaMalloc memory, at a non-zero offset in a larger MR.
 *
 * Usage:
 *   hyb_desc_test check   <hybrid>   all ops x desc modes, optimized key
 *   hyb_desc_test inject  <hybrid>   FI_INJECT: both keys, with and without
 *                                    FI_COMPLETION, source poisoned after post
 *   hyb_desc_test famo-dc <hybrid>   known v2.6.0 crash (expected failure)
 *   hyb_desc_test mrcache <hybrid> <[inj-]op> <host|cuda> <n> <valid|null>
 *                         [opt|std]  n posts, then a clean close, so the MR
 *                                    cache logs its stats (FI_LOG_LEVEL=info)
 *
 * Env: FI_CXI_DISABLE_NON_INJECT_{AMO,RMA}_IDC=1 sends check mode to the DMA
 * paths. FI_CXI_DISABLE_HMEM_DEV_REGISTER=1 tests MDs with no dev reg handle.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda_runtime.h>
#include <rdma/fabric.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_cxi_ext.h>

#define CHK(x) do { int _r = (x); if (_r) { \
	fprintf(stderr, "%s:%d %s = %d (%s)\n", __FILE__, __LINE__, #x, _r, \
		fi_strerror(-_r)); exit(2); } } while (0)
#define CUCHK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
	fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, \
		cudaGetErrorString(_e)); exit(2); } } while (0)

#define OPT_KEY 7		/* < 100: optimized MR, IDC capable */
#define STD_KEY 1000		/* >= 100: standard MR, never IDC */
#define TGT_LEN 4096
#define SRC_LEN (64 * 1024)
#define SRC_OFF 12344		/* non-zero, 8-byte aligned, not page aligned */
#define MAX_INJECT 192		/* CXIP_INJECT_SIZE */

enum mem { MEM_HOST, MEM_CUDA };
enum dmode { D_VALID, D_NULL, D_GARBAGE, D_OTHER_MR };
static const char *mem_s[] = { "host", "cuda" };
static const char *dmode_s[] = { "valid", "null", "garbage", "other-mr" };

enum kind { K_AMO, K_FETCH, K_CMP, K_RMA };

/* One operation and its expected result. AMOs use target word 0 and the
 * source/compare/result words at SRC_OFF + soff. RMA writes len bytes from
 * SRC_OFF + soff to target offset toff.
 */
struct op {
	const char *id;
	enum kind kind;
	enum fi_op op;
	enum fi_datatype dt;
	uint64_t init, operand, cmp, want, want_res;
	bool chk_res;
	size_t len, soff;
	uint64_t toff;
};

#define MS_INIT 0xff00ff00ff00ff00ULL
#define MS_VAL 0x123456789abcdef0ULL
#define MS_MASK 0x00000000ffffffffULL

static const struct op ops[] = {
	{ "sum", K_AMO, FI_SUM, FI_INT64, 1000, 0x1234, 0, 1000 + 0x1234 },
	{ "write", K_AMO, FI_ATOMIC_WRITE, FI_UINT64, 0xdeadbeef,
	  0x0102030405060708ULL, 0, 0x0102030405060708ULL },
	{ "famo", K_FETCH, FI_SUM, FI_INT64, 500, 77, 0, 577, 500, true },
	{ "cswap", K_CMP, FI_CSWAP, FI_UINT64, 42, 99, 42, 99, 42, true },
	{ "cswap-nm", K_CMP, FI_CSWAP, FI_UINT64, 43, 99, 42, 43, 43, true },
	{ "mswap", K_CMP, FI_MSWAP, FI_UINT64, MS_INIT, MS_VAL, MS_MASK,
	  (MS_INIT & ~MS_MASK) | (MS_VAL & MS_MASK) },
	/* 192 B: CXIP_INJECT_SIZE and max unrestricted IDC. 224 B: max
	 * restricted IDC. 1 B and 13 B at odd offsets.
	 */
	{ "rma8", K_RMA, .len = 8, .soff = 256, .toff = 1024 },
	{ "rma192", K_RMA, .len = 192, .soff = 512, .toff = 1283 },
	{ "rma224", K_RMA, .len = 224, .soff = 1024, .toff = 1536 },
	{ "rma1", K_RMA, .len = 1, .soff = 1539, .toff = 1795 },
	{ "rma13", K_RMA, .len = 13, .soff = 2061, .toff = 2051 },
};
#define NOPS (sizeof(ops) / sizeof(ops[0]))

struct region {
	enum mem mem;
	uint8_t *base;		/* SRC_LEN bytes */
	struct fid_mr *mr;
};

struct descs { void *s, *c, *r; };

static struct fi_info *info;
static struct fid_fabric *fab;
static struct fid_domain *dom;
static struct fid_av *av;
static struct fid_ep *ep_i, *ep_t;
static struct fid_cq *cq_i, *cq_t;
static struct fid_mr *opt_mr, *std_mr;
static uint8_t *opt_tgt, *std_tgt;	/* host targets for OPT_KEY, STD_KEY */
static fi_addr_t tgt_addr;
static struct region src[2], aux[2], res[2];	/* operand, compare, result */
static uint8_t garbage_blob[1024];	/* fake desc; must be ignored */
static uint64_t next_key = 200;
static int hybrid, passes, failures;
static bool selective;	/* TX CQ uses FI_SELECTIVE_COMPLETION */
static bool force_dc;	/* FI_DELIVERY_COMPLETE on fetching AMOs too */

static double now_ns(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e9 + ts.tv_nsec;
}

/* Drive progress; exit on a CQ error. */
static void cq_progress(void)
{
	struct fi_cq_err_entry err = {};

	if (fi_cq_read(cq_i, NULL, 0) == -FI_EAVAIL) {
		fi_cq_readerr(cq_i, &err, 0);
		fprintf(stderr, "CQ error: %d (%s) prov_errno %d\n", err.err,
			fi_strerror(err.err), err.prov_errno);
		exit(3);
	}
}

static void wait_cq(int n)
{
	struct fi_cq_entry ce[16];
	double t0 = now_ns();

	while (n > 0) {
		ssize_t r = fi_cq_read(cq_i, ce, n < 16 ? n : 16);

		if (r > 0)
			n -= r;
		else if (r == -FI_EAVAIL)
			cq_progress();
		else if (r != -FI_EAGAIN || now_ns() - t0 > 20e9) {
			fprintf(stderr, "fi_cq_read: %zd\n", r);
			exit(3);
		}
	}
}

static void put_val(struct region *r, size_t off, const void *v, size_t len)
{
	if (r->mem == MEM_CUDA)
		CUCHK(cudaMemcpy(r->base + off, v, len, cudaMemcpyHostToDevice));
	else
		memcpy(r->base + off, v, len);
}

static void get_val(struct region *r, size_t off, void *v, size_t len)
{
	if (r->mem == MEM_CUDA)
		CUCHK(cudaMemcpy(v, r->base + off, len, cudaMemcpyDeviceToHost));
	else
		memcpy(v, r->base + off, len);
}

static void poison(struct region *r, size_t off, size_t len)
{
	uint8_t p[256];

	memset(p, 0xcc, len);
	put_val(r, off, p, len);
}

/* Wait until len bytes at p (host) or r+off (host or device) equal want. */
static bool poll_eq(const uint8_t *p, struct region *r, size_t off,
		    const void *want, size_t len)
{
	uint8_t cur[256];
	double t0 = now_ns();

	do {
		if (r)
			get_val(r, off, cur, len);
		else
			memcpy(cur, p, len);
		if (!memcmp(cur, want, len))
			return true;
		cq_progress();
	} while (now_ns() - t0 < 2e9);
	return false;
}

static struct fid_mr *reg(void *buf, size_t len, enum fi_hmem_iface iface,
			  uint64_t access, uint64_t key, struct fid_ep *ep)
{
	struct fid_mr *mr;
	struct iovec iov = { buf, len };
	struct fi_mr_attr attr = {
		.mr_iov = &iov, .iov_count = 1, .access = access,
		.requested_key = key, .iface = iface,
	};

	CHK(fi_mr_regattr(dom, &attr, 0, &mr));
	CHK(fi_mr_bind(mr, &ep->fid, 0));
	CHK(fi_mr_enable(mr));
	return mr;
}

static void region_init(struct region *r, enum mem mem)
{
	r->mem = mem;
	if (mem == MEM_CUDA)
		CUCHK(cudaMalloc((void **)&r->base, SRC_LEN));
	else
		r->base = aligned_alloc(4096, SRC_LEN);
	r->mr = reg(r->base, SRC_LEN,
		    mem == MEM_CUDA ? FI_HMEM_CUDA : FI_HMEM_SYSTEM,
		    FI_READ | FI_WRITE | FI_SEND | FI_RECV, next_key++, ep_i);
}

static void *desc_for(struct region *r, struct region *other, enum dmode d)
{
	switch (d) {
	case D_VALID: return fi_mr_desc(r->mr);
	case D_GARBAGE: return garbage_blob;
	case D_OTHER_MR: return fi_mr_desc(other->mr);
	default: return NULL;
	}
}

static struct descs descs_for(enum mem m, enum dmode d)
{
	struct descs x = {
		desc_for(&src[m], &aux[m], d),
		desc_for(&aux[m], &src[m], d),
		/* The NIC writes the result through its desc, so a desc of
		 * another MR is a user error there.
		 */
		d == D_OTHER_MR ? NULL : desc_for(&res[m], &src[m], d),
	};
	return x;
}

static uint64_t check_flags(const struct op *o)
{
	/* Fetching AMOs with FI_DELIVERY_COMPLETE and no bound counter crash
	 * in v2.6.0 (see famo-dc), so only force_dc adds it to them.
	 */
	bool dc = o->kind == K_AMO || o->kind == K_RMA || force_dc;

	return FI_COMPLETION | (dc ? FI_DELIVERY_COMPLETE : 0);
}

/* Post one operation; retry on -FI_EAGAIN. */
static ssize_t post_op(const struct op *o, enum mem m, struct descs ds,
		       bool std, uint64_t flags)
{
	uint64_t key = std ? STD_KEY : OPT_KEY;
	struct fi_ioc ioc = { src[m].base + SRC_OFF + o->soff, 1 };
	struct fi_ioc cioc = { aux[m].base + SRC_OFF + o->soff, 1 };
	struct fi_ioc rioc = { res[m].base + SRC_OFF + o->soff, 1 };
	struct fi_rma_ioc tioc = { .addr = 0, .count = 1, .key = key };
	struct fi_msg_atomic amsg = {
		.msg_iov = &ioc, .desc = &ds.s, .iov_count = 1,
		.addr = tgt_addr, .rma_iov = &tioc, .rma_iov_count = 1,
		.datatype = o->dt, .op = o->op,
	};
	struct iovec iov = { src[m].base + SRC_OFF + o->soff, o->len };
	struct fi_rma_iov riov = { .addr = o->toff, .len = o->len, .key = key };
	struct fi_msg_rma rmsg = {
		.msg_iov = &iov, .desc = &ds.s, .iov_count = 1,
		.addr = tgt_addr, .rma_iov = &riov, .rma_iov_count = 1,
	};
	ssize_t ret;

	for (;;) {
		switch (o->kind) {
		case K_RMA:
			ret = fi_writemsg(ep_i, &rmsg, flags);
			break;
		case K_AMO:
			ret = fi_atomicmsg(ep_i, &amsg, flags);
			break;
		case K_FETCH:
			ret = fi_fetch_atomicmsg(ep_i, &amsg, &rioc, &ds.r, 1,
						 flags);
			break;
		default:
			ret = fi_compare_atomicmsg(ep_i, &amsg, &cioc, &ds.c, 1,
						   &rioc, &ds.r, 1, flags);
		}
		if (ret != -FI_EAGAIN)
			return ret;
		cq_progress();
	}
}

/* Run one operation and check the target (and result) values. For
 * FI_INJECT, the source and compare buffers are poisoned right after the
 * post returns: the target must still get the original values.
 */
static bool run_one(const struct op *o, enum mem m, struct descs ds,
		    bool std, uint64_t flags, char *det, size_t dlen)
{
	uint8_t *T = std ? std_tgt : opt_tgt;
	size_t off = SRC_OFF + o->soff;
	uint8_t pat[256], guard[4];
	uint64_t zero = 0, got, fetched;
	ssize_t ret;
	bool ok;

	memset(guard, 0xee, sizeof(guard));
	if (o->kind == K_RMA) {
		for (size_t i = 0; i < o->len; i++)
			pat[i] = (uint8_t)(o->len + 7 * i + 1);
		memset(T + o->toff - 4, 0xee, o->len + 8);
		put_val(&src[m], off, pat, o->len);
	} else {
		memcpy(T, &o->init, 8);
		put_val(&src[m], off, &o->operand, 8);
		put_val(&aux[m], off, &o->cmp, 8);
		put_val(&res[m], off, &zero, 8);
	}

	ret = post_op(o, m, ds, std, flags);
	if (flags & FI_INJECT) {
		poison(&src[m], off, o->kind == K_RMA ? o->len : 8);
		poison(&aux[m], off, 8);
	}
	if (ret) {
		snprintf(det, dlen, "post=%zd (%s)", ret, fi_strerror(-ret));
		return false;
	}
	if ((flags & FI_COMPLETION) || !selective)
		wait_cq(1);

	if (o->kind == K_RMA) {
		snprintf(det, dlen, "payload or guard bytes wrong");
		return poll_eq(T + o->toff, NULL, 0, pat, o->len) &&
		       !memcmp(T + o->toff - 4, guard, 4) &&
		       !memcmp(T + o->toff + o->len, guard, 4);
	}
	ok = poll_eq(T, NULL, 0, &o->want, 8);
	if (o->chk_res)
		ok = poll_eq(NULL, &res[m], off, &o->want_res, 8) && ok;
	memcpy(&got, T, 8);
	get_val(&res[m], off, &fetched, 8);
	snprintf(det, dlen, "tgt=%#lx want=%#lx res=%#lx want=%#lx", got,
		 o->want, fetched, o->want_res);
	return ok;
}

static void result(const char *name, enum mem m, enum dmode d, bool ok,
		   const char *det)
{
	printf("%-4s hybrid=%d src=%-4s desc=%-8s %-24s %s\n",
	       ok ? "PASS" : "FAIL", hybrid, mem_s[m], dmode_s[d], name,
	       ok ? "" : det);
	if (ok)
		passes++;
	else
		failures++;
}

/* 64 back-to-back SUMs from consecutive offsets of one MR. */
static void run_sum64(enum mem m, enum dmode d, struct descs ds)
{
	struct op o = ops[0];
	uint64_t v, sum;
	char det[64];

	memset(opt_tgt, 0, 8);
	for (int i = 0; i < 64; i++) {
		v = i + 1;
		put_val(&src[m], SRC_OFF + 8 * i, &v, 8);
	}
	for (int i = 0; i < 64; i++) {
		o.soff = 8 * i;
		CHK((int)post_op(&o, m, ds, false, check_flags(&o)));
		wait_cq(1);
	}
	memcpy(&sum, opt_tgt, 8);
	snprintf(det, sizeof(det), "tgt=%lu want=%d", sum, 64 * 65 / 2);
	result("sum-x64", m, d, sum == 64 * 65 / 2, det);
}

/* check (inject=false) or inject (inject=true) mode for all sources and the
 * desc modes of the current hybrid setting.
 */
static void run_mode(bool inject)
{
	bool dma_env = getenv("FI_CXI_DISABLE_NON_INJECT_AMO_IDC") ||
		       getenv("FI_CXI_DISABLE_NON_INJECT_RMA_IDC");
	enum dmode dms[3];
	int nd = 0;
	char name[64], det[160];

	if (hybrid) {
		dms[nd++] = D_VALID;
		dms[nd++] = D_NULL;
		/* A desc of another MR must fall back to internal mapping.
		 * Only valid where the desc is used for CPU copies only: IDC,
		 * or the FI_INJECT bounce copy.
		 */
		if (inject || !dma_env)
			dms[nd++] = D_OTHER_MR;
	} else {
		dms[nd++] = D_GARBAGE;
		dms[nd++] = D_NULL;
	}

	for (int m = MEM_HOST; m <= MEM_CUDA; m++) {
		for (int di = 0; di < nd; di++) {
			enum dmode d = dms[di];
			struct descs ds = descs_for(m, d);

			if (!inject) {
				run_sum64(m, d, ds);
				for (size_t i = 0; i < NOPS; i++) {
					const struct op *o = &ops[i];

					/* Above 192 B the write can be a DMA,
					 * which uses the desc's MD directly.
					 */
					if (d == D_OTHER_MR && o->len > MAX_INJECT)
						continue;
					result(o->id, m, d,
					       run_one(o, m, ds, false,
						       check_flags(o), det,
						       sizeof(det)), det);
				}
				continue;
			}
			for (int std = 0; std <= 1; std++) {
				for (int comp = 1; comp >= 0; comp--) {
					for (size_t i = 0; i < NOPS; i++) {
						const struct op *o = &ops[i];
						uint64_t fl = FI_INJECT |
							(comp ? FI_COMPLETION : 0);

						if (o->len > MAX_INJECT)
							continue;
						snprintf(name, sizeof(name),
							 "inj-%s-%s-%s", o->id,
							 std ? "DMA" : "IDC",
							 comp ? "comp" : "nocomp");
						result(name, m, d,
						       run_one(o, m, ds, std, fl,
							       det, sizeof(det)),
						       det);
					}
				}
			}
		}
	}
	printf("SUMMARY %s hybrid=%d pass=%d fail=%d\n",
	       inject ? "inject" : "check", hybrid, passes, failures);
}

/* mrcache mode: n posts of one op, one checked run, then a clean close so
 * the MR cache logs "MR cache stats: searches X". The runner compares X for
 * n=0 and n>0; the checked run adds the same count to both.
 */
static int run_mrcache(int argc, char **argv)
{
	const char *id = argv[3];
	bool inject = !strncmp(id, "inj-", 4);
	enum mem m = strcmp(argv[4], "cuda") ? MEM_HOST : MEM_CUDA;
	long n = atol(argv[5]);
	enum dmode d = strcmp(argv[6], "valid") ? D_NULL : D_VALID;
	bool std = argc > 7 && !strcmp(argv[7], "std");
	struct descs ds = descs_for(m, d);
	const struct op *o = NULL;
	char det[160];
	bool ok;

	for (size_t i = 0; i < NOPS; i++)
		if (!strcmp(ops[i].id, inject ? id + 4 : id))
			o = &ops[i];
	if (!o) {
		fprintf(stderr, "unknown op %s\n", id);
		return 1;
	}
	uint64_t fl = inject ? FI_INJECT | FI_COMPLETION : check_flags(o);

	for (long i = 0; i < n; i++) {
		CHK((int)post_op(o, m, ds, std, fl));
		wait_cq(1);
	}
	ok = run_one(o, m, ds, std, fl, det, sizeof(det));
	printf("MRCACHE op=%s key=%s mem=%s n=%ld desc=%s hybrid=%d "
	       "result_ok=%d %s\n", id, std ? "std" : "opt", mem_s[m], n,
	       dmode_s[d], hybrid, ok, ok ? "" : det);

	CHK(fi_close(&opt_mr->fid));
	CHK(fi_close(&std_mr->fid));
	for (int k = MEM_HOST; k <= MEM_CUDA; k++) {
		CHK(fi_close(&src[k].mr->fid));
		CHK(fi_close(&aux[k].mr->fid));
		CHK(fi_close(&res[k].mr->fid));
	}
	CHK(fi_close(&ep_i->fid));
	CHK(fi_close(&ep_t->fid));
	CHK(fi_close(&cq_i->fid));
	CHK(fi_close(&cq_t->fid));
	CHK(fi_close(&av->fid));
	CHK(fi_close(&dom->fid));
	CHK(fi_close(&fab->fid));
	return ok ? 0 : 1;
}

static void setup(void)
{
	struct fi_info *hints = fi_allocinfo();
	struct fi_av_attr av_attr = { .type = FI_AV_TABLE };
	struct fi_cq_attr cq_attr = { .format = FI_CQ_FORMAT_CONTEXT,
				      .size = 1024 };
	struct fi_cxi_dom_ops *dom_ops;
	char addr[64];
	size_t alen = sizeof(addr);

	hints->caps = FI_RMA | FI_ATOMIC | FI_HMEM;
	hints->ep_attr->type = FI_EP_RDM;
	hints->domain_attr->mr_mode = FI_MR_ENDPOINT | FI_MR_ALLOCATED |
				      FI_MR_HMEM;
	hints->domain_attr->threading = FI_THREAD_SAFE;
	hints->fabric_attr->prov_name = strdup("cxi");
	CHK(fi_getinfo(FI_VERSION(1, 20), NULL, NULL, 0, hints, &info));
	fi_freeinfo(hints);

	CHK(fi_fabric(info->fabric_attr, &fab, NULL));
	CHK(fi_domain(fab, info, &dom, NULL));
	/* Hybrid mode must be set before the endpoints are created. */
	CHK(fi_open_ops(&dom->fid, FI_CXI_DOM_OPS_6, 0, (void **)&dom_ops,
			NULL));
	CHK(dom_ops->enable_hybrid_mr_desc(&dom->fid, hybrid));

	CHK(fi_av_open(dom, &av_attr, &av, NULL));
	CHK(fi_cq_open(dom, &cq_attr, &cq_i, NULL));
	CHK(fi_cq_open(dom, &cq_attr, &cq_t, NULL));
	CHK(fi_endpoint(dom, info, &ep_i, NULL));
	CHK(fi_endpoint(dom, info, &ep_t, NULL));
	CHK(fi_ep_bind(ep_i, &av->fid, 0));
	CHK(fi_ep_bind(ep_t, &av->fid, 0));
	/* Without FI_SELECTIVE_COMPLETION, every op gets a CQ event. */
	CHK(fi_ep_bind(ep_i, &cq_i->fid, FI_TRANSMIT | FI_RECV |
		       (selective ? FI_SELECTIVE_COMPLETION : 0)));
	CHK(fi_ep_bind(ep_t, &cq_t->fid, FI_TRANSMIT | FI_RECV));
	CHK(fi_enable(ep_i));
	CHK(fi_enable(ep_t));

	CHK(fi_getname(&ep_t->fid, addr, &alen));
	if (fi_av_insert(av, addr, 1, &tgt_addr, 0, NULL) != 1) {
		fprintf(stderr, "fi_av_insert failed\n");
		exit(2);
	}

	opt_tgt = calloc(1, TGT_LEN);
	std_tgt = calloc(1, TGT_LEN);
	opt_mr = reg(opt_tgt, TGT_LEN, FI_HMEM_SYSTEM,
		     FI_REMOTE_READ | FI_REMOTE_WRITE, OPT_KEY, ep_t);
	std_mr = reg(std_tgt, TGT_LEN, FI_HMEM_SYSTEM,
		     FI_REMOTE_READ | FI_REMOTE_WRITE, STD_KEY, ep_t);

	for (int m = MEM_HOST; m <= MEM_CUDA; m++) {
		region_init(&src[m], m);
		region_init(&aux[m], m);
		region_init(&res[m], m);
	}
	memset(garbage_blob, 0xa5, sizeof(garbage_blob));
}

int main(int argc, char **argv)
{
	const char *mode = argc > 2 ? argv[1] : "";

	if (strcmp(mode, "check") && strcmp(mode, "inject") &&
	    strcmp(mode, "famo-dc") &&
	    (strcmp(mode, "mrcache") || argc < 7)) {
		fprintf(stderr, "usage: see the comment at the top of %s\n",
			__FILE__);
		return 1;
	}
	setvbuf(stdout, NULL, _IOLBF, 0);
	hybrid = atoi(argv[2]);
	selective = !strcmp(mode, "inject");
	CUCHK(cudaSetDevice(0));
	setup();

	if (!strcmp(mode, "mrcache"))
		return run_mrcache(argc, argv);

	if (!strcmp(mode, "famo-dc")) {
		/* Expected to crash in v2.6.0: the fetching_amo_flush path
		 * dereferences a NULL txc->read_cntr.
		 */
		char det[160];
		enum dmode d = hybrid ? D_VALID : D_NULL;

		force_dc = true;
		result("famo-delivery-complete", MEM_HOST, d,
		       run_one(&ops[2], MEM_HOST, descs_for(MEM_HOST, d), false,
			       check_flags(&ops[2]), det, sizeof(det)), det);
		return failures ? 1 : 0;
	}

	run_mode(!strcmp(mode, "inject"));
	return failures ? 1 : 0;
}
