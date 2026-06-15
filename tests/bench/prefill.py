#!/opt/conda310/bin/python

import warnings
warnings.filterwarnings('ignore')
import torch, torch.distributed as dist, deep_ep
from deep_ep import ElasticBuffer
from deep_ep.utils.envs import init_dist
from deep_ep.utils.testing import bench
from deep_ep.utils.math import per_token_cast_to_fp8
from deep_ep.utils.refs import generate_pre_combine_data, ordered_accumulate, combine as ref_combine
import os

os.environ['EP_DISABLE_GIN'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

NUM_RUNS = 5
HIDDEN = 2048
NUM_TOPK = 8
NUM_EXPERTS = 256
TOKEN_SIZES = [256, 32768]

# 全局输入缓存：相同参数只生成一次，保证 bench 和 verify 使用同一份数据
_INPUT_CACHE = {}

def generate_inputs(num_tokens, hidden, num_topk, num_experts):
    """生成并缓存输入数据，同一份数据在 bench 和 verify 间共享。"""
    key = (num_tokens, hidden, num_topk, num_experts)
    if key not in _INPUT_CACHE:
        # 固定 seed 保证多进程（multiprocessing.spawn）各 rank 随机数一致
        torch.manual_seed(42)
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        x_fp8 = per_token_cast_to_fp8(x)
        x_fp8 = (x_fp8[0], x_fp8[1].T.contiguous().T)
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
        topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True,
                              sorted=False)[1].to(deep_ep.topk_idx_t)
        topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')
        _INPUT_CACHE[key] = (x_fp8, topk_idx, topk_weights)
    return _INPUT_CACHE[key]


def bench_legacy(group, num_ranks, num_tokens, hidden, num_topk, num_experts):
    """Legacy Normal dispatch + combine benchmark."""
    deep_ep.Buffer.set_num_sms(24)
    buffer = deep_ep.Buffer(group, int(2e9), 0, low_latency_mode=False,
                            num_qps_per_rank=1, explicitly_destroy=True)

    x_fp8, topk_idx, topk_weights = generate_inputs(num_tokens, hidden, num_topk, num_experts)

    num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)
    t_layout = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]

    default_config = deep_ep.Buffer.get_dispatch_config(num_ranks)
    dispatch_args = dict(x=x_fp8, num_tokens_per_rank=num_tokens_per_rank,
                         is_token_in_rank=is_token_in_rank,
                         num_tokens_per_expert=num_tokens_per_expert,
                         topk_idx=topk_idx, topk_weights=topk_weights,
                         config=default_config)

    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)
    t_no_handle = bench(lambda: buffer.dispatch(**dispatch_args))[0]

    combine_config = deep_ep.Buffer.get_combine_config(num_ranks)
    recv_bf16 = recv_x[0].to(torch.bfloat16) if isinstance(recv_x, tuple) \
        else recv_x.to(torch.bfloat16)
    t_combine = bench(lambda: buffer.combine(x=recv_bf16, handle=handle,
                                             config=combine_config))[0]

    buffer.destroy()
    return t_layout, t_no_handle, t_combine


def bench_elastic_no_expand(group, num_tokens, hidden, num_topk, num_experts):
    """Elastic dispatch (do_expand=False, do_cpu_sync=True) + combine benchmark."""
    ebuf = ElasticBuffer(group=group, num_max_tokens_per_rank=num_tokens,
                         hidden=hidden, num_topk=num_topk,
                         use_fp8_dispatch=True, allow_hybrid_mode=False,
                         explicitly_destroy=True)

    x_fp8, topk_idx, topk_weights = generate_inputs(num_tokens, hidden, num_topk, num_experts)

    def dispatch():
        return ebuf.dispatch(x=x_fp8, topk_idx=topk_idx, topk_weights=topk_weights,
                             num_experts=num_experts, num_sms=24,
                             do_expand=False, do_cpu_sync=True)

    t_dispatch = bench(dispatch)[0]

    recv, _, _, handle, _ = dispatch()
    recv_bf16 = recv[0].to(torch.bfloat16) if isinstance(recv, tuple) else recv.to(torch.bfloat16)
    t_combine = bench(lambda: ebuf.combine(x=recv_bf16, handle=handle,
                                           topk_weights=None, num_sms=24))[0]

    ebuf.destroy()
    return t_dispatch, t_combine


def bench_elastic_expand(group, num_tokens, hidden, num_topk, num_experts):
    """Elastic dispatch (do_expand=True, do_cpu_sync=True) + combine benchmark."""
    ebuf = ElasticBuffer(group=group, num_max_tokens_per_rank=num_tokens,
                         hidden=hidden, num_topk=num_topk,
                         use_fp8_dispatch=True, allow_hybrid_mode=False,
                         explicitly_destroy=True)

    x_fp8, topk_idx, topk_weights = generate_inputs(num_tokens, hidden, num_topk, num_experts)

    def dispatch():
        return ebuf.dispatch(x=x_fp8, topk_idx=topk_idx, topk_weights=topk_weights,
                             num_experts=num_experts, num_sms=24,
                             do_expand=True, do_cpu_sync=True)

    t_dispatch = bench(dispatch)[0]

    recv, _, _, handle, _ = dispatch()
    recv_bf16 = recv[0].to(torch.bfloat16) if isinstance(recv, tuple) else recv.to(torch.bfloat16)
    t_combine = bench(lambda: ebuf.combine(x=recv_bf16, handle=handle,
                                           topk_weights=None, num_sms=24))[0]

    ebuf.destroy()
    return t_dispatch, t_combine


def verify_correctness(group, num_ranks, num_tokens, hidden, num_topk, num_experts, local_rank):
    """参考官方 refs 构造 combine 输入，验证 Elastic 各模式 combine 输出一致性。"""
    x_fp8, topk_idx, topk_weights = generate_inputs(num_tokens, hidden, num_topk, num_experts)

    rank = dist.get_rank()
    ref_y = generate_pre_combine_data(
        rank * num_tokens + torch.arange(num_tokens, device='cuda'),
        num_tokens, num_topk, hidden)
    ref_y[topk_idx == -1] = 0
    ref_out = ref_combine(
        ref_y, topk_idx,
        1, num_ranks, num_experts,
        None,
        True, False)

    def primary_rows(tensor_or_tuple):
        return tensor_or_tuple[0].shape[0] if isinstance(tensor_or_tuple, tuple) else tensor_or_tuple.shape[0]

    # --- Elastic no-expand ---
    ebuf = ElasticBuffer(group=group, num_max_tokens_per_rank=num_tokens,
                         hidden=hidden, num_topk=num_topk,
                         use_fp8_dispatch=True, allow_hybrid_mode=False,
                         explicitly_destroy=True)
    recv, recv_topk_idx, recv_topk_weights, handle, _ = ebuf.dispatch(
        x=x_fp8, topk_idx=topk_idx, topk_weights=topk_weights,
        num_experts=num_experts, num_sms=24,
        do_expand=False, do_cpu_sync=True)
    num_recv_tokens = handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
    src_token_global_idx = handle.recv_src_metadata[:num_recv_tokens, 0]
    local_y = generate_pre_combine_data(src_token_global_idx, num_tokens, num_topk, hidden)
    local_y[recv_topk_idx[:num_recv_tokens] == -1] = 0
    input_for_combine = torch.empty((primary_rows(recv), hidden), dtype=torch.bfloat16, device='cuda')
    input_for_combine[:num_recv_tokens] = ordered_accumulate(local_y)
    no_expand_out, _, _ = ebuf.combine(
        x=input_for_combine, handle=handle, topk_weights=recv_topk_weights, num_sms=24)
    ebuf.destroy()

    # --- Elastic expand ---
    ebuf = ElasticBuffer(group=group, num_max_tokens_per_rank=num_tokens,
                         hidden=hidden, num_topk=num_topk,
                         use_fp8_dispatch=True, allow_hybrid_mode=False,
                         explicitly_destroy=True)
    recv, _, _, handle, _ = ebuf.dispatch(
        x=x_fp8, topk_idx=topk_idx, topk_weights=topk_weights,
        num_experts=num_experts, num_sms=24,
        do_expand=True, do_cpu_sync=True)
    num_recv_tokens = handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
    src_token_global_idx = handle.recv_src_metadata[:num_recv_tokens, 0]
    local_y = generate_pre_combine_data(src_token_global_idx, num_tokens, num_topk, hidden)
    input_for_expand_combine = torch.empty((primary_rows(recv) + 1, hidden), dtype=torch.bfloat16, device='cuda')
    input_for_expand_combine[handle.recv_src_metadata[:num_recv_tokens, 2:].flatten()] = local_y.view(-1, hidden)
    input_for_expand_combine = input_for_expand_combine[:-1, ...]
    expand_out, _, _ = ebuf.combine(x=input_for_expand_combine, handle=handle, topk_weights=None, num_sms=24)
    ebuf.destroy()

    # --- 对比（fp8 有精度损失，用较宽松的 atol） ---
    ok_no_expand = torch.allclose(ref_out.float(), no_expand_out.float(), atol=0.1, rtol=0.0)
    ok_expand    = torch.allclose(ref_out.float(), expand_out.float(),    atol=0.1, rtol=0.0)

    if local_rank == 0:
        diff_no_expand = (ref_out.float() - no_expand_out.float()).abs().max().item()
        diff_expand = (ref_out.float() - expand_out.float()).abs().max().item()
        status_no_expand = 'PASS' if ok_no_expand else 'FAIL'
        status_expand    = 'PASS' if ok_expand    else 'FAIL'
        print(f'[verify tokens={num_tokens}] '
              f'E-no-expand: {status_no_expand} (max_diff={diff_no_expand:.6f}) | '
              f'E-expand: {status_expand} (max_diff={diff_expand:.6f})',
              flush=True)


def test(local_rank, num_local_ranks):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    all_results = []

    for num_tokens in TOKEN_SIZES:
        hidden, num_topk, num_experts = HIDDEN, NUM_TOPK, NUM_EXPERTS

        # 所有 rank 都参与 verify（dispatch/combine 是分布式操作），rank 0 负责打印
        verify_correctness(group, num_ranks, num_tokens, hidden, num_topk, num_experts, local_rank)
        dist.barrier()
        l_layouts, l_dispatches, l_combines = [], [], []
        e_dispatches, e_combines = [], []
        e_dispatches_exp, e_combines_exp = [], []

        for run in range(NUM_RUNS):
            t_layout, t_dispatch, t_combine = bench_legacy(
                group, num_ranks, num_tokens, hidden, num_topk, num_experts)
            l_layouts.append(t_layout)
            l_dispatches.append(t_dispatch)
            l_combines.append(t_combine)
            dist.barrier()

            t_e_dispatch, t_e_combine = bench_elastic_no_expand(
                group, num_tokens, hidden, num_topk, num_experts)
            e_dispatches.append(t_e_dispatch)
            e_combines.append(t_e_combine)
            dist.barrier()

            t_e_dispatch_exp, t_e_combine_exp = bench_elastic_expand(
                group, num_tokens, hidden, num_topk, num_experts)
            e_dispatches_exp.append(t_e_dispatch_exp)
            e_combines_exp.append(t_e_combine_exp)
            dist.barrier()

        if local_rank == 0:
            avg = lambda lst: sum(lst) / len(lst)
            a_l_layout       = avg(l_layouts)
            a_l_dispatch     = avg(l_dispatches)
            a_l_combine      = avg(l_combines)
            a_e_dispatch     = avg(e_dispatches)
            a_e_combine      = avg(e_combines)
            a_e_dispatch_exp = avg(e_dispatches_exp)
            a_e_combine_exp  = avg(e_combines_exp)

            a_l_total     = a_l_layout + a_l_dispatch + a_l_combine
            a_e_total     = a_e_dispatch + a_e_combine
            a_e_total_exp = a_e_dispatch_exp + a_e_combine_exp

            dispatch_boost     = (a_l_layout + a_l_dispatch) / a_e_dispatch - 1
            dispatch_exp_boost = (a_l_layout + a_l_dispatch) / a_e_dispatch_exp - 1
            combine_boost      = a_l_combine / a_e_combine - 1
            combine_exp_boost  = a_l_combine / a_e_combine_exp - 1
            total_boost        = a_l_total / a_e_total - 1
            total_exp_boost    = a_l_total / a_e_total_exp - 1

            all_results.append({
                'tokens': num_tokens,
                'l_layout': a_l_layout, 'l_dispatch': a_l_dispatch, 'l_combine': a_l_combine,
                'l_total': a_l_total,
                'e_dispatch': a_e_dispatch, 'e_combine': a_e_combine, 'e_total': a_e_total,
                'e_dispatch_exp': a_e_dispatch_exp, 'e_combine_exp': a_e_combine_exp,
                'e_total_exp': a_e_total_exp,
                'dispatch_boost': dispatch_boost, 'dispatch_exp_boost': dispatch_exp_boost,
                'combine_boost': combine_boost, 'combine_exp_boost': combine_exp_boost,
                'total_boost': total_boost, 'total_exp_boost': total_exp_boost,
            })

    if local_rank == 0 and all_results:
        def fmt_us(t):
            return f'{int(t*1e6):>5}us'
        def fmt_pct(v):
            return f'{v*100:>+6.1f}%'
        col_w = 11
        cols = ['mode', 'dispatch', 'combine', 'total', 'D-boost', 'C-boost', 'Total-boost']
        header = ' | '.join(f'{c:>{col_w}}' for c in cols)
        sep = '-' * len(header)
        print(flush=True)
        for r in all_results:
            print(f'=== tokens={r["tokens"]} ===', flush=True)
            print(header, flush=True)
            print(sep, flush=True)

            # Legacy 行：dispatch = layout + dispatch，因为 layout 是 Legacy 独有开销
            l_dispatch_total = r['l_layout'] + r['l_dispatch']
            rows = [
                ('Legacy',
                 fmt_us(l_dispatch_total), fmt_us(r['l_combine']), fmt_us(r['l_total']),
                 '      -    ', '      -    ', '      -    '),
                ('E-no-expand',
                 fmt_us(r['e_dispatch']), fmt_us(r['e_combine']), fmt_us(r['e_total']),
                 fmt_pct(r['dispatch_boost']), fmt_pct(r['combine_boost']),
                 fmt_pct(r['total_boost'])),
                ('E-expand',
                 fmt_us(r['e_dispatch_exp']), fmt_us(r['e_combine_exp']), fmt_us(r['e_total_exp']),
                 fmt_pct(r['dispatch_exp_boost']), fmt_pct(r['combine_exp_boost']),
                 fmt_pct(r['total_exp_boost'])),
            ]
            for mode, disp, comb, total, d_boost, c_boost, t_boost in rows:
                print(f'{mode:>{col_w}} | {disp:>{col_w}} | {comb:>{col_w}} | {total:>{col_w}} | {d_boost:>{col_w}} | {c_boost:>{col_w}} | {t_boost:>{col_w}}', flush=True)
            print(flush=True)

    dist.destroy_process_group()


if __name__ == '__main__':
    num_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    torch.multiprocessing.spawn(test, args=(num_gpus,), nprocs=num_gpus)
