import torch
import torch.distributed as dist
import deep_ep
from deep_ep import ElasticBuffer
from deep_ep.utils.envs import init_dist
from deep_ep.utils.testing import bench
from deep_ep.utils.math import per_token_cast_to_fp8

NUM_RUNS = 5


def test(local_rank, num_local_ranks):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    for num_tokens in [1, 2, 4, 8, 16, 32, 64, 128]:
        hidden, num_topk, num_experts = 7168, 8, 256

        results = {
            'll_dispatch': [], 'll_combine': [],
            'v2_dispatch': [], 'v2_combine': [],
        }

        for run in range(NUM_RUNS):
            # ============================================================
            # Legacy Low-Latency Mode
            # ============================================================
            num_local_experts = num_experts // num_ranks
            rdma_size = deep_ep.Buffer.get_low_latency_rdma_size_hint(
                num_tokens, hidden, num_ranks, num_experts)
            buffer = deep_ep.Buffer(group, int(1e9), rdma_size,
                                    low_latency_mode=True,
                                    num_qps_per_rank=num_local_experts,
                                    explicitly_destroy=True)

            x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
            scores = torch.randn((num_tokens, num_experts), dtype=torch.float32,
                                 device='cuda').abs() + 1
            topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True,
                                  sorted=True)[1].to(deep_ep.topk_idx_t)
            topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32,
                                       device='cuda').abs()

            # Warmup
            packed_recv_x, packed_recv_count, handle, event, hook = \
                buffer.low_latency_dispatch(x, topk_idx, num_tokens, num_experts,
                                            use_fp8=False, async_finish=False)
            event.current_stream_wait()
            simulated_gemm_x = packed_recv_x.clone()
            buffer.low_latency_combine(simulated_gemm_x, topk_idx, topk_weights, handle,
                                       async_finish=False)

            # Bench dispatch
            def ll_dispatch():
                return buffer.low_latency_dispatch(x, topk_idx, num_tokens, num_experts,
                                                   use_fp8=False, async_finish=False)
            t_ll_d = bench(ll_dispatch)[0]

            # Bench combine
            packed_recv_x2, _, handle2, event2, _ = ll_dispatch()
            event2.current_stream_wait()
            gemm_out = packed_recv_x2.clone()

            def ll_combine():
                return buffer.low_latency_combine(gemm_out, topk_idx, topk_weights, handle2,
                                                  async_finish=False)
            t_ll_c = bench(ll_combine)[0]

            results['ll_dispatch'].append(t_ll_d)
            results['ll_combine'].append(t_ll_c)

            buffer.destroy()
            dist.barrier()

            # ============================================================
            # Elastic V2 (decode: do_cpu_sync=False, do_expand=True)
            # ============================================================
            ebuf = ElasticBuffer(group=group,
                                 num_max_tokens_per_rank=num_tokens,
                                 hidden=hidden, num_topk=num_topk,
                                 use_fp8_dispatch=False,
                                 allow_hybrid_mode=False,
                                 explicitly_destroy=True)

            x2 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
            scores2 = torch.randn((num_tokens, num_experts), dtype=torch.float32,
                                  device='cuda').abs() + 1
            topk_idx2 = torch.topk(scores2, num_topk, dim=-1, largest=True,
                                   sorted=False)[1].to(deep_ep.topk_idx_t)
            topk_weights2 = torch.randn((num_tokens, num_topk), dtype=torch.float32,
                                        device='cuda')

            # Warmup
            recv_x2, _, _, handle_v2, _ = ebuf.dispatch(
                x=x2, topk_idx=topk_idx2, topk_weights=topk_weights2,
                num_experts=num_experts, num_sms=24,
                do_expand=True, do_cpu_sync=False)
            recv_bf16 = recv_x2.to(torch.bfloat16) if not isinstance(recv_x2, tuple) \
                else recv_x2[0].to(torch.bfloat16)
            ebuf.combine(x=recv_bf16, handle=handle_v2, topk_weights=topk_weights2, num_sms=24)

            # Bench dispatch
            def v2_dispatch():
                return ebuf.dispatch(
                    x=x2, topk_idx=topk_idx2, topk_weights=topk_weights2,
                    num_experts=num_experts, num_sms=24,
                    do_expand=True, do_cpu_sync=False)
            t_v2_d = bench(v2_dispatch)[0]

            # Bench combine
            r2, _, _, h2, _ = v2_dispatch()
            r2bf = r2.to(torch.bfloat16) if not isinstance(r2, tuple) \
                else r2[0].to(torch.bfloat16)

            def v2_combine():
                return ebuf.combine(x=r2bf, handle=h2, topk_weights=topk_weights2, num_sms=24)
            t_v2_c = bench(v2_combine)[0]

            results['v2_dispatch'].append(t_v2_d)
            results['v2_combine'].append(t_v2_c)

            ebuf.destroy()
            dist.barrier()

        if local_rank == 0:
            us = lambda t: f'{t*1e6:.0f}'
            avg = lambda lst: sum(lst) / len(lst)

            a_ll_d = avg(results['ll_dispatch'])
            a_ll_c = avg(results['ll_combine'])
            a_v2_d = avg(results['v2_dispatch'])
            a_v2_c = avg(results['v2_combine'])

            a_ll_total = a_ll_d + a_ll_c
            a_v2_total = a_v2_d + a_v2_c

            dispatch_diff = (a_ll_d / a_v2_d - 1) * 100 if a_v2_d > 0 else 0
            combine_diff = (a_ll_c / a_v2_c - 1) * 100 if a_v2_c > 0 else 0
            total_diff = (a_ll_total / a_v2_total - 1) * 100 if a_v2_total > 0 else 0

            print(flush=True)
            print(f'=== decode tokens={num_tokens}  hidden={hidden}  topk={num_topk}  experts={num_experts}  ({NUM_RUNS} runs avg) ===', flush=True)
            print(f'  Legacy LL dispatch  : {us(a_ll_d)} us', flush=True)
            print(f'  Legacy LL combine   : {us(a_ll_c)} us', flush=True)
            print(f'  Legacy LL total     : {us(a_ll_total)} us', flush=True)
            print(f'  V2 dispatch         : {us(a_v2_d)} us', flush=True)
            print(f'  V2 combine          : {us(a_v2_c)} us', flush=True)
            print(f'  V2 total            : {us(a_v2_total)} us', flush=True)
            print(f'  Dispatch LL vs V2   : {dispatch_diff:+.1f}%', flush=True)
            print(f'  Combine  LL vs V2   : {combine_diff:+.1f}%', flush=True)
            print(f'  Total    LL vs V2   : {total_diff:+.1f}%', flush=True)

            print(f'  --- per-run details ---', flush=True)
            for i in range(NUM_RUNS):
                ll_d = results['ll_dispatch'][i]
                ll_c = results['ll_combine'][i]
                v2_d = results['v2_dispatch'][i]
                v2_c = results['v2_combine'][i]
                print(f'  run{i+1}: LL_d={us(ll_d)} LL_c={us(ll_c)} V2_d={us(v2_d)} V2_c={us(v2_c)} '
                      f'LL_tot={us(ll_d+ll_c)} V2_tot={us(v2_d+v2_c)}', flush=True)

    dist.destroy_process_group()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-local-ranks', type=int, default=8)
    args = parser.parse_args()
    torch.multiprocessing.spawn(test, args=(args.num_local_ranks,), nprocs=args.num_local_ranks)
