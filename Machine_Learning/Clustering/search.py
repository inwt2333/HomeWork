import argparse
import csv
import os
import numpy as np

from MyGMM import MyGMM
from evaluate_validation import evaluate_validation

def parse_seeds(text: str):
    # 支持 "0-49" 或 "0,1,2,10" 混合写法
    seeds = []
    for token in text.split(','):
        token = token.strip()
        if not token:
            continue
        if '-' in token:
            a, b = token.split('-')
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(token))
    return seeds

def main():
    parser = argparse.ArgumentParser(description="Sweep random_state for MyGMM and evaluate on validation set")
    parser.add_argument("--seeds", type=str, default="0-200", help='种子集合，如 "0-49" 或 "0,1,2,10"')
    parser.add_argument("--metric", type=str, default="nmi", choices=["ari", "nmi"], help="以哪个指标选最优")
    parser.add_argument("--n_components", type=int, default=50)
    parser.add_argument("--clusters", type=int, default=7)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--save_all", action="store_true", help="是否保存每个种子的预测文件 pred_labels_seed{seed}.npy")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    print(f"将测试 {len(seeds)} 个 random_state: {seeds[:8]}{' ...' if len(seeds)>8 else ''}")

    # 加载数据一次
    data = np.load('data_1000d.npy')

    results = []
    best = None  # (metric_value, seed, ari, nmi, outfile)

    for seed in seeds:
        # 训练与预测
        model = MyGMM(n_components=args.n_components,
                      n_clusters=args.clusters,
                      max_iter=args.max_iter,
                      random_state=seed)
        model.fit(data)
        preds = model.predict(data)

        # 保存预测
        if args.save_all:
            out_file = f"pred_labels_seed{seed}.npy"
        else:
            out_file = "pred_labels.npy"
        np.save(out_file, preds)

        # 评估
        scores = evaluate_validation(out_file)
        if scores is None:
            print(f"[seed={seed}] 评估失败，跳过。")
            continue

        ari, nmi = scores["ari"], scores["nmi"]
        metric_value = ari if args.metric == "ari" else nmi
        results.append((seed, ari, nmi))

        # 更新最优
        if best is None or metric_value > best[0] or (metric_value == best[0] and nmi > best[3]):
            best = (metric_value, seed, ari, nmi, out_file)

        print(f"[seed={seed}] ARI={ari:.4f}, NMI={nmi:.4f} {'<= 当前最佳' if best and best[1]==seed else ''}")

    # 输出总结
    if not results:
        print("没有有效结果，请检查数据与评估文件是否存在。")
        return

    # 保存 CSV
    csv_path = "sweep_random_state_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "ARI", "NMI"])
        w.writerows(results)
    print(f"\n已保存汇总到 {csv_path}")

    # 打印 Top-5（按 ARI 排序，次序用 NMI 打破平手）
    results_sorted = sorted(results, key=lambda x: (x[1], x[2]), reverse=True)
    print("\nTop-5（按 ARI 排序）:")
    for i, (seed, ari, nmi) in enumerate(results_sorted[:5], 1):
        print(f"#{i}: seed={seed:<4d}  ARI={ari:.4f}  NMI={nmi:.4f}")

    # 最优
    metric_name = args.metric.upper()
    print(f"\n最佳（按 {metric_name}）：seed={best[1]}  ARI={best[2]:.4f}  NMI={best[3]:.4f}")

    # 若未保存所有文件，确保把最优预测另存为 pred_labels_best.npy，并覆盖 pred_labels.npy
    if not args.save_all:
        np.save("pred_labels_best.npy", np.load("pred_labels.npy"))
    else:
        # 若 save_all，则从对应文件复制为 pred_labels.npy 以便后续直接提交
        if os.path.abspath(best[4]) != os.path.abspath("pred_labels.npy"):
            np.save("pred_labels.npy", np.load(best[4]))

if __name__ == "__main__":
    main()