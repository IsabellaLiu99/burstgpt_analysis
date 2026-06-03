"""
BurstGPT Dataset Analysis
=========================
数据来源：BurstGPT (Wang et al., KDD 2025)
GitHub：https://github.com/HPMLL/BurstGPT
原始文件：BurstGPT_1.csv（49MB，1,429,737条真实Azure GPT-3.5/GPT-4请求）

运行方法：
1. 下载原始数据：
   git clone https://github.com/HPMLL/BurstGPT
   或直接下载 data/BurstGPT_1.csv

2. 安装依赖：
   pip install pandas numpy

3. 运行：
   python burstgpt_analysis.py

输出：
   burstgpt_analysis.json（与报告中使用的完全一致）
"""

import pandas as pd
import numpy as np
import json

# ── 1. 载入原始数据 ─────────────────────────────────────────────
print("Loading BurstGPT_1.csv ...")
df = pd.read_csv("BurstGPT_1.csv")

# 原始列：Timestamp, Model, Request tokens, Response tokens, Total tokens, Log Type
# Timestamp 单位是秒，从服务启动开始计时
print(f"  Rows: {len(df):,}")
print(f"  Columns: {df.columns.tolist()}")
print(f"  Duration: {df['Timestamp'].max()/86400:.1f} days")
print(f"  Models: {df['Model'].value_counts().to_dict()}")

# ── 2. 时间分桶 ──────────────────────────────────────────────────
df['minute_bin'] = (df['Timestamp'] // 60).astype(int)   # 每分钟一个桶
df['hour_of_day'] = ((df['Timestamp'] % 86400) // 3600).astype(int)  # 0–23
df['day_bin']    = (df['Timestamp'] // 86400).astype(int)  # 第几天

# ── 3. 每分钟 token 吞吐量（推理计算负载的代理指标）────────────
tpm = df.groupby('minute_bin').agg(
    total_tokens=('Total tokens', 'sum'),
    req_count=('Timestamp', 'count')
).reset_index()
tpm['tps'] = tpm['total_tokens'] / 60.0   # tokens per second

# 归一化到 p99.5 峰值（去除极端突刺）
p99 = tpm['tps'].quantile(0.995)
tpm['norm'] = (tpm['tps'] / p99).clip(0, 1.0)

# ── 4. 日内均值负载曲线（每小时，跨61天取均值）──────────────────
n_days = df['day_bin'].nunique()
hourly = df.groupby('hour_of_day').size().reset_index(name='total_req')
hourly['avg_rph'] = hourly['total_req'] / n_days          # 平均每小时请求数
peak_rph = hourly['avg_rph'].max()
hourly['norm'] = hourly['avg_rph'] / peak_rph              # 归一化

peak_hour   = int(hourly.loc[hourly['avg_rph'].idxmax(), 'hour_of_day'])
valley_hour = int(hourly.loc[hourly['avg_rph'].idxmin(), 'hour_of_day'])
peak_valley_ratio = float(hourly['avg_rph'].max() / hourly['avg_rph'].min())

print(f"\n  Peak hour:   {peak_hour}:00")
print(f"  Valley hour: {valley_hour}:00")
print(f"  Peak/valley ratio: {peak_valley_ratio:.2f}×")

# ── 5. 每日请求量（61天趋势）────────────────────────────────────
daily = df.groupby('day_bin').agg(
    req=('Timestamp', 'count'),
    tok=('Total tokens', 'sum')
).reset_index()

# ── 6. 整体统计 ──────────────────────────────────────────────────
stats = {
    'total_requests':    len(df),
    'duration_days':     round(float(df['Timestamp'].max()) / 86400, 1),
    'mean_tps':          round(float(tpm['tps'].mean()), 1),
    'peak_tps':          round(float(tpm['tps'].max()), 1),
    'p99_tps':           round(float(p99), 1),
    'peak_hour':         peak_hour,
    'valley_hour':       valley_hour,
    'peak_valley_ratio': round(peak_valley_ratio, 2),
    'n_days':            int(n_days),
    'gpt4_fraction_pct': round(float((df['Model'] == 'GPT-4').mean() * 100), 1),
    'mean_tokens_per_req': round(float(df['Total tokens'].mean()), 1),
}

print(f"\nStats: {stats}")

# ── 7. 输出 JSON ─────────────────────────────────────────────────
# 前2天分钟级数据（2880分钟，用于图表）
week1 = tpm[tpm['minute_bin'] < 10080].copy()

output = {
    # 24小时均值负载曲线（24个点）
    'hourly_avg': [
        {'h': int(r.hour_of_day), 'rph': round(float(r.avg_rph), 1), 'norm': round(float(r.norm), 4)}
        for r in hourly.itertuples()
    ],
    # 61天每日请求量（61个点）
    'daily': [
        {'d': int(r.day_bin), 'req': int(r.req), 'tok': int(r.tok)}
        for r in daily.itertuples()
    ],
    # 前2天分钟级token吞吐量（2880个点，用于展示突发性）
    'week1_min': [
        {'t': int(r.minute_bin), 'tps': round(float(r.tps), 1), 'norm': round(float(r.norm), 4)}
        for r in week1.itertuples()
    ],
    # 汇总统计
    'stats': stats,
}

with open('burstgpt_analysis.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved: burstgpt_analysis.json")
print(f"  hourly_avg: {len(output['hourly_avg'])} points")
print(f"  daily:      {len(output['daily'])} points")
print(f"  week1_min:  {len(output['week1_min'])} points")
