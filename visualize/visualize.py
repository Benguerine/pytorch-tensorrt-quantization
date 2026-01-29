import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Set professional academic style
plt.style.use('seaborn-v0_8-paper')
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['figure.titlesize'] = 14
rcParams['lines.linewidth'] = 2
rcParams['lines.markersize'] = 6


benchmark_df = pd.read_csv('../results/benchmark_results.csv', index_col=0)
pytorch_df = pd.read_csv('../results/pytorch_results.csv')


colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']


fig1, axes = plt.subplots(2, 2, figsize=(10, 8))

methods = benchmark_df.index
x_pos = np.arange(len(methods))

# Plot 1: Accuracy
ax1 = axes[0, 0]
accuracy = benchmark_df['Accuracy (%)']
ax1.plot(x_pos, accuracy, marker='o', linewidth=2, markersize=7, 
         color=colors[0], markerfacecolor=colors[0], markeredgecolor='white', markeredgewidth=1.5)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(methods, rotation=30, ha='right')
ax1.set_ylabel('Accuracy (%)', fontweight='bold')
ax1.set_xlabel('Method', fontweight='bold')
ax1.set_title('(a) Model Accuracy', loc='left', fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax1.set_ylim([98.5, 100.5])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Plot 2: Inference Time
ax2 = axes[0, 1]
avg_time = benchmark_df['Avg Time (ms)']
ax2.plot(x_pos, avg_time, marker='s', linewidth=2, markersize=7,
         color=colors[1], markerfacecolor=colors[1], markeredgecolor='white', markeredgewidth=1.5)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(methods, rotation=30, ha='right')
ax2.set_ylabel('Inference Time (ms)', fontweight='bold')
ax2.set_xlabel('Method', fontweight='bold')
ax2.set_title('(b) Average Inference Time', loc='left', fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax2.set_yscale('log')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Plot 3: Speedup
ax3 = axes[1, 0]
speedup = benchmark_df['Speedup']
ax3.plot(x_pos, speedup, marker='^', linewidth=2, markersize=8,
         color=colors[2], markerfacecolor=colors[2], markeredgecolor='white', markeredgewidth=1.5)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(methods, rotation=30, ha='right')
ax3.set_ylabel('Speedup Factor (×)', fontweight='bold')
ax3.set_xlabel('Method', fontweight='bold')
ax3.set_title('(c) Speedup Relative to PyTorch Native', loc='left', fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax3.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Baseline')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Plot 4: Throughput
ax4 = axes[1, 1]
throughput = benchmark_df['Throughput (batch/s)']
ax4.plot(x_pos, throughput, marker='D', linewidth=2, markersize=6,
         color=colors[3], markerfacecolor=colors[3], markeredgecolor='white', markeredgewidth=1.5)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(methods, rotation=30, ha='right')
ax4.set_ylabel('Throughput (batch/s)', fontweight='bold')
ax4.set_xlabel('Method', fontweight='bold')
ax4.set_title('(d) Processing Throughput', loc='left', fontweight='bold')
ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('../visualize/plots/figure1_benchmark_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../visualize/plots/figure1_benchmark_comparison.png', dpi=300, bbox_inches='tight')
print("Figure 1 saved: Benchmark comparison")


fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))

pytorch_methods = pytorch_df['Method']
x_pos_pytorch = np.arange(len(pytorch_methods))

# Plot 1: Latency
ax5 = axes2[0]
latency = pytorch_df['Latency_ms']
ax5.plot(x_pos_pytorch, latency, marker='o', linewidth=2, markersize=8,
         color=colors[0], markerfacecolor=colors[0], markeredgecolor='white', markeredgewidth=1.5)
ax5.set_xticks(x_pos_pytorch)
ax5.set_xticklabels(pytorch_methods, rotation=20, ha='right')
ax5.set_ylabel('Latency (ms)', fontweight='bold')
ax5.set_xlabel('Precision Mode', fontweight='bold')
ax5.set_title('(a) Inference Latency', loc='left', fontweight='bold')
ax5.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

# Add value labels
for i, (x, y) in enumerate(zip(x_pos_pytorch, latency)):
    ax5.annotate(f'{y:.2f}', xy=(x, y), xytext=(0, 8), 
                textcoords='offset points', ha='center', fontsize=9)

# Plot 2: Throughput
ax6 = axes2[1]
pytorch_throughput = pytorch_df['Throughput_img_s']
ax6.plot(x_pos_pytorch, pytorch_throughput, marker='s', linewidth=2, markersize=8,
         color=colors[1], markerfacecolor=colors[1], markeredgecolor='white', markeredgewidth=1.5)
ax6.set_xticks(x_pos_pytorch)
ax6.set_xticklabels(pytorch_methods, rotation=20, ha='right')
ax6.set_ylabel('Throughput (images/s)', fontweight='bold')
ax6.set_xlabel('Precision Mode', fontweight='bold')
ax6.set_title('(b) Processing Throughput', loc='left', fontweight='bold')
ax6.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)

# Add value labels
for i, (x, y) in enumerate(zip(x_pos_pytorch, pytorch_throughput)):
    ax6.annotate(f'{y:.0f}', xy=(x, y), xytext=(0, 8), 
                textcoords='offset points', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('../visualize/plots/figure2_pytorch_precision.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../visualize/plots/figure2_pytorch_precision.png', dpi=300, bbox_inches='tight')
print("Figure 2 saved: PyTorch precision comparison")


fig3, ax = plt.subplots(1, 1, figsize=(10, 6))

# Create multi-line plot with secondary y-axis
ax_twin = ax.twinx()

# Plot throughput on primary axis
line1 = ax.plot(x_pos, throughput, marker='o', linewidth=2.5, markersize=8,
                color=colors[0], markerfacecolor=colors[0], markeredgecolor='white', 
                markeredgewidth=1.5, label='Throughput', zorder=3)

# Plot inference time on secondary axis
line2 = ax_twin.plot(x_pos, avg_time, marker='s', linewidth=2.5, markersize=8,
                     color=colors[3], markerfacecolor=colors[3], markeredgecolor='white',
                     markeredgewidth=1.5, label='Inference Time', zorder=3)

# Formatting
ax.set_xticks(x_pos)
ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=10)
ax.set_ylabel('Throughput (batch/s)', fontweight='bold', color=colors[0], fontsize=11)
ax.set_xlabel('Optimization Method', fontweight='bold', fontsize=11)
ax_twin.set_ylabel('Inference Time (ms)', fontweight='bold', color=colors[3], fontsize=11)
ax.set_title('Performance Comparison: Throughput vs. Inference Time', 
             fontweight='bold', fontsize=13, pad=15)

# Color the y-axis labels
ax.tick_params(axis='y', labelcolor=colors[0])
ax_twin.tick_params(axis='y', labelcolor=colors[3])

# Grid and spines
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
ax.spines['top'].set_visible(False)
ax_twin.spines['top'].set_visible(False)

# Combined legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='upper left', frameon=True, fancybox=True, 
          shadow=True, framealpha=0.95)

plt.tight_layout()
plt.savefig('../visualize/plots/figure3_performance_overview.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../visualize/plots/figure3_performance_overview.png', dpi=300, bbox_inches='tight')
print("Figure 3 saved: Performance overview")


fig4, ax = plt.subplots(1, 1, figsize=(10, 6))

avg_time_vals = benchmark_df['Avg Time (ms)']
std_time_vals = benchmark_df['Std Time (ms)']

# Plot with error bars
ax.errorbar(x_pos, avg_time_vals, yerr=std_time_vals, 
            marker='o', linewidth=2, markersize=8, capsize=5, capthick=2,
            color=colors[1], markerfacecolor=colors[1], markeredgecolor='white',
            markeredgewidth=1.5, ecolor=colors[1], elinewidth=2, alpha=0.8)

ax.set_xticks(x_pos)
ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=10)
ax.set_ylabel('Inference Time (ms)', fontweight='bold', fontsize=11)
ax.set_xlabel('Optimization Method', fontweight='bold', fontsize=11)
ax.set_title('Inference Time with Standard Deviation', 
             fontweight='bold', fontsize=13, pad=15)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_yscale('log')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('../visualize/plots/figure4_error_analysis.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../visualize/plots/figure4_error_analysis.png', dpi=300, bbox_inches='tight')
print("Figure 4 saved: Error analysis")

print("\n✓ All figures generated successfully!")
print("  - PDF format: suitable for LaTeX/academic publications")
print("  - PNG format: suitable for presentations and web")