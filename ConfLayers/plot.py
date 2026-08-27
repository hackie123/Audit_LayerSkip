import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def baseline_find_time(file_name, i, nb_samples, baseline):
    start_time = 0
    periods = []
    instances = 0
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            if instances == 2*i and '/'+str(nb_samples) in line and '00:00<' not in line:
                time = line.split('/'+str(nb_samples)+' [')[1].split('<')[0].split(':')
                current_time=float(time[0])*60+float(time[1])
                baseline.append(current_time)
                period = current_time - start_time
                periods.append(period)
                start_time=current_time
                # print("current time {}".format(current_time))
            if "Loading checkpoint shards: 100" in line:
                instances+=1
        mean_period = np.mean(periods[1:-1])
        print(periods)
        print("mean {}".format(mean_period))
    return baseline, mean_period

def find_time(file_name, i, nb_samples, matches, baseline):
    end = False
    start_time = 0
    end_time = 0
    periods = []
    inf_end = 0
    instances = 0
    j=0
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            if instances == 2*i:
                if end and '/'+str(nb_samples) in line:
                    time = line.split('/'+str(nb_samples)+' [')[1].split('<')[0].split(':')
                    end_time=float(time[0])*60+float(time[1])
                    start_time=float(time[0])*60+float(time[1])
                    end = False
                    print("start time {}".format(start_time))
                    inf_end = int(line.split('/'+str(nb_samples))[0].split('|')[2])
                    # print("inf_end {}".format(inf_end))
                if end_time > 0 and '/'+str(nb_samples) in line:
                    time = line.split('/'+str(nb_samples)+' [')[1].split('<')[0].split(':')
                    current_time=float(time[0])*60+float(time[1])
                    period = current_time - start_time
                    periods.append(period)
                    
                    matches.append(baseline[j]/current_time)
                    if len(matches)!=0:
                        print("baseline {}".format(baseline[j]))
                        print("current_time {}".format(current_time))
                        print("match {}".format(matches[-1]))
                    j+=1
                    start_time=current_time
                elif end_time==0 and '/'+str(nb_samples) in line and '00:00' not in line:
                    time = line.split('/'+str(nb_samples)+' [')[1].split('<')[0].split(':')
                    current_time=float(time[0])*60+float(time[1])
                    
                    matches.append(baseline[j]/current_time)
                    print("baseline {}".format(baseline[j]))
                    print("current_time {}".format(current_time))
                    print("match {}".format(matches[-1]))
                    j+=1
                if "Stopped" in line:
                    end=True
                    # print("Stopped")
            if "Loading checkpoint shards: 100" in line:
                instances+=1
        mean_period = np.mean(periods[1:-1])
        print(periods)
        print("mean period {}".format(mean_period))
    return matches, mean_period, inf_end, end_time


def set_bottom_title(ax, title, pad=0.15):
    ax.text(0.5, -pad, title,
            transform=ax.transAxes,
            ha='center', va='top')


# PLOT SPEEDUP VS LAMBDA
print("SPEEDUP VS LAMBDA")
plt.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
nb_samples = 50
total_samples = 100

lam = [-0.4, -0.2, 0, 0.2, 0.4]
speedups = [1.088, 1.08, 1.12, 1.165, 1.139]
skip = [0.579, 0.55, 0.5, 0.447, 0.421] 

axes[0].plot(lam, speedups, marker='o', color='royalblue', linestyle='-', label='Speedup')
axes[0].set_title("(a) LLaMa-2-13B")
axes[0].set_ylabel("Speedup", color='royalblue')
axes[0].tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=False)
axes[0].grid()

ax2_0 = axes[0].twinx()
ax2_0.plot(lam, skip, marker='s', color='darkorange', linestyle='--', label='Accuracy')
ax2_0.set_ylabel("Skip Ratio", color='darkorange')

speedups = [0.991, 1.096, 1.17, 1.190, 1.164]
skip = [0.633, 0.6, 0.466, 0.433, 0.433] 

axes[1].plot(lam, speedups, marker='o', color='royalblue', linestyle='-', label='Speedup')
axes[1].set_title("(b) LLaMa-3-8B")
axes[1].set_xlabel("Lambda")
axes[1].set_ylabel("Speedup", color='royalblue')
axes[1].grid()

ax2_1 = axes[1].twinx()
ax2_1.plot(lam, skip, marker='s', color='darkorange', linestyle='--', label='Accuracy')
ax2_1.set_ylabel("Skip Ratio", color='darkorange')

fig.subplots_adjust(bottom=0.2)
plt.subplots_adjust(hspace=0.5)
# plt.xlabel("Lambda")
# plt.ylabel("Speedup")
plt.savefig("plots/speedup-vs-lam.pdf")


#PLOT SWIFT VS CL
print("SWIFT VS CL")
total_samples = 100
plt.rcParams.update({'font.size': 32})
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10))
baseline=[]
swift=[]
cl=[]

baseline, baseline_mean_period = baseline_find_time("logs/all_llama2-13b_num-100_new-512_data-cnndm.out", 1, total_samples, baseline)
swift, mean_period, inf_end1, end_time = find_time("logs/all_llama2-13b_num-100_new-512_data-cnndm.out", 2, total_samples, swift, baseline)
cl, mean_period, inf_end2, end_time = find_time("logs/conflayers_llama2-13b_num-100_new-512_data-cnndm.out", 1, total_samples, cl, baseline)

axes[0].plot(swift[:50], color='purple', marker='o', label='SWIFT')
axes[0].plot(cl[:50], color='red', marker='o', label='ConfLayers')
axes[0].axvline(inf_end1-1, color='purple', linestyle='--', label='SWIFT Search End', linewidth=4)
axes[0].axvline(inf_end2-1, color='red', linestyle='--', label='ConfLayers Search End', linewidth=4)
axes[0].set_title("(a) CNN-DM")
axes[0].set_ylabel("Speedup")
axes[0].tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=False)
axes[0].grid()

baseline=[]
swift=[]
cl=[]

baseline, baseline_mean_period = baseline_find_time("logs/all_llama2-13b_num-100_new-512_data-gsm8k.out", 1, total_samples, baseline)
swift, mean_period, inf_end1, end_time = find_time("logs/all_llama2-13b_num-100_new-512_data-gsm8k.out", 2, total_samples, swift, baseline)
cl, mean_period, inf_end2, end_time = find_time("logs/conflayers_llama2-13b_num-100_new-512_data-gsm8k.out", 1, total_samples, cl, baseline)

axes[1].plot(swift[:50], color='purple', marker='o', label='SWIFT')
axes[1].plot(cl[:50], color='red', marker='o', label='ConfLayers')
axes[1].axvline(inf_end1-1, color='purple', linestyle='--', linewidth=4)
axes[1].axvline(inf_end2-1, color='red', linestyle='--', linewidth=4)
axes[1].set_title("(b) GSM8K")
axes[1].set_ylabel("Speedup")
# axes[1].tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=False)
axes[1].grid()


h, l = axes[0].get_legend_handles_labels()
axes[0].legend(loc='upper center',        # position relative to bbox
    bbox_to_anchor=(0.5, 1.8),# x=center (0.5), y=slightly above plot (1.15)
    ncol=2,                    # number of legend columns (optional)
    frameon=False,              # remove box border for cleaner look
    fontsize=28
)
# axes[0].set_label()
axes[1].set_xlabel("Input Prompt Index")
fig.subplots_adjust(top=0.8, bottom=0.15)
plt.subplots_adjust(hspace=0.3)
plt.savefig("plots/swift-vs-cl.pdf")



#PLOT SPEEDUP VS WINDOW
print("SPEEDUP VS WINDOW")
plt.rcParams.update({'font.size': 32})
win10=[]
win20=[]
win_adpt=[]
baseline=[]
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10))
nb_samples = 50
total_samples = 100

baseline, baseline_mean_period = baseline_find_time("logs/all_llama2-13b_num-100_new-512_data-gsm8k.out", 1, total_samples, baseline)
win_adpt, mean_period, inf_end, end_time = find_time("logs/conflayers_llama2-13b_num-100_new-512_data-gsm8k.out", 1, total_samples, win_adpt, baseline)
win10, mean_period, inf_end, end_time = find_time("logs/conflayers_llama2-13b_num-100_new-512_data-gsm8k_win-10.out", 1, total_samples, win10, baseline)
win20, mean_period, inf_end, end_time = find_time("logs/conflayers_llama2-13b_num-100_new-512_data-gsm8k_win-20.out", 1, total_samples, win20, baseline)

axes[0].plot(win_adpt, marker='o', color='royalblue', label='Adaptive Window')
axes[0].plot(win10, marker='o', color='orange', label='Window = 10')
axes[0].plot(win20, marker='o', color='green', label='Window = 20')
axes[0].set_title("(a) LLaMa-2-13B")
axes[0].set_ylabel("Speedup")
axes[0].tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=False)
axes[0].grid()

win10=[]
win20=[]
win_adpt=[]
baseline=[]
baseline, baseline_mean_period = baseline_find_time("logs/all_llama2-70b_num-100_new-512_data-gsm8k.out", 1, total_samples, baseline)
win_adpt, mean_period, inf_end, end_time = find_time("logs/conflayers_llama2-70b_num-100_new-512_data-gsm8k.out", 1, total_samples, win_adpt, baseline)
win10, mean_period, inf_end, end_time = find_time("logs/conflayers_llama2-70b_num-100_new-512_data-gsm8k_win-10.out", 1, total_samples, win10, baseline)
win20, mean_period, inf_end, end_time = find_time("logs/conflayers_llama2-70b_num-100_new-512_data-gsm8k_win-20.out", 1, total_samples, win20, baseline)

axes[1].plot(win_adpt, marker='o', color='royalblue')
axes[1].plot(win10, marker='o', color='orange')
axes[1].plot(win20, marker='o', color='green')
axes[1].set_title("(b) LLaMa-2-70B")
axes[1].set_xlabel("Input Prompt Index")
axes[1].set_ylabel("Speedup")
axes[1].grid()

h, l = axes[0].get_legend_handles_labels()
axes[0].legend(loc='upper center',        # position relative to bbox
    bbox_to_anchor=(0.5, 1.5),# x=center (0.5), y=slightly above plot (1.15)
    ncol=3,                    # number of legend columns (optional)
    frameon=False,              # remove box border for cleaner look
    fontsize=28
)
# axes[0].set_label()
fig.subplots_adjust(top=0.85, bottom=0.15)
plt.subplots_adjust(hspace=0.3)
# plt.xlabel("Lambda")
# plt.ylabel("Speedup")
plt.savefig("plots/speedup-vs-window.pdf")



#PLOT SPEEDUP VS SET
print("SPEEDUP VS SET")
plt.rcParams.update({'font.size': 18})
labels = ['GSM8K', 'WMT14', 'Alpaca']
x = np.arange(len(labels))   # base x locations
width = 0.18
bar_spacing = 1.15

colors = [
    "#F4A7A1",  # light red (soft rose)
    "#A8E6A2",  # light green (mint)
    "#A1C9F4",  # light blue (sky pastel)
    "#FFB482",  # light orange (peach)
    "#7E54AE",  # light purple (lavender)
    "#F7E479",  # light yellow (buttery)
    
]

data = [
    [1.125, 1.172, 1.134],
    [1.097, 1.107, 1.117],
    [1.085, 1.068, 1.035],
    [1.043, 1.029, 0.998],
]
labels_bars = ['ConfLayers with Task-specific Skip Layer Set', 'ConfLayers with CNN-DM Skip Layer Set', 'SWIFT with Task-specific Skip Layer Set', 'SWIFT with CNN-DM Skip Layer Set']

# Example line data: 4 datasets × 3 groups = 12 points (1 per bar)
line_data = [
     0.86, 0.914,0.974, 0.964,
     0.954, 0.976,0.961, 0.892,
     0.967, 0.975,0.901, 0.919,

]

n_datasets = len(data)      
n_groups = len(labels)  
bar_centers = [[None]*n_datasets for _ in range(n_groups)]

fig, ax1 = plt.subplots(figsize=(14, 3.5))
ax2 = ax1.twinx()

# For storing x positions for line data
x_line_positions = []

for i, (d, c, lbl, line_vals) in enumerate(zip(data, colors, labels_bars, line_data)):
    offset = (i - 1.5) * width * bar_spacing
    bar_positions = x + offset
    bars = ax1.bar(bar_positions, d, width, label=lbl, color=c, edgecolor='gray', linewidth=0.6, zorder=3)
    
    for grp_idx, pos in enumerate(bar_positions):
        bar_centers[grp_idx][i] = pos
    
    # Add text labels above bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.005,
            f'{height:.3f}',
            ha='center', va='bottom', fontsize=12, color='black', zorder=4
        )
    
x_seq = []
marker_colors=[]
for grp_idx in range(n_groups):
    for ds_idx in range(n_datasets):
        x_seq.append(bar_centers[grp_idx][ds_idx])
        marker_colors.append(colors[ds_idx])

assert len(x_seq) == len(line_data), "line_data must have n_groups * n_datasets elements"

ax2.plot(
    x_seq, line_data, color="#656565", linewidth=2.0, zorder=5, label='Line across bars'
)

for xi, yi, mc in zip(x_seq, line_data, marker_colors):
    ax2.scatter(xi, yi, color=mc, s=80, zorder=6)
    ax2.text(
        xi, yi + 0.015, f"{yi:.3f}",
        ha='center', va='bottom', fontsize=12, color="#656565", zorder=7
    )

# Labeling
ax1.set_ylabel('Speedup')
ax1.set_axisbelow(False)
# ax1.set_title('Grouped Bar Plot (4 bars per group)')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend(
    loc='upper center',        # position relative to bbox
    bbox_to_anchor=(0.5, 1.55),# x=center (0.5), y=slightly above plot (1.15)
    ncol=2,                    # number of legend columns (optional)
    frameon=False,              # remove box border for cleaner look
)
ax1.set_ylim((0.9, 1.4))
ax1.grid()

ax2.set_ylabel("Acceptance Rate", color="#656565")
ax2.set_ylim((0.5, 1.05))  # match scales, or customize
ax2.grid(False)  # turn off right grid

sns.despine(ax=ax1, right=False)  # keep right spine for secondary axis
ax1.grid(True, axis='y', linestyle='--', alpha=0.6, zorder=0)
ax1.tick_params(axis='y')
ax2.tick_params(axis='y', color="#656565")
ax2.spines['right'].set_color("#656565")
for tick in ax2.get_yticklabels():
    tick.set_color("#656565")

plt.subplots_adjust(
    top=0.6,    # move legend/title closer
)
plt.tight_layout()
plt.savefig("plots/speedup-vs-set.pdf")
