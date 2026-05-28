# Project 3：分子动力学短程力计算 — CPU 性能优化

## 0. 快速开始

项目 3 是编程作业，请首先通过 [GitHub Classroom 链接](https://classroom.github.com/a/9aCVN0wH) 接受作业。这个仓库含有初始代码。

为方便起见，我们今年提供了统一的基于 SSH 协议的在线判题系统。您可以参照下面的说明进行配置。但请注意，您**必须**将代码推送到 GitHub 并提交到 Gradescope。我们将通过 Gradescope 平台对您的代码进行审阅。但是我们会根据 SOJ 的重测结果作为最终的给分依据。

**重要提示**

为了促进大家使用 AI 辅助探索，今年我们取消了前 30 名的额外分数，并允许您在探索过程中使用 AI，但我们**也会**对代码进行抄袭检查。对于那些达到前 30 名加速比的同学，或如果您收到了相关请求，请将您的**解题报告**提交至 `hezb2023<at>shanghaitech.edu.cn` 和 `hanlt2025<at>shanghaitech.edu.cn`。如果您使用了 AI，请在下方附上您**完整的**的 AI 对话记录。我们会仔细审查，并在项目结束后分享一些优秀的思路或解题报告供大家学习！

## 1. 物理背景

### 1.1 Lennard-Jones 势函数

分子动力学（Molecular Dynamics, MD）模拟通过求解牛顿运动方程数值地演化大量粒子的轨迹，是计算化学、材料科学、生物物理的核心方法（GROMACS、LAMMPS、NAMD 等软件每天在世界上消耗的 CPU 核时以亿计）。

**Lennard-Jones (LJ) 势** 是描述非键合相互作用（范德瓦尔斯力 + 短程排斥）的最经典模型：

$$
U_{LJ}(r) = 4\varepsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right]
$$

其中：
- $r$ 是两粒子间距
- $\sigma$ 决定平衡距离（势能极小点在 $r = 2^{1/6}\sigma$）
- $\varepsilon$ 是势阱深度
- 第 12 次方项描述短程电子云排斥
- 第 6 次方项描述长程吸引（伦敦色散力）

本项目使用**约化单位**（reduced units）：$\varepsilon = \sigma = 1$，因此

$$
U_{LJ}(r) = 4 \left( r^{-12} - r^{-6} \right)
$$

### 1.2 力的标量形式

粒子 $i$ 受到粒子 $j$ 的力（沿 $\vec{r}_{ij} = \vec{r}_i - \vec{r}_j$ 方向）：

$$
\vec{F}_{ij} = -\nabla_i U_{LJ}(r_{ij}) = \frac{24}{r_{ij}^2}\left(2 r_{ij}^{-12} - r_{ij}^{-6}\right) \cdot \vec{r}_{ij}
$$

记 $f_{\text{scal}} = \dfrac{24}{r^2}\left(2 r^{-12} - r^{-6}\right)$，则 $\vec{F}_{ij} = f_{\text{scal}} \cdot \vec{r}_{ij}$。

牛顿第三定律保证 $\vec{F}_{ji} = -\vec{F}_{ij}$。

### 1.3 截断与周期性边界（Periodic Boundary Conditions, PBC）

直接对所有 $\mathcal{O}(N^2)$ 粒子对计算开销过大。物理上 LJ 势在 $r > 2.5\sigma$ 处衰减极快（约为 $\varepsilon$ 的 1.6%），因此在所有主流 MD 包中采用**截断**：

$$
U(r) = \begin{cases} U_{LJ}(r), & r < r_c \\ 0, & r \ge r_c \end{cases}, \quad r_c = 2.5
$$

为模拟无限大体系，采用**周期性边界条件**：粒子位于一个边长为 $L$ 的立方盒子中，每个粒子的"镜像副本"周期性铺满空间。两粒子间距取**最近映像约定**（minimum image convention）：

$$
\Delta r_{\alpha} = \Delta r_{\alpha} - L \cdot \text{round}(\Delta r_{\alpha} / L), \quad \alpha \in \{x, y, z\}
$$

### 1.4 Cell List 加速到 $\mathcal{O}(N)$

把盒子均匀划分成边长 $\geq r_c$ 的 cell。某粒子的所有 cutoff 内邻居必在自身 cell 或 26 个相邻 cell 中。计算复杂度从 $\mathcal{O}(N^2)$ 降至 $\mathcal{O}(N)$（线性！），代价是数据结构变复杂。

本项目提供的 baseline 已实现 cell list，但**低效**（无 SIMD/OpenMP），等待你来优化。

---

## 2. 任务定义

实现函数：
```c
void compute_forces_optimized(ParticleSystem *sys);
```

**输入**：`sys` 中已填好 `pos[N]`（粒子位置）、`n`（粒子数）、`box`（盒子边长）

**输出**：填充 `sys->force[N]`，使每个粒子受到的合力满足相对误差 $< 10^{-6}$（与 baseline 对比）

**约束**：
- 不得使用 BLAS / FFTW / Intel MKL 等数值计算库
- 仅允许修改 `src/optimized.c` 和 `numactl` 配置文件（见下文），其他所有配置将被忽略
- 不得攻击/伪造数据

---

## 3. 输入参数

`init_particles(N, density, seed)` 在 $[0, L)^3$ 立方盒子中放置 $N$ 个粒子：
- 盒子边长 $L = (N / \rho)^{1/3}$，其中 $\rho$ 为请求密度
- 粒子初始位置：由 `seed` 控制的多种随机分层布局混合，保留最小间距结构（避免粒子重合产生发散力）
- 力守恒：理论上 $\sum_i \vec{F}_i = \vec{0}$（评测会检查）

**评测规模**：benchmark 会在 `2e6`、`4e6`、`8e6` 等名义大规模附近采样；
精确规模和 seed 由评测端决定。

对每个采样规模 $N_i$，`bench.sh` 会直接记录一次 baseline 正式运行时间 $B_i$，
不做 baseline warmup；在一次 optimized warmup 后记录 optimized 5 次正式运行中的
最优时间 $O_i$。该规模的 speedup 为：

$$
s_i = \frac{B_i}{O_i}
$$

最终成绩使用各规模 speedup 的几何平均：

$$
\text{GeoSpeedup} = \exp\left(\frac{1}{k}\sum_{i=1}^{k}\ln(s_i)\right)
$$

`bench.sh` 使用未四舍五入的各规模 speedup 计算 `GeoSpeedup`，只在最终打印时
保留两位小数。这个公式等价于 $(s_1s_2\cdots s_k)^{1/k}$，使用 log 写法是为了
数值更稳定。

---

## 4. 本地自测

### 4.1 构建

```bash
make            # 编译全部二进制到 bin/
```

### 4.2 正确性验证

```bash
make correctness    # brute-force vs baseline，小规模 (512, 1000, 2000)
make test N=4096    # baseline vs optimized，检查力误差 < 1e-6
```

### 4.3 性能测试

```bash
./bench.sh
```

默认情况下，`bench.sh` 从 `/dev/urandom` 读取 `seed`，并对每个名义规模做小幅随机扰动。
如果需要可复现的本地调试，可以显式传入 seed 并关闭规模扰动：

```bash
N_JITTER=0 ./bench.sh "2000000 4000000 8000000" 0.5 42
```

输出示例：
```
=== N-body Benchmark ===
NominalSizes=2000000 4000000 8000000  density=0.5  seed=2179341566  n_jitter_pct=3  optimized_runs=5 (+ optimized warmup per size)

===============================
N=1987342  density=0.5  seed=2179341566  baseline_runs=1  optimized_runs=5 (+ optimized warmup)

[1 - Baseline]
  Run 1  : 1.234 s
  Time   : 1.234 s

[2 - Optimized]
  Warmup : done
  Run 1  : 0.456 s
  ...
  Best   : 0.440 s

[3] Correctness
...

-------------------------------
  N         : 2000000
  Baseline  : 1.234 s (single run)
  Optimized : 0.440 s (best of 5)
  Speedup   : 2.80x
-------------------------------

...

===============================
GeoSpeedup : 2.75x
Sizes      : 2000000 4000000 8000000
===============================
```

### 4.4 常用开发命令

| 命令 | 说明 |
|------|------|
| `make` | 编译全部 |
| `make correctness` | 小规模正确性验证 |
| `make test N=...` | baseline vs optimized 力对比 |
| `make clean` | 清理 `bin/`、`tmp/`、`src/*.o` |
| `cd src && make` | 快速编译检查（`-O0 -g`，便于调试） |

---

## 5. 提交到评测系统

评测通过 SOJ（SSH 协议在线评测）进行。

### 5.0 在 Github 上认领添加公钥

终端执行 `ssh-keygen`，一路回车即可

此时会生成两个文件，以 `/home/zambar/.ssh/id_ed25519` 这个文件为例，这个就是私钥，请妥善保管，不要在任何地方公开！而 `/home/zambar/.ssh/id_ed25519.pub` 就是公钥。

前往 https://github.com/settings/keys 点击 `New SSH Key`，然后在内容处粘贴公钥即可。

添加完成后，确保本地运行下列指令能正常返回：

```bash
ssh -T git@github.com
```

如果返回类似

```
Hi HeZeBang! You've successfully authenticated, but GitHub does not provide shell access.
```

说明已经配置成功。

### 5.1 配置 SSH

在 `~/.ssh/config` 中添加：

```
Host soj
    HostName 10.15.89.111
    Port 2222
    User [Your_Github_UserName]
```

测试连接：

```bash
ssh soj
```

应看到 SOJ 欢迎信息和可用命令提示。

### 5.2 上传文件

需要上传两个文件，使用 `scp`（路径格式为 `<题目ID>/<文件名>`）：

```bash
# 上传优化代码
scp src/optimized.c soj:proj3/optimized.c

# 上传 affinity 配置
scp affinity.conf soj:proj3/affinity.conf
```

> **注意**：如果 `scp` 报错，可能是 OpenSSH 版本问题：
> - `< 8.7`：改用 `sftp` 命令
> - `8.7 ~ 9.0`：添加 `-s` 选项，如 `scp -s ...`
> - `> 9.0`：直接使用即可

### 5.3 提交评测

上传完成后，在 SSH shell 中触发评测：

```bash
ssh soj submit proj3
```

查看提交状态：

```bash
ssh soj list          # 列出所有提交
ssh soj status <id>   # 查看某次提交详情
```

### 5.4 affinity.conf 说明

为了给大家最真实的调优体验，我们允许大家通过修改配置文件的方式修改 `numactl` 运行参数。你可以查找相关资料来知道这个是怎么用的，并且将如何优化你的程序运行。

`affinity.conf` 控制 `numactl` 如何绑定 CPU/内存。格式为 `key = value`，每行一条：

```ini
# CPU 绑定（最多选一项）
physcpubind = 0-127         # 指定物理 CPU 范围
cpunodebind = 0             # 指定 NUMA 节点

# 内存绑定（最多选一项）
membind     = 0             # 仅从指定节点分配
interleave  = 0,1           # 跨节点轮转分配
preferred   = 0             # 优先某节点，可溢出
localalloc  = true          # 在 CPU 所在节点分配
```

留空表示不使用 `numactl`，由内核调度器选择 CPU。参考 `affinity.conf.example`。

## 6. 评分标准

根据**几何平均加速比** (GeoSpeedup) 采用分段线性插值计算分数：

| 加速比 (GeoSpeedup) | 分数 |
|-----|-------|
| 0x | 0% |
| 1x | 20% |
| 10x | 30% |
| 20x | 40% |
| 30x | 50% |
| 40x | 60% |
| 50x | 80% |
| 85x+ | 100% |

相邻关键点间采用线性插值，例如：
- 5x 加速比 → 25%（1x 的 20% 和 10x 的 30% 的中点）
- 15x 加速比 → 35%（10x 的 30% 和 20x 的 40% 的中点）

**编译失败、答案错误**时，无论加速比如何，分数为 0 分。

## 7. 附录

### 评测机参数

我们的提交和最终评测的评测机均在这一台 AMD EPYC 机器上运行，因此以下信息可能对你有帮助。

#### `lscpu`

```
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           43 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  128
On-line CPU(s) list:                     0-127
Vendor ID:                               AuthenticAMD
Model name:                              AMD EPYC 7742 64-Core Processor
CPU family:                              23
Model:                                   49
Thread(s) per core:                      1
Core(s) per socket:                      64
Socket(s):                               2
Stepping:                                0
Frequency boost:                         enabled
CPU max MHz:                             2250.0000
CPU min MHz:                             1500.0000
BogoMIPS:                                4500.19
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 cqm rdt_a rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif v_spec_ctrl umip rdpid overflow_recov succor smca sme sev sev_es ibpb_exit_to_user
Virtualization:                          AMD-V
L1d cache:                               4 MiB (128 instances)
L1i cache:                               4 MiB (128 instances)
L2 cache:                                64 MiB (128 instances)
L3 cache:                                512 MiB (32 instances)
NUMA node(s):                            2
NUMA node0 CPU(s):                       0-63
NUMA node1 CPU(s):                       64-127
```

#### Core-to-core Latency

![epyc](./img/epyc-7742-64c-636.png)

#### 拓扑

![lstopo](./img/topo.svg)

---

如果你对题目/评测机有问题，你可以联系：

- 任何任课老师
- `hezb2023<at>shanghaitech.edu.cn`
- `hanlt2025<at>shanghaitech.edu.cn`
