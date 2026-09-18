# PyTorch on JetPack 7.2 for Jetson Orin NX (the undocumented path)

Working CUDA-enabled PyTorch on a Jetson Orin NX running JetPack 7.2, in one
script: [`install.sh`](install.sh).

NVIDIA just released JetPack 7.2 support for Orin. It is a real step forward —
Ubuntu 24.04, CUDA 13.2, Python 3.12, ROS 2 Jazzy — but the PyTorch story is
not documented yet. These are field notes from getting it working, so you don't
spend the same hours on it.

---

## The problem

You flash JetPack 7.2, run `pip install torch`, and it appears to work.
`torch.cuda.is_available()` returns `True`. Then the first real GPU operation
dies:

```
CUDA error: no kernel image is available for execution on the device
```

The wheel installed fine, reports CUDA, and cannot run a single kernel on your
board. That is the whole trap: **`torch.cuda.is_available()` returning `True` is
not proof of anything.** Every wrong wheel below returns `True`.

The cause is architecture mismatch. Jetson Orin is compute capability `sm_87`.
The wheels pip hands you by default are built for datacenter GPUs (`sm_110`,
`sm_121`) and contain no `sm_87` kernels at all.

### Why the obvious paths fail

| What you try | What happens |
|---|---|
| `pip install torch` | Installs an `sm_110`/`sm_121` build. Reports CUDA available, fails on first kernel launch. |
| Community SBSA `cu130` wheels | Also `sm_110`/`sm_121`, and they need system libs a fresh flash lacks (NVPL, cuDSS). Even after adding those, still no `sm_87` kernel. |
| Jetson PyTorch container | No `r39` image exists yet. `autotag` falls back to a JetPack 6 (`r36.4`) image, which fails on the JP7.2 driver with `CUDA error 801 (operation not supported)`. |
| Building from source | Works, eventually. Hours of compile time on the board, and unnecessary. |

---

## Why this solution

There **is** a correct wheel. It is just invisible unless you ask for it
precisely:

```bash
pip install --pre torch --extra-index-url https://download.pytorch.org/whl/cu132
```

Four things in that line are load-bearing:

**1. `--pre` is mandatory.** The working wheels are prereleases. Without `--pre`,
pip returns nothing and you wrongly conclude no wheel exists. It exists.

**2. `cu132` must match your CUDA.** JetPack 7.2 ships CUDA 13.2, so the wheel
suffix is `cu132`. The aarch64 build at that index includes `sm_87`. A different
CUDA version needs a different suffix.

**3. `--extra-index-url`, not `--index-url`.** `--index-url` *replaces* PyPI, so
numpy and every other dependency fail to resolve. `--extra-index-url` adds the
CUDA index alongside PyPI.

**4. Python 3.12 only.** The `cu132` aarch64 wheels are built for `cp312`.
JetPack 7.2 ships 3.12. There is no matching 3.13 wheel, so moving to 3.13 puts
you straight back into building from source. Old 3.10 wheels don't work either.

Two more things that bite before you even get to the wheel:

**CUDA must be on your PATH, not just installed.** A fresh flash ships the CUDA
13.2 runtime, but if `CUDA_HOME` / `PATH` / `LD_LIBRARY_PATH` don't point at it,
torch won't find it at runtime. "Installed" is not "visible." This is the first
thing to check if imports or `nvcc` behave oddly.

**Two system libraries are missing from a fresh flash.** The wheel links against
NVPL (CPU BLAS/LAPACK) and cuDSS (sparse solver). Without them, `import torch`
fails on a missing `lib*.so` before it ever reaches CUDA.

`install.sh` handles all of it.

---

## Prerequisites

### Hardware and OS

Verified on exactly this:

| | |
|---|---|
| Module | NVIDIA Jetson Orin NX 16GB, P-number `p3767-0000` |
| Reported board | NVIDIA Jetson Orin NX Engineering Reference Developer Kit |
| JetPack | 7.2 — L4T **39.2.1** |
| OS | Ubuntu 24.04 "Noble Numbat" |
| Kernel | 6.8.12-1021-tegra |
| CUDA | 13.2.86 |
| cuDNN | 9.20.0 |
| Python | 3.12 |
| GPU arch | sm_87 (capability 8.7) |
| PyTorch | torch 2.14.0+cu132 (prerelease), torchvision from the same index |
| numpy | 2.2.6 |
| ROS | ROS 2 Jazzy (system install, outside the Conda env) |
| LeRobot | 0.6.0 (installed `--no-deps`, see below) |
| Power mode | 15W (mode 2) |

Those values come from `jetson_release` (jetson-stats 7.2.1) on the machine this
was built on. "Reported board" is the device-tree name, which reads as the NVIDIA
reference devkit when the board is flashed with the reference BSP — including on
third-party carriers. The module P-number is the value to actually match against.

**Power mode matters for benchmarks, not correctness.** 15W is a conservative
mode; nothing here depends on it, but if throughput looks low, check
`sudo nvpmodel -q` and consider MAXN before concluding something is wrong with
the build.

The exact wheel that lands is, for example:

```
torch-2.14.0+cu132-cp312-cp312-manylinux_2_28_aarch64.whl
```

Read that filename as the checklist: `cu132` (your CUDA), `cp312` (your Python),
`aarch64` (your arch), `manylinux_2_28` (glibc ≥ 2.28, which Ubuntu 24.04 is well
past). If all four match your board, it's the right wheel.

**These are prereleases, so the version moves.** This started at 2.12.1 and is
now 2.14.0 — the index rolls forward and `--pre` always takes the newest. That's
fine; the architecture is what matters, not the version. If you need
reproducibility across boards, pin explicitly (`torch==2.14.0+cu132`) rather
than letting each install drift to whatever is newest that day.

Other Orin modules (Orin Nano, AGX Orin) are also `sm_87` and will very likely
work. A different JetPack/CUDA version, or a non-Orin board (Thor is `sm_110`),
needs a different wheel — see [For other hardware](#for-other-hardware).

### 1. JetPack 7.2 flashed

Before any PyTorch work, the JP7.2 BSP has to be on the board. Full flashing
steps live in Seeed's wiki:
<https://wiki.seeedstudio.com/recomputer_jetson_super_getting_started/#flash-jetpack-os>

- **Standard NVIDIA devkit** — grab the image from the NVIDIA Jetson download
  center and flash from an Ubuntu host over USB-C.
- **reComputer, ISO route (what I did)** — download the JP7.2 image for your
  *exact* module from the wiki above and flash it directly. Match module and
  version carefully; for Orin NX 16GB it is the JP7.2 / Orin NX 16GB row. Check
  the SHA256 before flashing. This is the reliable path today.
- **Seeed developer tool** — `pip install --upgrade seeed-jetson-developer==0.2.0`,
  then follow the same wiki page. Heads up: at time of writing the tool's UI did
  not list reComputer Super + JP7.2 for me, and auto-detection offered an AGX
  Orin image (wrong module) that failed to flash. Confirm it actually offers your
  board and the 7.2 BSP before relying on it; otherwise use the ISO route.

Confirm before continuing:

```bash
cat /etc/nv_tegra_release      # expect R39 (L4T 39.2.x)
ls -d /usr/local/cuda-13.2     # the toolkit the wheels must match
nvcc --version                 # only works once CUDA is on PATH — see Step 0
```

`jetson_release` (from `jetson-stats`) gives the whole picture in one shot and is
worth installing:

```bash
sudo pip3 install -U jetson-stats
jetson_release
```

> **`Jetpack missing!` in that output does not mean JetPack is missing.**
> `jetson-stats` maps L4T versions to JetPack names from a built-in table, and
> L4T 39.2.1 is newer than the table it shipped with, so it prints
> `Jetpack missing!` while correctly reporting L4T 39.2.1, CUDA 13.2.86 and
> cuDNN 9.20.0 right below. This is a cosmetic lookup gap in the tool, not a
> problem with your flash — the same "JP7.2 is newer than the tooling" theme as
> the missing PyTorch wheels. Trust `/etc/nv_tegra_release` instead.
>
> Similarly, `jtop: Service: Inactive` just means the daemon isn't running yet:
> `sudo systemctl restart jtop.service`, then log out and back in.

### 2. A Conda environment with Python 3.12

`install.sh` installs into an **existing** Conda env — it does not create one.
The default name in the script is `aditya`; change `CONDA_ENV` at the top if
yours differs.

If you don't have one yet, Miniforge is the aarch64-friendly choice:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh
# restart your shell, then:
conda create -n aditya python=3.12 -y
```

The script looks for Conda under `~/miniforge3`, then `~/miniconda3`, then
whatever `conda info --base` reports. **Run it from a shell where Conda is
initialized**, or Step 4 will fail.

> Prefer a plain `venv`? That works too — `python3.12 -m venv ~/.venvs/jetson-jp72
> --system-site-packages`, activate it, and run the pip steps by hand. The
> `--system-site-packages` flag is what lets the env see JetPack's TensorRT
> bindings. Conda envs do not inherit system Python packages, which is why the
> script's TensorRT check (Step 13) is informational rather than fatal.

### 3. Network and sudo

The script runs `apt-get` (needs sudo) and downloads wheels from
`download.pytorch.org`.

If `apt` fails with a lock held by another process, that is first-boot
auto-updates running. Let them finish, or reboot. **Do not delete the lock file.**

---

## How to run it

```bash
git clone https://github.com/Aadi0032007/jp_72_toch_build_aadi.git
cd jp_72_toch_build_aadi
chmod +x install.sh
./install.sh
```

Run it from a Conda-initialized shell. It is idempotent — safe to re-run.

### What the script does, step by step

| Step | What it does | Why |
|---|---|---|
| 0 | Locates `/usr/local/cuda-13.2`, exports `CUDA_HOME` / `PATH` / `LD_LIBRARY_PATH`, runs `nvcc --version` | CUDA has to be *visible*, not just installed. Hard-fails early with a list of what CUDA installs it did find. |
| 1 | Appends the same exports to `~/.bashrc` (guarded, won't duplicate) | So new terminals aren't silently broken. |
| 2 | `build-essential`, `cmake`, `git`, `pkg-config`, `wget`, `libopenblas-dev`, `libjpeg-dev`, `zlib1g-dev` | Build basics a fresh flash lacks. |
| 3 | Adds NVIDIA's CUDA apt repo, installs **NVPL** and **cuDSS**, writes `/etc/ld.so.conf.d/cudss.conf`, runs `ldconfig` | The two libs the wheel links against. cuDSS installs to a versioned subdir that isn't on the linker path by default. Both checks skip if already present. |
| 4 | Sources `conda.sh` and activates `$CONDA_ENV` | Everything after this lands in the env, not system Python. |
| 5 | Asserts Python is exactly **3.12** and machine is **aarch64** | Fails loudly now rather than after a confusing wheel resolution. |
| 6 | Upgrades `pip`, `setuptools`, `wheel`, `packaging` | Old pip mishandles the prerelease index. |
| 7 | Uninstalls `torch`, `torchvision`, `torchaudio`, `torchdata`, `torchtext`, `torch-tensorrt`, then `pip cache purge` | Clears any wrong-arch wheel that snuck in — and the cache, so pip can't reuse a bad one. |
| 8 | **`pip install --pre torch --extra-index-url .../cu132`** | The one that works. |
| 9 | **Verification**: prints version, device, capability, then runs a real 2048×2048 matmul and `torch.cuda.synchronize()` | The only test that counts. Raises if CUDA is unavailable. |
| 10 | Installs `torchvision` from the same index — *after* torch passes | No point adding it on a broken base. |
| 11 | `numpy`, `pillow`, `matplotlib`, `jupyterlab`, `ipykernel` | Common working set. |
| 12 | Registers a Jupyter kernel named `Jetson JP7.2 - <env>` | So notebooks pick the right interpreter. |
| 13 | Tries `import tensorrt`, reports but does not fail | Conda doesn't inherit JetPack's system Python packages. Does not affect PyTorch CUDA. |
| 14 | Prints a final stack summary | One place to screenshot when asking for help. |

Note the ordering: **torch is verified with a real kernel launch (Step 9) before
anything else is installed on top of it.** If it's broken, you find out at step
9, not at step 14.

### A clean pass looks like

```
GPU             : Orin
Capability      : (8, 7)
Running CUDA matrix multiplication test...
GPU computation result: -1234.56
CUDA KERNEL TEST: PASS
```

Capability `(8, 7)` is `sm_87` — the right architecture. The printed number is
the proof a kernel actually ran.

You may still see an `sm_87 is not compatible` warning. It is cosmetic (NVIDIA
confirmed it is being removed). What matters is whether the matmul completes.
If you instead get `no kernel image is available`, you are on a wrong-arch
wheel — go back to [For other hardware](#for-other-hardware).

### Verifying later, by hand

```bash
conda activate aditya
python -c "import torch; x=torch.randn(1024,1024,device='cuda'); torch.cuda.synchronize(); print('OK', float((x@x).sum()), torch.__version__)"
```

Expect `OK`, a number, and a version ending in `+cu132`.

### Troubleshooting

- **`ERROR: Could not find /usr/local/cuda-13.2`** — you're not on JP7.2, or CUDA
  landed elsewhere. `ls -d /usr/local/cuda*` and point `CUDA_DIR` at the right one.
- **`ERROR: Conda not found`** — run from a Conda-initialized shell, or fix
  `CONDA_ENV` / the Miniforge path.
- **`Python 3.12 required`** — your env is on another version. Recreate it with
  `python=3.12`. Do not work around this.
- **Import fails on some other `lib*.so`** — same pattern as NVPL/cuDSS every
  time: `find / -name "libNAME.so*" 2>/dev/null`, add its directory to a file
  under `/etc/ld.so.conf.d/`, run `sudo ldconfig`.
- **pip can't find a wheel** — see [For other hardware](#for-other-hardware) for
  how to inspect the tags pip actually sees.

---

## Don't install `torchaudio` (yet)

The script deliberately skips it. It is the usual source of version conflicts
and you almost certainly don't need it for a CV or robotics stack. If you do
need it later, pin it to your exact installed torch version rather than letting
pip pick.

---

## ROS 2 Jazzy (system, not the Conda env)

ROS goes in the system environment, **not** the torch env. ROS 2 Jazzy is built
against the system Python 3.12 and its system packages (including a specific
numpy), so sourcing it inside the cu132 env collides two Python paths and two
numpys. Keep them as separate worlds that talk over ROS topics — which is the
intended architecture anyway. If a single process genuinely needs both torch and
ROS, use a `--system-site-packages` venv rather than pip-installing ROS into the
Conda env.

Install in a plain terminal, with Conda deactivated (`conda deactivate`):

```bash
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

sudo apt install -y software-properties-common curl
sudo add-apt-repository universe -y

export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb"
sudo apt install -y /tmp/ros2-apt-source.deb

sudo apt update && sudo apt upgrade -y
sudo apt install -y ros-jazzy-desktop   # or: ros-jazzy-ros-base ros-jazzy-turtlesim (lighter)
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source /opt/ros/jazzy/setup.bash
```

Sanity check with turtlesim, in two plain terminals (source ROS in each):

```bash
ros2 run turtlesim turtlesim_node        # terminal 1
ros2 run turtlesim turtle_teleop_key     # terminal 2, arrow keys to drive
```

---

## LeRobot (in the torch env, but do NOT let it touch torch)

LeRobot belongs in the *same* environment as torch because it imports torch
directly. The trap: **LeRobot 0.6.0 pins `torch<2.12.0`**, but the only working
Orin wheel is a prerelease above that ceiling (2.14.0 at time of writing, and
climbing). A plain `pip install lerobot` "resolves"
that by ripping out your cu132 build and installing a generic `torch 2.11.0`
that fails on the Orin with `no kernel image available`. The version number even
looks reasonable, so nothing seems wrong until the first kernel launch.

Install it without deps, then add its runtime deps manually, skipping anything
torch-related:

```bash
conda activate aditya

pip install lerobot --no-deps

# LeRobot's runtime deps, pinned to what it wants, with NO torch/torchvision:
pip install \
  "cmake>=3.29,<4.2" "draccus==0.10.0" "opencv-python-headless>=4.9,<4.14" \
  einops gymnasium huggingface-hub safetensors termcolor tqdm \
  "numpy>=2.0,<2.3" "packaging<26.0,>=24.2" "requests<3.0,>=2.32.0"
```

After this, `pip check` will still report torch, torchvision, and setuptools as
"incompatible" with LeRobot's pins. **That is expected and intended**: the torch
pins are a soft ceiling you are deliberately overriding, and setuptools is a
cosmetic build-tool nag. Everything else should be clean.

LeRobot 0.6.0 ran fine against torch 2.12.1 in practice. The wheel index has
since rolled to 2.14.0, which is further past the pin, so exercise the paths you
actually use (`lerobot-info`, a short `lerobot-record`) rather than assuming the
override is still free. If something does break on a newer torch, pinning back
to `torch==2.12.1+cu132` from the same index is the fallback — it is still a
valid `sm_87` build.

**Never use `pip install 'lerobot[extra]'`.** Any bracketed extras form
(`lerobot[feetech]`, `lerobot[deepdiff-dep]`, …) re-resolves the whole dependency
tree and pulls the generic `torch 2.11.0` wheel, silently replacing your cu132
build with one that fails on the Orin. Install each extra's underlying package by
its bare name instead. The ones the SO101 path needs:

```bash
pip install feetech-servo-sdk deepdiff
```

**numpy ABI note.** If your environment can see packages built against numpy
1.26 — an old system `pandas` via a `--system-site-packages` venv, or a stale
conda-channel build — it can crash with `numpy.dtype size changed ... binary
incompatibility` against numpy 2.x. Fix by installing a matching pandas into the
env so it shadows the other one, kept inside LeRobot's numpy ceiling:

```bash
pip install --ignore-installed "pandas>=2.2,<2.3" "numpy>=2.0,<2.3"
```

### Verify the whole stack coexists

Real kernel, not just `is_available`:

```bash
python -c "import torch, torchvision, lerobot, numpy; print('numpy', numpy.__version__); x=torch.randn(1024,1024,device='cuda'); torch.cuda.synchronize(); print('STACK OK', float((x@x).sum()), '| torch', torch.__version__, '| tv', torchvision.__version__, '| lerobot', lerobot.__version__)"
```

`STACK OK` with torch still on `+cu132` means torch, torchvision, and LeRobot all
share the same GPU build.

### SO101 teleop

LeRobot 0.6.0 ships hyphenated console scripts (not `python -m lerobot.*`).

```bash
lerobot-find-port                                 # identify the arm ports
```

**Serial port permission.** The quick fix that works immediately in the current
shell (NVIDIA's SO101 docs list this) is `chmod`, but it resets on replug:

```bash
sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1
```

The permanent fix is the `dialout` group, but it only takes effect after a full
logout/login — not just a new terminal:

```bash
sudo usermod -aG dialout $USER
```

First-time arms need their motor IDs set, then calibration, before teleop:

```bash
lerobot-setup-motors --teleop.type=so101_leader  --teleop.port=/dev/ttyACM0
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM1
lerobot-calibrate    --teleop.type=so101_leader  --teleop.port=/dev/ttyACM0 --teleop.id=leader_arm
lerobot-calibrate    --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=follower_arm
```

Then teleop:

```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=follower_arm \
  --teleop.type=so101_leader  --teleop.port=/dev/ttyACM0 --teleop.id=leader_arm
```

**Failure decoder.** `Permission denied` on the port → the chmod/dialout step
above. `Missing motor IDs ... found: {}` on an open port → the arm is unpowered,
mis-wired, or its motors were never ID'd (`lerobot-setup-motors`). Arm IDs
(`leader_arm`, `follower_arm`) are labels calibration files key to, so keep them
consistent between calibrate and teleoperate.

Other useful scripts: `lerobot-info`, `lerobot-find-cameras`, `lerobot-record`,
`lerobot-train`.

---

## JetPack 6.2 vs JetPack 7.2 (why this matters)

The point of moving to JP7.2 is that it fixes the top domino, and every layer
below falls into place. JP6.2 was a working stack, but only one exact
combination worked, and reaching it took source builds and manual library
installs.

| Layer | JetPack 6.2 (the old way) | JetPack 7.2 (this repo) |
|---|---|---|
| Ubuntu | 22.04 | 24.04 |
| Kernel | 5.15 | 6.8 (`6.8.12-1021-tegra`) |
| Flashing | Ubuntu host required | ISO / direct image, no host |
| Python | 3.10 only (3.13 silently pulls CPU torch) | 3.12 |
| CUDA | 12.6 (not on PATH by default) | 13.2.86 (still not on PATH by default) |
| ROS | Humble (Ubuntu 22 tier 1) | Jazzy (Ubuntu 24 tier 1) |
| PyTorch | torch 2.5.0, NVIDIA redist wheel only | torch 2.14.0+cu132 (prerelease, `--pre`) |
| torchvision | 0.20.0, built from source (`MAX_JOBS=2`, arch 8.7) | matching `+cu132` prerelease wheel, no source build |
| numpy | pinned 1.26.0 | 2.2.6 |
| cuSPARSELt | installed by hand | in the CUDA 13 toolkit |
| libcudss | missing on JP6.2, blocked torch 2.8+ | present / installable |
| LeRobot | reinstalls CPU torch unless installed last | `--no-deps` + bare extras |

The honest nuance: JP7.2 is not zero-friction. The working torch is a prerelease
`cu132` wheel that is undocumented and invisible to normal pip listing, and
LeRobot still tries to pull a wrong-arch torch. But the base OS, CUDA, ROS, and
Python now line up cleanly, where on JP6.2 every one of those was a fight. The
remaining friction is at the framework edge, not the platform.

---

## For other hardware

Verified only on Jetson Orin NX 16GB, JetPack 7.2, CUDA 13.2, Python 3.12. If
any of those differ:

- **Different Orin module (Orin Nano, AGX Orin)** — same `sm_87` arch, so the
  `cu132` wheel should work. Confirm with the Step 9 kernel test before trusting it.
- **Different CUDA version** — the wheel suffix must match. CUDA 13.0 needs
  `cu130`, 13.3 a `cu133` build, and so on. Check `nvcc --version` and change
  `CU_INDEX` at the top of `install.sh`.
- **Thor / non-Orin (`sm_110`+)** — different architecture entirely. The
  datacenter wheels that fail on Orin may be exactly right for you. Don't copy
  these workarounds without checking your `sm_` target.
- **Python 3.13** — no matching wheel for this stack at time of writing. Stay on
  3.12 or expect to build from source.

If pip can't find a wheel, diagnose what tags it actually sees:

```bash
python -m pip index versions torch --pre --extra-index-url https://download.pytorch.org/whl/cu132
python -m pip debug --verbose | grep -A 30 "Compatible tags"
```

That tells you whether pip sees a compatible `cp312` + `aarch64` wheel at all.

---

## Status

As of writing, this path is undocumented and relies on prerelease wheels, so
treat it as "works today, may change." If NVIDIA or Seeed publish official JP7.2
Orin wheels or an `r39` container, prefer those. Reported upstream to both.

*Maintained as field notes, not official guidance. Corrections welcome.*
