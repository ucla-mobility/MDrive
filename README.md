# MDriveBench: MDriveBench: Multi-Agent Multi-Granular Driving Benchmark
## Repository Structure and Scope

This repository implements **MDriveBench**, a multi-agent driving benchmark.  
It was originally built on top of **CoLMDriver**; CoLMDriver is now one of multiple models in the repository, and the benchmark infrastructure has been built on top of it.

MDriveBench provides:
- Benchmark infrastructure (CARLA integration, scenarios, evaluation, analysis)
- Multiple baseline and LLM-based driving models (TCP, CoDriving, LMDrive, UniAD, CoLMDriver, and VAD)
- Training code for CoLMDriver components

---

## Table of Contents
- [Quickstart](#quickstart)
- [Results Analysis and Visualization](#results-analysis-and-visualization)
  - [Results Analysis](#results-analysis)
  - [Visualizing Results](#visualizing-results)
- [Challenge Submission Instructions](#challenge-submission-instructions)
- [Baseline Evaluation Setup](#baseline-evaluation-setup)
  - [Evaluation of baselines](#evaluation-of-baselines)
  - [TCP Environment Setup](#tcp-environment-setup)
  - [CoDriving Environment Setup](#codriving-environment-setup)
  - [LMDrive Environment Setup](#lmdrive-environment-setup)
  - [UniAD Environment Setup](#uniad-environment-setup)
  - [VAD Environment Setup](#vad-environment-setup)
  - [CoLMDriver Environment Setup](#colmdriver-environment-setup)
  - [CoLMDriver Model Setup](#colmdriver-model-setup)
- [Full Benchmark Evaluation (Internal)](#full-benchmark-evaluation-internal)

---
## Quickstart

### 1) Download and set up CARLA
Use the setup script below. It applies compatibility fixes, so start CARLA from this install.

```bash
./download_and_setup_carla.sh
export CARLA_ROOT=$PWD/carla912
```

### 2) Create baseline eval environment
```bash
conda env create -f model_envs/run_custom_eval_baseline.yaml --solver libmamba
conda activate run_custom_eval_baseline
```

### 3) Start CARLA manually (fixed port)
```bash
# terminal A
$CARLA_ROOT/CarlaUE4.sh --world-port=2014 -RenderOffScreen
```

### 4) Run benchmark scenarios with your custom planner
Run LLM-Generated Scenarios:
```bash
# terminal B
python tools/run_custom_eval.py \
  --routes-dir scenarioset/llmgen \
  --agent /abs/path/to/agents.py \
  --agent-config /abs/path/to/agent_config.yaml
```

Run V2X-PnP Real-to-Sim Scenarios:
```bash
# terminal B
python tools/run_custom_eval.py \
  --routes-dir scenarioset/v2xpnp \
  --agent /abs/path/to/agents.py \
  --agent-config /abs/path/to/agent_config.yaml \
  --custom-actor-control-mode replay \
  --log-replay-actors
```


Warmup outputs are written to `results/results_driving_custom/warmupscenarios/<scenario_name>/`.

## Results Analysis and Visualization

### Results Analysis
Use `visualization/results_analysis.py` on any results folder, not just CoLMDriver outputs.

```bash
# Single run folder
python visualization/results_analysis.py \
  results/results_driving_custom/<run_tag> \
  --output-dir report/<run_tag>

# Compare multiple run folders and export one markdown summary
python visualization/results_analysis.py \
  results/results_driving_custom/<run_tag_a> \
  results/results_driving_custom/<run_tag_b> \
  --output-dir report/compare \
  --markdown report/compare/summary.md
```

The script generates markdown/CSV summaries and plots (driving score, success rate, infractions, negotiation stats when available).

### Visualizing Results
```bash
# Build a video from one scenario result folder
python visualization/gen_video.py \
  results/results_driving_custom/<run_tag>/<scenario_name>/<route_run_dir> \
  --output <scenario_name>.mp4
```

Optional flags include `--fps`, `--width`, `--height`, and `--font-scale`.

---

## Challenge Submission Instructions
To ensure your model is evaluated accurately, you must submit a single .zip file containing your model and code.

### Required ZIP File Structure
Your ZIP file must be organized as follows:
```
team_name.zip
├── agents.py           # Main agent class (must inherit from BaseAgent)
├── config/             # Folder containing all .yaml or .py configs
├── src/                # Folder containing model architecture & utilities
├── weights/            # Folder containing all trained checkpoints (.pth/.ckpt)
└── model_env.yaml      # Conda environment specification
```

### Environment Specification
MDriveBench supports two methods of environment provisioning. To ensure 100% reproducibility, we strongly recommend providing a Dockerfile.

1. ***Docker (Primary):*** Your Dockerfile should be based on a stable CUDA image (e.g., nvidia/cuda:11.3.1-devel-ubuntu20.04). It must install all necessary libraries so that the agent can run immediately upon container launch.

2. ***Conda (Fallback):*** If no Dockerfile is provided, we will build a dedicated environment using your model_env.yaml.
Note: Your code must be compatible with Python 3.7 to interface with the CARLA 0.9.12 API.
Do not include CARLA in your environment files; the evaluation server will automatically link the standardized CARLA 0.9.12 build.

### Evaluation Protocol
Our team will manually verify your submission using the following pipeline:

1. Env Build: The evaluator prioritizes the Dockerfile. If missing, it builds the Conda environment from model_env.yaml.
2. Path Injection: Standardized CARLA 0.9.12 PythonAPI will be appended to your PYTHONPATH.
3. Execution: Your agent will be run through a batch of closed-loop scenarios (OpenCDA, InterDrive, and Safety-critical).
4. Scoring: We will record the Driving Score (DS) and Success Rate (SR) as the official leaderboard metrics.

---
## Baseline Evaluation Setup

### Evaluation of baselines
Setup and get ckpts.

| Methods   | TCP | CoDriving               |
|-----------|---------|---------------------------|
| Installation Guide  | [github](https://github.com/OpenDriveLab/TCP)  | [github](https://github.com/CollaborativePerception/V2Xverse) |
| Checkpoints     |  [google drive](https://drive.google.com/file/d/1D-10aMUAOPk1yiOr-PvSOJMS_xi_eR7U/view?usp=sharing)  |  [google drive](https://drive.google.com/file/d/1Izg9wZ3ktR-mwn7J_ZqxrwBmtI1YJ6Xi/view?usp=sharing)   |

The downloaded checkpoints should follow this structure:
```Shell
|--CoLMDriver
    |--ckpt
        |--codriving
            |--perception
            |--planning
        |--TCP
            |--new.ckpt
```

### TCP Environment Setup
1. **Create TCP conda environment**
```bash
cd CoLMDriver
conda env create -f model_envs/tcp_codriving.yaml -n tcp_codriving
conda activate tcp_codriving
```
2. **Set CARLA path environment variables**
```bash
export CARLA_ROOT=PATHTOYOURREPOROOT/CoLMDriver/carla912
export PYTHONPATH=$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.12-py3.7-linux-x86_64.egg
```

### CoDriving Environment Setup
1. **Create CoDriving conda environment**
```bash
cd CoLMDriver
conda env create -f model_envs/tcp_codriving.yaml -n tcp_codriving
conda activate tcp_codriving
```
2. **Set CARLA path environment variables**
```bash
export CARLA_ROOT=PATHTOYOURREPOROOT/CoLMDriver/carla912
export PYTHONPATH=$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.12-py3.7-linux-x86_64.egg
```

### LMDrive Environment Setup

1. **Clone LMDrive into the assets directory**
```bash
git clone https://github.com/opendilab/LMDrive simulation/assets/LMDrive
```

2. **Prepare LMDrive checkpoints**
```bash
cd simulation/assets/LMDrive
mkdir -p ckpt
```

Download and place the following into `simulation/assets/LMDrive/ckpt`:
- Vision encoder: https://huggingface.co/OpenDILabCommunity/LMDrive-vision-encoder-r50-v1.0  
- LMDrive LLaVA weights: https://huggingface.co/OpenDILabCommunity/LMDrive-llava-v1.5-7b-v1.0  

Download and place the following into `CoLMDriver/ckpt/llava-v1.5-7b`:
- Base LLaVA model: https://huggingface.co/liuhaotian/llava-v1.5-7b  

3. **Create environment and install dependencies**
```bash
cd CoLMDriver
conda env create -f model_envs/lmdrive.yaml -n lmdrive
conda activate lmdrive

pip install carla-birdeye-view==1.1.1 --no-deps
pip install -e simulation/assets/LMDrive/vision_encoder
```

4. **Set CARLA path environment variables**
```bash
export CARLA_ROOT=PATHTOYOURREPOROOT/CoLMDriver/carla912
export PYTHONPATH=$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.12-py3.7-linux-x86_64.egg
```

### UniAD Environment Setup

UniAD is a unified perception–prediction–planning autonomous driving model.  
We evaluate it on the InterDrive benchmark using its official pretrained weights and a standardized conda environment to avoid dependency conflicts.

To ensure consistent and reproducible evaluation of the UniAD baseline model, we standardize the environment setup using a pre-built conda environment.
This avoids dependency conflicts and ensures that anyone can run UniAD without rebuilding environments from scratch.

The YAML file for the UniAD environment is located in:

`model_envs/uniad_env.yaml`

To create and activate the environment:

```bash
conda env create -f model_envs/uniad_env.yaml -n uniad_env
conda activate uniad_env
```

UniAD runs inside the `uniad_env` conda environment, which contains all required CUDA, PyTorch, CARLA, and UniAD dependencies.

#### Additional Files

Create a ckpt/UniAD directory if it does not exist:
`mkdir -p CoLMDriver/ckpt/UniAD`

Download the UniAD checkpoint from https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/uniad_base_b2d.pth
and place it here:

`CoLMDriver/ckpt/UniAD/uniad_base_b2d.pth`

Download the UniAD config file from https://github.com/Thinklab-SJTU/Bench2DriveZoo/blob/uniad/vad/adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py and place it in:

`simulation/assets/UniAD/base_e2e_b2d.py`

### VAD Environment Setup

The YAML file for the VAD environment is located in:

`model_envs/vad_env.yaml`

1. **Create VAD conda environment**
```bash
cd CoLMDriver
conda env create -f model_envs/vad_env.yaml -n vad
conda activate vad
```
#### **2. Start a Carla Instance**
```bash
CUDA_VISIBLE_DEVICES=0 $CARLA_ROOT/CarlaUE4.sh --world-port=2000 -prefer-nvidia
```

3. **Run VAD on Interdrive**
```bash
# CARLA must already be running on port 2000
bash scripts/eval/eval_mode.sh 0 2000 vad ideal Interdrive_all
```

### CoLMDriver Environment Setup

Use this section only for CoLMDriver-specific workflows.

#### vLLM env
```Shell
conda create -n vllm python=3.10
conda activate vllm
pip install vllm
```

#### CoLMDriver env
```Shell
conda create --name colmdriver python=3.7 cmake=3.22.1
conda activate colmdriver
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install cudnn -c conda-forge

pip install -r opencood/requirements.txt
pip install -r simulation/requirements.txt
pip install openai
```

#### Install Spconv (1.2.1)
We use spconv 1.2.1 to generate voxel features in the CoLMDriver perception stack.

```Shell
conda activate colmdriver
conda install -y cmake=3.22.1 ninja boost ccache -c conda-forge
pip install pybind11 numpy

git clone -b v1.2.1 --recursive https://github.com/traveller59/spconv.git
cd spconv
python setup.py bdist_wheel
pip install dist/spconv-1.2.1-*.whl
cd ..
```

#### Finish CoLMDriver local build setup
```Shell
conda activate colmdriver
python setup.py develop
python opencood/utils/setup.py build_ext --inplace
```

#### Install pypcd
```Shell
git clone https://github.com/klintan/pypcd.git
cd pypcd
pip install python-lzf
python setup.py install
cd ..
```

### CoLMDriver Model Setup

**Step 1:** Download checkpoints from [Google drive](https://drive.google.com/file/d/1z3poGdoomhujCNQtoQ80-BCO34GTOLb-/view?usp=sharing). The downloaded checkpoints of CoLMDriver should follow this structure:
```Shell
|--CoLMDriver
    |--ckpt
        |--colmdriver
            |--LLM
            |--perception
            |--VLM
            |--waypoints_planner
```

To download the checkpoints through command line and move them into the correct directories (no GUI required):
```Shell
# In CoLMDriver repository directory, with colmdriver conda env activated
pip install gdown
gdown 1z3poGdoomhujCNQtoQ80-BCO34GTOLb-

mkdir ckpt
mv colmdriver.zip ckpt
cd ckpt
unzip colmdriver.zip
rm colmdriver.zip

# Fix obsolete dataset dependency bug
sed -i "s|root_dir: .*|root_dir: $(pwd)|; s|test_dir: .*|test_dir: $(pwd)|; s|validate_dir: .*|validate_dir: $(pwd)|" colmdriver/percpetion/config.yaml
touch dataset_index.txt
```

**Step 2:** Running VLM, LLM (from repository root)
```Shell
#Enter conda ENV
conda activate vllm
# VLM on call
CUDA_VISIBLE_DEVICES=6 vllm serve ckpt/colmdriver/VLM --port 1111 --max-model-len 8192 --trust-remote-code --enable-prefix-caching

# LLM on call (in new terminal, with vllm env activated)
CUDA_VISIBLE_DEVICES=7 vllm serve ckpt/colmdriver/LLM --port 8888 --max-model-len 4096 --trust-remote-code --enable-prefix-caching
```
**Make sure that the CUDA_VISIBLE_DEVICES variable is set to a GPU available, which can be checked using the ```nvidia-smi``` command**

Note: make sure that the selected ports (1111,8888) are not occupied by other services. If you use other ports, please modify values of key 'comm_client' and 'vlm_client' in `simulation/leaderboard/team_code/agent_config/colmdriver_config.yaml` accordingly.

---

## Full Benchmark Evaluation (Internal)
This section is for internal/lab benchmark operations (manual evaluation workflow and submission verification).

### Evaluation Metrics
MDriveBench Leaderboard evaluates on two metrics:
1. **Driving Score (DS)**: Score of route completion adjusted by infraction penalties
2. **Success Rate (SR)**: The percentage of routes completed without failure.

#### Evaluation Scenarios
A full evaluation consists of three distinct benchmarks:
***OpenCDA (12 Scenarios):*** Uses ZIP-based scenario loading. Ensure all 12 ZIPs (including Scenes A, D, G, J) are in the `opencdascenarios/` folder.
***InterDrive (Full Suite):*** Cooperative driving evaluated via the `Interdrive_all` set.
***Safety-Critical:*** Pre-crash scenarios.

### Evaluation Workflow
Evaluation consists of 3 main phases: Submission Retrieval, Environment Setup, and Checkpoint Evaluation.

Before internal evaluation, ensure CARLA and required model-specific environments are prepared (see Quickstart and Baseline Evaluation Setup).

1. Verify CARLA 0.9.12 is installed and the egg is linked.
2. Ensure model-specific environments are functional (for CoLMDriver: `vllm` for inference and `colmdriver` for simulation).
3. Confirm model-specific dependencies are installed where required (for CoLMDriver/TCP/CoDriving stacks: `spconv` and `pypcd`).

#### 1. Submission Retrieval
To transfer participant submissions from Hugging Face to the lab's local evaluation server:

***Step A:*** Download and unzip the participant's `.zip` file from the submission portal into the `submissions/` directory.
```
unzip Team-A_submission.zip -d submissions/Team-A
```

***Step B:*** Verify structure. Ensure the unzipped folder contains the following files:
```
agents.py
config/
src/
weights/
model_env.yaml
```

***Step C:*** Symbolic linking. Point the evaluation suite to the new submission.
```
# Remove previous link and point to the current team
rm -rf simulation/leaderboard/team_code
ln -s ${PWD}/submissions/Team-A simulation/leaderboard/team_code
```

#### 2. Environment Setup
To prevent discrepancies caused by library version mismatches, build a fresh environment for every team.
```
# Build the team's specific environment
conda env create -f submissions/Test-Team/model_env.yaml -n mdrive_eval_test
conda activate mdrive_eval_test
```

#### 3. Checkpoint Evaluation
***Step A:*** Inject the standardized CARLA paths into the active team environment.
```
export CARLA_ROOT=${CARLA_ROOT:-$PWD/carla912}
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.12-py3.7-linux-x86_64.egg
```

***Step B:*** Running VLM, LLM (from repository root)
```
# Enter conda ENV
conda activate vllm
# VLM on call
CUDA_VISIBLE_DEVICES=6 vllm serve ckpt/colmdriver/VLM --port 1111 --max-model-len 8192 --trust-remote-code --enable-prefix-caching

# LLM on call (in new terminal, with vllm env activated)
CUDA_VISIBLE_DEVICES=7 vllm serve ckpt/colmdriver/LLM --port 8888 --max-model-len 4096 --trust-remote-code --enable-prefix-caching
```

Make sure `CUDA_VISIBLE_DEVICES` is set to an available GPU (`nvidia-smi`).
If you use ports other than `1111`/`8888`, update `comm_client` and `vlm_client` in `simulation/leaderboard/team_code/agent_config/colmdriver_config.yaml`.

***Step C:*** Run evaluation
```
# ==============================================================================
# BATCH 1: OpenCDA Scenarios (12 ZIPs)
# ==============================================================================
echo ">>> [BATCH 1/3] Running OpenCDA Scenarios..."
SCENARIO_DIR="opencdascenarios"
for zipfile in "$SCENARIO_DIR"/*.zip; do
    name=$(basename "$zipfile" .zip)
    $RUN_CMD tools/run_custom_eval.py \
      --zip "$zipfile" \
      --scenario-name "$name" \
      --results-tag "${name}_${TEAM_NAME}" \
      --agent "$SUB_DIR/agents.py" \
      --agent-config "$SUB_DIR/config/submission_config.yaml" \
      --port $PORT
done

# ==============================================================================
# BATCH 2: InterDrive Benchmark (Full Suite)
# ==============================================================================
echo ">>> [BATCH 2/3] Running InterDrive All..."
# Note: eval_mode.sh must be present in your scripts/eval directory
bash scripts/eval/eval_mode.sh $GPU $PORT $TEAM_NAME ideal Interdrive_all

# ==============================================================================
# BATCH 3: Warmup Scenarios
# ==============================================================================
echo ">>> [BATCH 3/3] Running Warmup Scenarios..."
$RUN_CMD tools/run_custom_eval.py \
    --routes-dir "warmupscenarios" \
    --agent "$SUB_DIR/agents.py" \
    --agent-config "$SUB_DIR/config/submission_config.yaml" \
    --port $PORT \
    --results-tag "warmup_${TEAM_NAME}"

echo "Evaluation Complete for $TEAM_NAME."
```

***Step D:*** Record DS and SR. Extract the Driving Score (DS) and Success Rate (SR) from the generated `summary.json`. Verify logs manually if the score is unexpectedly low to ensure no simulator glitches occurred.
