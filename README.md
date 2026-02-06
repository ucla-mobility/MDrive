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
- [Challenge Submission Instructions](#instructions)
- [Global Setup](#global-setup)
  - [General Setup](#general-setup)
  - [vLLM env](#vllm-env)
  - [CoLMDriver env](#colmdriver-env)
- [Full Benchmark Evaluation](#evaluation)
- [Baseline Evaluation Setup](#baseline-evaluation-setup)
  - [Evaluation of baselines](#evaluation-of-baselines)
  - [TCP Environment Setup](#tcp-environment-setup)
  - [CoDriving Environment Setup](#codriving-environment-setup)
  - [LMDrive Environment Setup](#lmdrive-environment-setup)
  - [UniAD Environment Setup](#uniad-environment-setup)
  - [VAD Environment Setup](#vad-environment-setup)
  - [CoLMDriver Model Setup](#colmdriver-model-setup)
- [Benchmark Evaluation on InterDrive](#benchmark-evaluation-on-interdrive)
- [LLM-Driven Scenario Generation](#llm-driven-scenario-generation)
- [Results Analysis and Visualization](#results-analysis-and-visualization)
  - [Results Analysis](#results-analysis)
  - [Visualizing Results](#visualizing-results)
- [Dataset](#dataset)
- [Training](#training)
  - [Perception module](#perception-module)
  - [Planning module](#planning-module)
  - [VLM planner](#vlm-planner)
- [Acknowledgements](#acknowledgements)

---
## MDriveBench Submission Instructions
To ensure your model is evaluated accurately, participants must submit a single .zip file containing the model, source code, and environment specifications.

### Required ZIP File Structure
Your ZIP file must be organized as follows:
```
team_name.zip
├── agents.py           # Main agent class (must inherit from TrackableAgent)
├── Dockerfile          # Highly recommended
├── model_env.yaml      # Conda/Pip environment specification (Fallback)
├── config/             # Folder containing submission_config.yaml and other configs
├── src/                # Folder containing model architecture & utilities
└── weights/            # Folder containing all trained checkpoints (.pth/.ckpt)
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
## Global Setup

### General Setup
Two environments are needed: 'vllm' for MLLMs inference and 'colmdriver' for simulation.

### vLLM env
#### Step 1: Install conda (if not installed already) 
```Shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

#### Step 2: Environment Creation and VLLM download 
```Shell
conda create -n vllm python=3.10
conda activate vllm
pip install vllm
```

### CoLMDriver env
#### Step 1: Basic Installation for colmdriver
Get code and create pytorch environment.
```Shell
git clone https://github.com/marco-cos/CoLMDriver.git
cd CoLMDriver

conda create --name colmdriver python=3.7 cmake=3.22.1
conda activate colmdriver
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install cudnn -c conda-forge

pip install -r opencood/requirements.txt
pip install -r simulation/requirements.txt
pip install openai
```

#### Step 2: Download and setup CARLA 0.9.10.1.
```Shell
chmod +x simulation/setup_carla.sh
./simulation/setup_carla.sh
easy_install carla/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
mkdir external_paths
ln -s ${PWD}/carla/ external_paths/carla_root
# If you already have a Carla, just create a soft link to external_paths/carla_root
```

The file structure should be:
```Shell
|--CoLMDriver
    |--external_paths
        |--carla_root
            |--CarlaUE4
            |--Co-Simulation
            |--Engine
            |--HDMaps
            |--Import
            |--PythonAPI
            |--Tools
            |--CarlaUE4.sh
            ...
```

Note: we choose the setuptools==41 to install because this version has the feature `easy_install`. After installing the carla.egg you can install the lastest setuptools to avoid No module named distutils_hack.

Steps 3,4,5 are for perception module.

#### Step 3: Install Spconv (1.2.1)
We use spconv 1.2.1 to generate voxel features in perception module.

To install spconv 1.2.1, please follow the guide in https://github.com/traveller59/spconv/tree/v1.2.1.

Or run the following commands:
```Shell
# 1. Activate your environment (if not activated already)
conda activate colmdriver

# 2. Install dependencies (these are all user-space)
conda install -y cmake=3.22.1 ninja boost ccache -c conda-forge
pip install pybind11 numpy

# 3. Clone spconv recursively (submodules required!)
git clone -b v1.2.1 --recursive https://github.com/traveller59/spconv.git
cd spconv

# 4. Build the wheel (will compile in your conda CUDA toolchain)
python setup.py bdist_wheel

# 5. Install the resulting .whl (no sudo needed)
pip install dist/spconv-1.2.1-*.whl

cd ..
```

#### Step 4: Set up
```Shell
# Set up
python setup.py develop

# Bbx IOU cuda version compile
python opencood/utils/setup.py build_ext --inplace 
```

#### Step 5: Install pypcd
```Shell
# go to another folder
cd ..
git clone https://github.com/klintan/pypcd.git
cd pypcd
pip install python-lzf
python setup.py install
cd ..
```

---

## Full Benchmark Evaluation
This section serves as the internal manual evaluation pipeline for MDriveBench Challenge. It provides the our lab with a standardized workflow to manually evaluate participant submissions and ensures that all results are benchmarked against the same hardware and software constraints.

### Evaluation Metrics
MDriveBench Leadboard evaluates on two metrics: 
1. **Driving Score (DS)**: Score of route completion adjusted by infraction penalties
2. **Sucess Rate (SR)**: The precentage of routes completed without failure.

#### Evaluation Scenarios
A full evaluation consists of three distinct benchmarks:
***OpenCDA (12 Scenarios):*** Uses ZIP-based scenario loading. Ensure all 12 ZIPs (including Scenes A, D, G, J) are in the opencdascenarios/ folder.
***InterDrive (Full Suite):*** Cooperative driving evaluated via the Interdrive_all set.
***Safety-Critical:*** Pre-crash scenarios.

### Evaluation Workflow
Evaluation consists of 3 main phases: Submission Retrieval, Environment Setup, and Checkpoint Evaluation.

Before evaluating any submission, ensure to follow the CoLMDriver Global Setup (https://github.com/marco-cos/CoLMDriver)

1. Verify CARLA 0.9.15 is installed and the egg is linked.
2. Ensure the vllm (for inference) and colmdriver (for simulation) environments are functional.
3. Confirm spconv (1.2.1) and pypcd are installed in the base environment, as many baselines and submissions rely on these for voxel feature generation.

#### 1. Submission Retrieval
To transfer participant submissions from  HuggingFace to the lab's local evaluation server:

***Step A:*** Download & Unzip Download the participant's .zip file from the submission portal into the submissions/ directory.
```
unzip Team-A_submission.zip -d submissions/Team-A
```

***Step B:*** Verify Structure Ensure the unzipped folder contains the following files:
```
agents.py
config/
src/
weights/
model_env.yaml
```

***Step C:*** Symbolic Linking Point the evaluation suite to the new submission.
```
# Remove previous link and point to the current team
rm -rf leaderboard/team_code
ln -s ${PWD}/submissions/Team-A leaderboard/team_code
```


#### 2. Enviroment Setup
To prevent discrepancies caused by library version mismatches, we build a fresh environment for every team.
```
# Build the team's specific environment
conda env create -f submissions/Test-Team/model_env.yaml -n mdrive_eval_test
conda activate mdrive_eval_test
```
#### 3. Checkpoint Evaluation
***Step A:*** Inject the standardized CARLA paths into the active team environment.
```
export CARLA_ROOT=/path/to/lab/CARLA_0.9.15
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
```
***Step B:*** Running VLM, LLM (from repository root)
```
#Enter conda ENV
conda activate vllm
# VLM on call
CUDA_VISIBLE_DEVICES=6 vllm serve ckpt/colmdriver/VLM --port 1111 --max-model-len 8192 --trust-remote-code --enable-prefix-caching

# LLM on call (in new terminal, with vllm env activated)
CUDA_VISIBLE_DEVICES=7 vllm serve ckpt/colmdriver/LLM --port 8888 --max-model-len 4096 --trust-remote-code --enable-prefix-caching
Make sure that the CUDA_VISIBLE_DEVICES variable is set to a GPU available, which can be checked using the nvidia-smi command

Note: make sure that the selected ports (1111,8888) are not occupied by other services. If you use other ports, please modify values of key 'comm_client' and 'vlm_client' in simulation/leaderboard/team_code/agent_config/colmdriver_config.yaml accordingly.
```

***Step C:*** Run Evaluation
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
# BATCH 3: Safety-Critical Scenarios (4 Routes)
# ==============================================================================
echo ">>> [BATCH 3/3] Running Safety-Critical Scenarios..."
$RUN_CMD tools/run_custom_eval.py \
    --agent "$SUB_DIR/agents.py" \
    --routes "data/warmup/safety_critical.xml" \
    --scenarios "data/warmup/safety_critical.json" \
    --port $PORT \
    --results-tag "safety_${TEAM_NAME}"

echo "Evaluation Complete for $TEAM_NAME."
```

***Step D:*** Record DS & SR Collect DS & SR Extract the Driving Score (DS) and Success Rate (SR) from the generated summary.json. Verify logs manually if the score is unexpectedly low to ensure no simulator glitches occurred.

### Proposed Automated Evaluation Script (Docker approach with Conda .yaml fallback)
The purpose of this evaluation script is that it automates the environment building and the simulation loop as described in the steps 2 and 3.
#### Evaluation Protocol
The evaluation script evaluate_all.sh follows a hybrid infrastructure:
Docker Execution: If the submission contains a Dockerfile, the evaluator builds a container to isolate the environment. This is the recommended method for models with complex dependencies (like VAD).
Conda Fallback: If no Dockerfile is detected, the script creates a Conda environment from model_env.yaml.
Hardware Mapping: Host GPUs are mapped directly to the evaluation process via CUDA_VISIBLE_DEVICES or the --gpus all Docker flag.

Save the script as ``` evaluate_all.sh ```
Make it executable:
```
chmod +x evaluate_all.sh
```
Run it:
```
./evaluate_all.sh
```

```
#!/bin/bash

# ==============================================================================
# MDriveBench Final Evaluation Script (Docker + Pre-setup Conda Fallback)
# Target Path: /data2/angela_test_envs/CoLMDriver
# ==============================================================================

TEAM_NAME=$1      # e.g., tcp, vad, colmdriver, or a new team name
SUBMISSION_ZIP=$2 
GPU=${3:-0}
PORT=2002 

# --- 1. Workspace Initialization ---
SUB_DIR="${PWD}/submissions/$TEAM_NAME"
mkdir -p "$SUB_DIR"
unzip -qo "$SUBMISSION_ZIP" -d "$SUB_DIR"

# --- 2. Environment Provisioning Logic ---
source "$(conda info --base)/etc/profile.d/conda.sh"

# PATH A: Docker Execution (If Dockerfile exists)
if [[ -f "$SUB_DIR/Dockerfile" ]]; then
    echo ">>> [ENV] Dockerfile detected. Building Image: mdrive_$TEAM_NAME"
    docker build -t "mdrive_$TEAM_NAME" "$SUB_DIR"
    RUN_CMD="docker run --rm --gpus all --net=host -v /path/to/lab/CARLA_0.9.10.1:/workspace/carla_root mdrive_$TEAM_NAME"

# PATH B: Pre-setup Lab Environments (Fallback 1)
else
    echo ">>> [ENV] No Dockerfile. Checking for pre-setup baseline in /data2..."
    case $TEAM_NAME in
      "tcp")        ENV_PATH="/data2/angela_test_envs/CoLMDriver/envs/tcp_codriving" ;;
      "vad")        ENV_PATH="/data2/angela_test_envs/CoLMDriver/envs/vad_env" ;;
      "colmdriver") ENV_PATH="/data2/angela_test_envs/CoLMDriver/envs/colmdriver" ;;
      "uniad")      ENV_PATH="/data2/angela_test_envs/CoLMDriver/envs/uniad_env" ;;
      "lmdrive")    ENV_PATH="/data2/angela_test_envs/CoLMDriver/envs/lmdrive" ;;
      *)            ENV_PATH="" ;; 
    esac

    if [[ -n "$ENV_PATH" && -d "$ENV_PATH" ]]; then
        echo ">>> [ENV] Activating pre-setup environment: $ENV_PATH"
        conda activate "$ENV_PATH"
        RUN_CMD="python"
    else
        # PATH C: Fresh Conda Build (Fallback 2)
        echo ">>> [ENV] New team detected. Building fresh Conda environment..."
        if ! conda info --envs | grep -q "mdrive_$TEAM_NAME"; then
            conda env create -f "$SUB_DIR/model_env.yaml" -n "mdrive_$TEAM_NAME"
        fi
        conda activate "mdrive_$TEAM_NAME"
        RUN_CMD="python"
    fi
fi

# --- 3. Global Paths & GPU ---
export CARLA_ROOT=/path/to/lab/CARLA_0.9.10.1
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export CUDA_VISIBLE_DEVICES=$GPU

# ==============================================================================
# BATCH 1: OpenCDA Scenarios (12 ZIPs)
# ==============================================================================

echo ">>> [BATCH 1/3] Running OpenCDA Scenarios from opencdascenarios/..."
for zipfile in opencdascenarios/*.zip; do
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
bash scripts/eval/eval_mode.sh $GPU $PORT $TEAM_NAME ideal Interdrive_all

# ==============================================================================
# BATCH 3: Safety-Critical Scenarios (4 Routes)
# ==============================================================================
echo ">>> [BATCH 3/3] Running Safety-Critical..."
$RUN_CMD tools/run_custom_eval.py \
    --agent "$SUB_DIR/agents.py" \
    --routes "data/warmup/safety_critical.xml" \
    --scenarios "data/warmup/safety_critical.json" \
    --port $PORT \
    --results-tag "safety_${TEAM_NAME}"

echo "Evaluation Complete for $TEAM_NAME."
```
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
export CARLA_ROOT=PATHTOYOURREPOROOT/CoLMDriver/external_paths/carla_root
export PYTHONPATH=$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
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
export CARLA_ROOT=PATHTOYOURREPOROOT/CoLMDriver/external_paths/carla_root
export PYTHONPATH=$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
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
export CARLA_ROOT=PATHTOYOURREPOROOT/CoLMDriver/external_paths/carla_root
export PYTHONPATH=$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
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
CUDA_VISIBLE_DEVICES=0 ./external_paths/carla_root/CarlaUE4.sh --world-port=2000 -prefer-nvidia
```

3. **Run VAD on Interdrive**
```bash
# CARLA must already be running on port 2000
bash scripts/eval/eval_mode.sh 0 2000 vad ideal Interdrive_all
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
#In CoLMDriver repostiory directory, with colmdriver conda env activated
pip install gdown
gdown 1z3poGdoomhujCNQtoQ80-BCO34GTOLb-

mkdir ckpt
mv colmdriver.zip ckpt
cd ckpt
unzip colmdriver.zip
rm colmdriver.zip

#Fix obsolete dataset dependancy bug
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
## Benchmark Evaluation on InterDrive

All models are evaluated on the InterDrive benchmark using a unified interface:

```bash
bash scripts/eval/eval_mode.sh <GPU_ID> <CARLA_PORT> <MODEL_NAME> <MODE> <SCENARIO_SET>
````

Where:

* `<MODEL_NAME>` ∈ `{ colmdriver, tcp, codriving, lmdrive, uniad, vad }`
* `<MODE>` ∈ `{ ideal, realtime }` (where supported)
* `<SCENARIO_SET>` ∈ `{ Interdrive_all, Interdrive_no_npc, Interdrive_npc }`

Make sure you have:

* The corresponding conda environment activated for each model (e.g., `tcp_codriving`, `lmdrive`, `uniad_env`, `colmdriver`, etc.)
* Any model-specific services running (e.g., VLM/LLM servers for CoLMDriver)

### Start CARLA

```bash
# Start CARLA server; change port if 2000 is already in use
CUDA_VISIBLE_DEVICES=0 ./external_paths/carla_root/CarlaUE4.sh --world-port=2000 -prefer-nvidia
```

If CARLA segfaults on startup, try:

```bash
conda install -c conda-forge libglvnd mesa-libgl-devel libegl libxrender libxext libxi
```

### Example evaluation commands

```bash
# TCP, full InterDrive
bash scripts/eval/eval_mode.sh 0 2000 tcp ideal Interdrive_all

# CoDriving, full InterDrive
bash scripts/eval/eval_mode.sh 0 2000 codriving ideal Interdrive_all

# LMDrive, full InterDrive
bash scripts/eval/eval_mode.sh 0 2000 lmdrive ideal Interdrive_all

# UniAD, full InterDrive
bash scripts/eval/eval_mode.sh 0 2000 uniad ideal Interdrive_all

# CoLMDriver: full benchmark, realtime mode, and subsets
bash scripts/eval/eval_mode.sh 0 2000 colmdriver ideal Interdrive_all
bash scripts/eval/eval_mode.sh 0 2000 colmdriver realtime Interdrive_all
bash scripts/eval/eval_mode.sh 0 2000 colmdriver ideal Interdrive_no_npc
bash scripts/eval/eval_mode.sh 0 2000 colmdriver ideal Interdrive_npc
```

Evaluation results are saved under:

```text
results/results_driving_<MODEL_NAME>
```

For example:

* `results/results_driving_colmdriver`
* `results/results_driving_tcp`
* `results/results_driving_lmdrive`

It’s recommended to run the LLM server, VLM server, CARLA server, and evaluation script in separate terminals.
CARLA processes may fail to stop cleanly; kill them manually if needed.

---

## LLM-Driven Scenario Generation

**TODO:** Add documentation for LLM-driven scenario generation, including:
- Natural-language specification of driving scenarios
- Conversion from language to executable scenarios
- Support for multi-agent negotiation and coordination behaviors

---

## Results Analysis and Visualization

### Results Analysis

The repository includes a comprehensive results analysis script that generates detailed reports, visualizations, and statistics about driving performance and negotiation behavior.

#### Basic Usage

```Shell
# Basic analysis of results directory
python visualization/results_analysis.py results/results_driving_colmdriver --output-dir report

# Multiple experiment folders
python visualization/results_analysis.py results/results_driving_colmdriver exp1 exp2 --output-dir report

# Generate single markdown report for multiple experiments
python visualization/results_analysis.py results/results_driving_colmdriver exp1 exp2 --output-dir report --markdown report/combined.md
```

#### Generated Analysis

The script generates:

- **Markdown Report**: Comprehensive analysis with embedded figures
- **CSV Data Tables**:
  - Per-route summary
  - Category summaries 
  - Negotiation statistics (scenario/agent/setting breakdowns)
  - Infractions breakdown
- **Visualizations**:
  - Driving scores by scenario category
  - Success rates across traffic conditions
  - NPC impact analysis
  - Negotiation frequency, rounds, and message-length distributions
  - Agent count distribution
  - Score distributions
  - Infractions breakdown
- **Artifacts**:
  - Text report summarizing negotiation behavior
  - Collected `nego.json` files copied into the output directory for easy sharing

#### Key Metrics Analyzed

- Driving Score (DS) and Success Rate
- Route Categories (IC/LM/LC) performance
- Impact of NPC traffic
- Negotiation behavior:
  - Frequency
  - Number of rounds
  - Consensus scores
  - Safety scores
- Agent counts and interactions
- Infractions breakdown

The analysis helps understand:
- How different traffic conditions affect performance
- Which scenarios trigger most negotiations
- How negotiation patterns vary across scenarios
- Where driving performance needs improvement

### Visualizing Results

The repository provides tools to generate videos from evaluation results:

```Shell
# Generate video for a specific scenario
python visualization/gen_video.py path/to/scenario/folder --output scenario.mp4

# Options:
--fps VALUE           Set video framerate (default: 10)
--width VALUE         Set output width in pixels
--height VALUE        Set output height in pixels
--font-scale VALUE    Adjust text overlay size
--min-hold VALUE     Minimum seconds to show overlay text

# Examples:
# Basic video with default settings
python visualization/gen_video.py results/results_driving_colmdriver/route_00/0000 --output route00_test.mp4

# High quality render with custom settings
python visualization/gen_video.py results/results_driving_colmdriver/route_00/0000 \
    --output route00_hq.mp4 \
    --fps 30 \
    --width 1920 \
    --height 1080 \
    --font-scale 1.2

# Process multiple scenarios
python visualization/gen_video.py results/results_driving_colmdriver/route_*/0000 \
    --output-dir videos/
```

Features:
- Multi-vehicle perspective rendering
- Negotiation overlay visualization
- Configurable resolution and framerate
- Automatic scenario discovery
- Progress tracking
- Font size and text display customization

---

## <span id="dataset"> Dataset
The dataset for training CoLMDriver is obtained from [V2Xverse](https://github.com/CollaborativePerception/V2Xverse), which contains experts behaviors in CARLA. You may get the dataset in two ways:
- Download from [this huggingface repository](https://huggingface.co/datasets/gjliu/V2Xverse).
- Generate the dataset by yourself, following this [guidance](https://github.com/CollaborativePerception/V2Xverse).

The dataset should be linked/stored under `external_paths/data_root/` follow this structure:
```Shell
|--data_root
    |--weather-0
        |--data
            |--routes_town{town_id}_{route_id}_w{weather_id}_{datetime}
                |--ego_vehicle_{vehicle_id}
                    |--2d_bbs_{direction}
                    |--3d_bbs
                    |--actors_data
                    |--affordances
                    |--bev_visibility
                    |--birdview
                    |--depth_{direction}
                    |--env_actors_data
                    |--lidar
                    |--lidar_semantic_front
                    |--measurements
                    |--rgb_{direction}
                    |--seg_{direction}
                    |--topdown
                |--rsu_{vehicle_id}
                |--log
            ...
```

## <span id="training"> Training

### Perception module
Our perception module follows [CoDriving](https://github.com/CollaborativePerception/V2Xverse).
To train perception module from scratch or a continued checkpoint, run the following commonds:
```Shell
# Single GPU training
python opencood/tools/train.py -y opencood/hypes_yaml/v2xverse/colmdriver_multiclass_config.yaml [--model_dir ${CHECKPOINT_FOLDER}]

# DDP training
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch  --nproc_per_node=2 --use_env opencood/tools/train_ddp.py -y opencood/hypes_yaml/v2xverse/colmdriver_multiclass_config.yaml [--model_dir ${CHECKPOINT_FOLDER}]

# Offline testing of perception
python opencood/tools/inference_multiclass.py --model_dir ${CHECKPOINT_FOLDER}
```
The training outputs can be found at `opencood/logs`.
Arguments Explanation:
- `model_dir` (optional) : the path of the checkpoints. This is used to fine-tune or continue-training. When the `model_dir` is
given, the trainer will discard the `hypes_yaml` and load the `config.yaml` in the checkpoint folder. In this case, ${CONFIG_FILE} can be `None`,
- `--nproc_per_node` indicate the GPU number you will use.

### Planning module
Given a checkpoint of perception module, we freeze its parameters and train the down-stream planning module in an end-to-end paradigm. The planner gets BEV perception feature and occupancy map as input and targets to predict the future waypoints of ego vehicle.

Train the planning module with a given perception checkpoint on multiple GPUs:
```Shell
# Train planner
bash scripts/train/train_planner_e2e.sh $GPU_ids $num_GPUs $perception_ckpt $planner_config $planner_ckpt_resume $name_of_log $save_path

# Example
bash scripts/train/train_planner_e2e.sh 0,1 2 ckpt/colmdriver/percpetion covlm_cmd_extend_adaptive_20 None log ./ckpt/colmdriver_planner

# Offline test
bash scripts/eval/eval_planner_e2e.sh 0,1 ckpt/colmdriver/percpetion covlm_cmd_extend_adaptive_20 ckpt/colmdriver/waypoints_planner/epoch_26.ckpt ./ckpt/colmdriver_planner
```

### VLM planner

#### Data generation

- Extract information from V2Xverse data (mentioned above): [MLLMs/data_transfer_sum.py](https://github.com/cxliu0314/CoLMDriver/blob/main/MLLMs/data_transfer_sum.py) 
- Generate json format training data: [MLLMs/data_transfer_query.py](https://github.com/cxliu0314/CoLMDriver/blob/main/MLLMs/data_transfer_query.py)

Our training data is also provided in [google drive](https://drive.google.com/file/d/1RH9iciUJ7fK5JpLSbYzCC_8Eb-hZnv9E/view?usp=sharing) for reference. Since the images are originated from local V2Xverse dataset, you still need to download the dataset to get full access.

#### Lora Finetuning

Using [ms-swift](https://github.com/modelscope/ms-swift) to finetune the MLLMs. Installation and details refer to the official repo. We provide an example script in [MLLMs/finetune.sh](https://github.com/cxliu0314/CoLMDriver/blob/main/MLLMs/finetune.sh)

---

## Acknowledgements
This implementation is based on code from several repositories.
- [V2Xverse](https://github.com/CollaborativePerception/V2Xverse)
- [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)
- [CoLMDriver](https://github.com/cxliu0314/CoLMDriver)
