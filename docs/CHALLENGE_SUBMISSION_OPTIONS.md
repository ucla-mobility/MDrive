# Challenge Submission Format: Options & Trade-offs

**Purpose:** This document outlines two approaches for accepting model submissions in our driving challenge, with detailed analysis of each.

---

## Table of Contents
1. [The Core Problem](#the-core-problem)
2. [Option A: Docker-Based Submissions](#option-a-docker-based-submissions)
3. [Option B: Fixed Environment Submissions](#option-b-fixed-environment-submissions)
4. [Side-by-Side Comparison](#side-by-side-comparison)
5. [Recommendations](#recommendations)
6. [Decision Points](#decision-points)

---

## The Core Problem

### Why We Can't "Just Run" Submitted Models

Different autonomous driving models have incompatible requirements:

| Model | Python | PyTorch | CUDA | Special Dependencies |
|-------|--------|---------|------|---------------------|
| TCP | 3.8 | 1.9+ | 11.x | Minimal |
| VAD | 3.8 | 1.9 | 11.x | mmcv, mmdet (specific versions) |
| UniAD | 3.8 | 1.9 | 11.x | Custom CUDA ops, mmcv, mmdet3d |
| LMDrive | 3.8+ | 2.0 | 11.x | transformers, LLM libs |

**Key issues:**
- `mmcv==1.7.0` might conflict with `mmcv==2.0.0`
- Custom CUDA operations compiled for CUDA 11.3 may not work on 11.8
- Different models may require mutually exclusive package versions

**Bottom line:** We cannot `pip install` everyone's dependencies into one environment.

---

## Option A: Docker-Based Submissions

### How It Works

CARLA uses a **client-server architecture** over TCP:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Our Infrastructure                           │
│                                                                      │
│   ┌───────────────────┐              ┌───────────────────────────┐  │
│   │   CARLA Server    │              │   Docker Container        │  │
│   │   (Native)        │◄────TCP─────►│   (Team's Submission)     │  │
│   │                   │    :2000     │                           │  │
│   │   • Runs natively │              │   • Team's Python env     │  │
│   │   • GPU rendering │              │   • Team's dependencies   │  │
│   │   • Physics sim   │              │   • Team's model weights  │  │
│   │                   │              │   • Team's agent code     │  │
│   └───────────────────┘              └───────────────────────────┘  │
│           │                                      │                   │
│           ▼                                      ▼                   │
│   ┌───────────────────┐              ┌───────────────────────────┐  │
│   │   Our Eval        │              │   Their Agent             │  │
│   │   Harness         │──scenarios──►│   • sensors()             │  │
│   │   • Load routes   │              │   • run_step()            │  │
│   │   • Score results │◄──controls───│   • Returns VehicleControl│  │
│   └───────────────────┘              └───────────────────────────┘  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Key insight:** CARLA server and Python client communicate over network sockets. The client can run inside a Docker container while CARLA runs natively. We do NOT need to containerize CARLA.

### What Teams Submit

```
submission/
├── Dockerfile              # Builds their environment
├── agent.py                # Implements AutonomousAgent interface
├── config.yaml             # Agent configuration
├── requirements.txt        # Their Python dependencies
├── src/                    # Their model code
│   ├── model.py
│   └── ...
└── weights/                # Model checkpoints (can be large: 1-20GB)
    └── model.ckpt
```

### Example Dockerfile (Team Provides)

```dockerfile
# Teams build FROM a base image we provide
FROM our_challenge/base:v1

# Install their specific dependencies
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

# Copy their code
COPY src/ /app/src/
COPY agent.py /app/
COPY config.yaml /app/
COPY weights/ /app/weights/

# Set environment variables for our harness to find their agent
ENV TEAM_AGENT=/app/agent.py
ENV TEAM_CONFIG=/app/config.yaml

# Entry point (our evaluation script, included in base image)
ENTRYPOINT ["python", "/eval/run_evaluation.py"]
```

### How We Run It

```bash
# Step 1: Start CARLA server (native, as we already do)
./CarlaUE4.sh -RenderOffScreen -carla-port=2000

# Step 2: Run team's Docker container
docker run \
    --gpus all \                          # GPU access for their model
    --network=host \                       # Access to localhost:2000 (CARLA)
    -v $(pwd)/scenarios:/scenarios:ro \    # Mount our scenarios (read-only)
    -v $(pwd)/results:/results \           # Mount results directory
    team_submission:latest \
    --routes /scenarios/test_route.xml \
    --output /results/team_results.json
```

### What We Provide

1. **Base Docker Image** containing:
   - Ubuntu 20.04
   - CUDA 11.8 runtime
   - Python 3.8
   - CARLA Python client library (matching our CARLA version)
   - Common packages (numpy, opencv, pytorch base)
   - Our evaluation harness script

2. **Evaluation Harness** that:
   - Loads their agent dynamically
   - Connects to CARLA
   - Runs scenarios
   - Collects results

3. **Documentation** on:
   - How to build a valid Docker image
   - The agent interface they must implement
   - How to test locally before submitting

### Pros & Cons

| Pros | Cons |
|------|------|
| ✅ **Complete isolation** - no dependency conflicts | ❌ Teams must learn Docker basics |
| ✅ **Any dependencies** - teams use whatever they need | ❌ Container images can be large (10-50GB) |
| ✅ **Reproducible** - same container = same behavior | ❌ Storage/transfer overhead for images |
| ✅ **Minimal work for us** - just `docker run` | ❌ Need to build & maintain base image |
| ✅ **Industry standard** - CARLA Leaderboard uses this | ❌ Debugging is harder for teams |
| ✅ **Scales to many submissions** | |

---

## Option B: Fixed Environment Submissions

### How It Works

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Our Infrastructure                            │
│                                                                       │
│   ┌───────────────────┐       ┌─────────────────────────────────────┐│
│   │   CARLA Server    │       │   Our Python Environment            ││
│   │   (Native)        │◄─────►│   (Fixed: Python 3.8, PyTorch 2.0)  ││
│   │                   │       │                                     ││
│   │                   │       │   ┌─────────────────────────────┐   ││
│   │                   │       │   │ Team's Submission           │   ││
│   │                   │       │   │ • agent.py                  │   ││
│   │                   │       │   │ • config.yaml               │   ││
│   │                   │       │   │ • src/ (their code)         │   ││
│   │                   │       │   │ • weights/                  │   ││
│   │                   │       │   └─────────────────────────────┘   ││
│   └───────────────────┘       └─────────────────────────────────────┘│
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

**Key insight:** Everyone uses the same Python environment. Teams must make their code work within our constraints.

### What Teams Submit

```
submission/
├── agent.py                # Implements AutonomousAgent interface
├── config.yaml             # Agent configuration  
├── src/                    # Their model code (must work in our env)
│   ├── model.py
│   └── ...
└── weights/                # Model checkpoints
    └── model.ckpt
```

**No Dockerfile.** Their code runs directly in our environment.

### What We Provide

1. **Exact Environment Specification:**
   ```yaml
   # environment.yaml - Teams must target this exactly
   name: challenge_env
   channels:
     - pytorch
     - nvidia
     - conda-forge
   dependencies:
     - python=3.8.18
     - pytorch=2.0.1
     - torchvision=0.15.2
     - cuda-toolkit=11.8
     - numpy=1.24.3
     - opencv=4.8.0
     - scipy=1.10.1
     - pip:
       - carla==0.9.15
       - pygame==2.5.0
       - networkx==3.1
       # ... complete list of allowed packages
   ```

2. **Clear Interface Definition:**
   ```python
   from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
   
   def get_entry_point():
       """Must return the class name of your agent."""
       return 'YourAgentClassName'
   
   class YourAgentClassName(AutonomousAgent):
       def setup(self, path_to_conf_file):
           """Initialize your model. Called once at start."""
           self.track = Track.SENSORS
           # Load your model weights here
           
       def sensors(self):
           """Define sensors your agent needs."""
           return [
               {'type': 'sensor.camera.rgb', 'id': 'front', ...},
               # Only standard CARLA sensors allowed
           ]
       
       def run_step(self, input_data, timestamp):
           """Execute one step. Called every simulation tick."""
           # input_data contains sensor readings
           # Return carla.VehicleControl
           control = carla.VehicleControl()
           control.steer = ...
           control.throttle = ...
           control.brake = ...
           return control
   ```

3. **Validation Script** teams run locally to check compatibility:
   ```bash
   python validate_submission.py --submission-dir ./my_submission/
   # Checks: imports work, interface correct, no banned packages
   ```

### How We Run It

```bash
# Activate our fixed environment
conda activate challenge_env

# Run evaluation with their submission
python tools/run_custom_eval.py \
    --agent submissions/team_A/agent.py \
    --agent-config submissions/team_A/config.yaml \
    --routes scenarios/test_scenario/ \
    --output results/team_A/
```

### The Compatibility Problem

**Many existing models WON'T work without modification:**

| Model | Issue | Can It Work? |
|-------|-------|--------------|
| TCP | Minimal dependencies | ✅ Likely works |
| VAD | Requires mmcv/mmdet specific versions | ⚠️ Maybe, if versions align |
| UniAD | Custom CUDA ops, heavy mmcv deps | ❌ Probably not without rewrite |
| LMDrive | LLM dependencies | ⚠️ Depends on transformers version |

Teams would need to **adapt their code** to work in our environment, which is significant effort.

### Pros & Cons

| Pros | Cons |
|------|------|
| ✅ **Simple for us** - just run Python | ❌ **Many models won't work** without adaptation |
| ✅ **Simple for teams** - no Docker knowledge | ❌ Teams must adapt to our exact versions |
| ✅ **Smaller submissions** - no container overhead | ❌ Limits what models can participate |
| ✅ **Easier debugging** | ❌ Dependency version conflicts likely |
| ✅ **Faster iteration** | ❌ We may need to add packages on request |

---

## Side-by-Side Comparison

| Aspect | Docker (Option A) | Fixed Env (Option B) |
|--------|-------------------|----------------------|
| **Team effort** | Build Docker image | Adapt code to our env |
| **Our effort** | Maintain base image, run containers | Maintain env, handle package requests |
| **Compatibility** | Any model works | Only compatible models work |
| **Submission size** | Large (10-50GB images) | Small (code + weights only) |
| **Debugging** | Harder (container isolation) | Easier (same env) |
| **Scalability** | Excellent | Good |
| **Industry standard** | Yes (CARLA Leaderboard, Waymo, nuScenes) | Less common |
| **Barrier to entry** | Medium (Docker knowledge) | Medium (adaptation effort) |

---

## Recommendations

### For an Open Public Challenge
**Use Docker (Option A).**

Reasoning:
- Maximizes participation (any model can work)
- Industry standard, teams may already have Docker experience
- Scales to many submissions with minimal per-submission effort
- One-time setup cost for base image pays off

### For a Research Paper Benchmark
**Either option works, or even manual integration.**

If we're evaluating a fixed set of known models (TCP, VAD, UniAD, etc.):
- We control which models to include
- We do the integration work ourselves
- No "submission format" needed - we just run what we've integrated

### Hybrid Approach
We could support both:
1. **Preferred:** Docker submissions
2. **Fallback:** Fixed environment for simple models

---

## Decision Points

Before finalizing, we need to decide:

### 1. Scope
- Is this an **open challenge** (unknown teams submit)?
- Or a **benchmark paper** (we evaluate known models)?

### 2. Scale
- How many submissions do we expect?
  - 5-10 → Manual integration or fixed env is fine
  - 50+ → Docker is necessary

### 3. Timeline
- When do we need this ready?
  - Docker: Need to build base image, write docs, test workflow
  - Fixed Env: Need to finalize environment, create validation tools

### 4. Model Weights
- How will teams transfer large model weights (often 5-20GB)?
  - Cloud storage link?
  - Direct upload?
  - Include in Docker image?

### 5. CARLA Version
- What exact CARLA version are we committing to?
- This is CRITICAL - teams need this to build compatible agents

### 6. Testing
- How do teams verify their submission works before submitting?
  - Do we provide test scenarios?
  - Do we provide a minimal CARLA setup guide?

---

## Appendix: Technical Details

### Our Current CARLA Setup

```bash
# Check our CARLA version
cat external_paths/carla_root  # Points to CARLA installation

# How we currently run evaluations
python tools/run_custom_eval.py \
    --planner tcp \
    --zip scenarios/test.zip \
    --port 2000
```

### The Agent Interface (What Teams Implement)

Located at: `simulation/leaderboard/leaderboard/autoagents/autonomous_agent.py`

Key methods:
- `setup(path_to_conf_file)` - Initialize model
- `sensors()` - Define required sensors
- `run_step(input_data, timestamp)` - Return `carla.VehicleControl`

### Currently Integrated Models

| Model | Agent File | Config |
|-------|-----------|--------|
| TCP | `team_code/tcp_agent.py` | `agent_config/tcp_5_10_config.yaml` |
| VAD | `team_code/vad_b2d_agent.py` | `agent_config/pnp_config_vad.yaml` |
| UniAD | `team_code/uniad_b2d_agent.py` | `agent_config/uniad.yaml` |
| LMDrive | `team_code/lmdriver_agent.py` | `agent_config/lmdriver_config_8_10.py` |
| CoLMDriver | `team_code/colmdriver_agent.py` | `agent_config/colmdriver_config.yaml` |

These were **manually integrated** by adapting each model's codebase to our stack.

---

## Questions?

Contact: [Your contact info]

Last updated: January 2026
