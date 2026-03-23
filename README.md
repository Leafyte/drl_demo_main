# 🥊 Deep RL Stickman Fight
### A research demo for *"Deep Reinforcement Learning for Game AI: Adaptive NPC Behavior and Player-Centric Evaluation Frameworks"*

---

## What is this?

A two-file interactive demo where a **PPO-based AI agent** fights a **rule-based FSM opponent** as animated stickmen. The backend runs the game logic and computes all metrics. The frontend renders the fight on a canvas and displays live charts, math equations, and ablation controls.

---

## Files

```
stickfight/
├── backend.py   — Flask REST API (game engine + RL logic)
└── index.html   — Frontend (stickman canvas + dashboard)
```

---

## Setup & Run

### 1. Install dependencies

```bash
pip install flask flask-cors
```

### 2. Start the backend

```bash
python backend.py
```

You should see:
```
==================================================
  Deep RL Stickman Fight — Backend
  http://localhost:5009
==================================================
```

### 3. Open the frontend

Just open `index.html` in your browser — no server needed.

> ⚠️ Make sure the backend is running **before** opening the frontend, otherwise you'll see a warning in the event log.

---

## How to Use

| Page | What it does |
|------|-------------|
| **Overview** | Paper summary, key stats |
| **Fight Sim** | Watch PPO vs FSM stickman fight live |
| **Metrics** | Charts for win rate, entropy, action distribution, reward |
| **Math** | All 6 paper equations with values from the backend |
| **Compare** | Side-by-side table + 4 ablation toggles |

### Simulation controls

- **▶ Start** — begin the fight
- **■ Stop** — pause mid-episode
- **↺ Reset** — start a fresh episode
- **⚡ Batch ×20** — silently run 20 episodes to build up metric history
- **Speed slider** — 1× (slow) to 12× (fast)

---

## Agents

### PPO Agent (Blue)
Simulated Proximal Policy Optimization — no neural net, but faithfully mimics the adaptive behavior:
- Maintains action logits updated each step via a policy-gradient-like rule
- Context-sensitive biasing (aggressive when close, defensive when low HP)
- Entropy bonus adds exploration noise
- Potential-based reward shaping for faster convergence
- Actions: `move_toward`, `move_away`, `attack`, `dodge`, `defend`, `jump_attack`

### FSM Agent (Orange)
Deterministic finite state machine baseline:

```
PATROL → ENGAGE → ATTACK → RETREAT
```

- `PATROL` — wander when enemy is far
- `ENGAGE` — move toward enemy when within 200px
- `ATTACK` — attack or occasionally defend when within range
- `RETREAT` — defend when HP drops below 20

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/reset` | Start a new episode |
| `POST` | `/api/step` | Advance one game step |
| `GET` | `/api/metrics` | Win rates, entropy, action distributions |
| `POST` | `/api/batch` | Run N silent episodes (`{"n": 20}`) |
| `GET` | `/api/equations` | Computed math values for the Math page |
| `GET` | `/api/ablation` | Get current ablation config |
| `POST` | `/api/ablation` | Toggle a setting (`{"key": "entropy_bonus"}`) |

---

## Ablation Settings

Toggle these on the Compare page (also hits the backend):

| Key | Effect when OFF |
|-----|----------------|
| `entropy_bonus` | No exploration noise → lower strategy diversity |
| `reward_shaping` | No distance potential → slower convergence |
| `partial_obs` | Full arena visibility → slightly higher win rate |
| `action_masking` | Allows attacking out of range → ~8% wasted actions |

---

## Metrics Explained

| Metric | Formula | Description |
|--------|---------|-------------|
| **Strategy Entropy** | `H(π) = -Σ π(a\|s) log₂ π(a\|s)` | Action diversity in bits (max 2.58 for 6 actions) |
| **Engagement Index** | `EI = 0.4·(D/D_max) + 0.35·H(π)/H_max + 0.25·WR` | Composite player experience score |
| **Discounted Return** | `G_t = Σ γᵏ · r_{t+k}` | Cumulative future reward (γ = 0.99) |
| **PPO Clip Loss** | `L = 𝔼[min(r·Â, clip(r,1-ε,1+ε)·Â)]` | Prevents destructive policy updates (ε = 0.2) |

---

## Changing the Port

Two places to update:

**`backend.py`** — last line:
```python
app.run(debug=True, port=5009)  # ← change here
```

**`index.html`** — top of the `<script>` block:
```javascript
const API = 'http://localhost:5009/api';  // ← change here
```

---

## Requirements

- Python 3.8+
- flask
- flask-cors
- Any modern browser (Chrome, Firefox, Edge, Safari)
