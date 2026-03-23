"""
Deep RL Game AI — Stickman Fight Backend
=========================================
Run:  python backend.py
Then: open index.html in browser (or serve with live-server)

Endpoints:
  POST /api/step          — advance simulation one step
  POST /api/reset         — reset a new episode
  GET  /api/metrics       — get full metrics history
  POST /api/batch         — run N episodes silently, return stats
  GET  /api/ablation      — current ablation config
  POST /api/ablation      — toggle an ablation setting
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import random
import math
import time

app = Flask(__name__)
CORS(app)
@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")
# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
ARENA_W        = 900   # logical arena width  (pixels)
ARENA_H        = 400   # logical arena height
GROUND_Y       = 340   # y where stickmen stand
MAX_HP         = 100
MAX_STEPS      = 300
ATTACK_RANGE   = 80    # px — close enough to land a hit
MOVE_SPEED     = 14    # px per step
GAMMA          = 0.99
EPSILON_CLIP   = 0.2

ACTIONS = ["move_toward", "move_away", "attack", "dodge", "defend", "jump_attack"]

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def dist(a, b):
    return abs(a["x"] - b["x"])

def entropy(probs):
    """Shannon entropy in bits."""
    h = 0.0
    for p in probs:
        if p > 1e-9:
            h -= p * math.log2(p)
    return round(h, 4)

def softmax(logits):
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]

def sample_action(probs):
    r = random.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            return i
    return len(probs) - 1

def discounted_return(rewards, gamma=GAMMA):
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
    return round(G, 3)

def ppo_clip(ratio, advantage, eps=EPSILON_CLIP):
    clipped = clamp(ratio, 1 - eps, 1 + eps) * advantage
    unclipped = ratio * advantage
    return round(min(unclipped, clipped), 4)

# ─────────────────────────────────────────────
#  ABLATION STATE
# ─────────────────────────────────────────────
ablation = {
    "entropy_bonus":   True,
    "reward_shaping":  True,
    "partial_obs":     True,
    "action_masking":  False,
}

# ─────────────────────────────────────────────
#  PPO AGENT
# ─────────────────────────────────────────────
class PPOAgent:
    """
    Simulated PPO agent.
    Weights are updated each step using a lightweight
    policy-gradient-like rule — no neural net needed.
    """
    def __init__(self):
        # Logits per action (initialised slightly varied)
        self.logits   = [0.5, -0.3, 0.8, 0.2, 0.1, 0.3]
        self.old_probs = softmax(self.logits[:])
        self.action_counts = {a: 0 for a in ACTIONS}
        self.cumulative_reward = 0.0
        self.reward_history    = []
        self.entropy_history   = []
        self.last_action_idx   = 0
        self.last_probs        = softmax(self.logits[:])
        self.pos   = {"x": 160, "y": GROUND_Y}
        self.hp    = MAX_HP
        self.state = "idle"   # idle | walk | attack | defend | dodge | jump
        self.facing = 1       # 1 = right, -1 = left
        self.anim_frame = 0
        self.hit_flash  = 0   # countdown frames for red flash

    def get_probs(self, obs):
        """
        Context-sensitive logit adjustment — simulates what
        a trained policy would have learned.
        """
        logits = self.logits[:]
        d = obs["dist"]
        my_hp    = obs["my_hp"]
        enemy_hp = obs["enemy_hp"]

        # Close → prefer attack / jump_attack
        if d < ATTACK_RANGE:
            logits[2] += 1.2   # attack
            logits[5] += 0.7   # jump_attack
            logits[0] -= 0.4   # move_toward less useful
        # Far → move in
        elif d > 200:
            logits[0] += 1.0   # move_toward
            logits[2] -= 0.6

        # Low HP → defend / dodge
        if my_hp < 30:
            logits[3] += 1.1   # dodge
            logits[4] += 0.9   # defend
            logits[2] -= 0.5

        # Enemy low HP → go aggressive
        if enemy_hp < 30:
            logits[2] += 0.8
            logits[5] += 0.5

        # Entropy bonus: add exploration noise if enabled
        if ablation["entropy_bonus"]:
            logits = [l + random.gauss(0, 0.15) for l in logits]

        # Action masking: if out of range, mask attack
        if ablation["action_masking"] and d > ATTACK_RANGE + 20:
            logits[2] = -9.0
            logits[5] = -9.0

        return softmax(logits)

    def select_action(self, obs):
        probs = self.get_probs(obs)
        self.last_probs = probs
        idx = sample_action(probs)
        self.last_action_idx = idx
        self.action_counts[ACTIONS[idx]] += 1
        return ACTIONS[idx], probs

    def update_weights(self, reward, advantage=None):
        """Simulated PPO weight update."""
        if advantage is None:
            advantage = reward  # simplified
        lr = 0.04
        ratio = self.last_probs[self.last_action_idx] / max(
            self.old_probs[self.last_action_idx], 1e-8)
        loss = ppo_clip(ratio, advantage)

        # Gradient-like nudge
        for i in range(len(self.logits)):
            if i == self.last_action_idx:
                self.logits[i] += lr * loss
            else:
                self.logits[i] -= lr * loss * 0.1
        # Soft-clip logits
        self.logits = [clamp(l, -3.0, 3.0) for l in self.logits]
        self.old_probs = self.last_probs[:]

        self.cumulative_reward += reward
        self.reward_history.append(round(reward, 3))
        ent = entropy(self.last_probs)
        self.entropy_history.append(ent)

    def current_entropy(self):
        return entropy(self.last_probs)


# ─────────────────────────────────────────────
#  FSM AGENT
# ─────────────────────────────────────────────
class FSMAgent:
    """
    Deterministic finite state machine baseline.
    States: PATROL → ENGAGE → ATTACK → RETREAT
    """
    def __init__(self):
        self.action_counts = {a: 0 for a in ACTIONS}
        self.cumulative_reward = 0.0
        self.reward_history    = []
        self.entropy_history   = []
        self.fsm_state  = "PATROL"
        self.pos    = {"x": ARENA_W - 160, "y": GROUND_Y}
        self.hp     = MAX_HP
        self.state  = "idle"
        self.facing = -1
        self.anim_frame = 0
        self.hit_flash  = 0
        self.patrol_dir = -1
        self.patrol_steps = 0
        self.last_action = "idle"

    def _fixed_probs(self, action_name):
        """FSM always commits fully — very low entropy."""
        probs = [0.02] * len(ACTIONS)
        idx = ACTIONS.index(action_name)
        probs[idx] = 0.90
        # distribute remaining 0.10
        for i in range(len(probs)):
            if i != idx:
                probs[i] = 0.10 / (len(ACTIONS) - 1)
        return probs

    def select_action(self, obs):
        d      = obs["dist"]
        my_hp  = obs["my_hp"]

        # FSM transitions
        if my_hp < 20:
            self.fsm_state = "RETREAT"
        elif d < ATTACK_RANGE:
            self.fsm_state = "ATTACK"
        elif d < 200:
            self.fsm_state = "ENGAGE"
        else:
            self.fsm_state = "PATROL"

        if self.fsm_state == "PATROL":
            action = "move_away"
            self.patrol_steps += 1
            if self.patrol_steps > 30:
                self.patrol_dir  *= -1
                self.patrol_steps = 0
        elif self.fsm_state == "ENGAGE":
            action = "move_toward"
        elif self.fsm_state == "ATTACK":
            action = "attack" if random.random() > 0.15 else "defend"
        else:  # RETREAT
            action = "defend"

        self.action_counts[action] += 1
        self.last_action = action
        probs = self._fixed_probs(action)
        self.entropy_history.append(entropy(probs))
        return action, probs

    def update_reward(self, reward):
        self.cumulative_reward += reward
        self.reward_history.append(round(reward, 3))

    def current_entropy(self):
        if self.entropy_history:
            return self.entropy_history[-1]
        return 0.0


# ─────────────────────────────────────────────
#  GAME STATE
# ─────────────────────────────────────────────
class GameState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.ppo = PPOAgent()
        self.fsm = FSMAgent()
        self.step    = 0
        self.done    = False
        self.winner  = None
        self.events  = []   # list of strings for the log
        self.episode_rewards_ppo = []
        self.episode_rewards_fsm = []

    def _obs(self, me, enemy):
        return {
            "dist":     dist(me.pos, enemy.pos),
            "my_hp":    me.hp,
            "enemy_hp": enemy.hp,
            "my_x":     me.pos["x"],
            "enemy_x":  enemy.pos["x"],
            "step":     self.step,
        }

    def _apply_action(self, agent, action, enemy, is_ppo=True):
        """Mutate agent/enemy positions & HP; return reward."""
        reward  = 0.0
        d       = dist(agent.pos, enemy.pos)
        dx      = 1 if enemy.pos["x"] > agent.pos["x"] else -1
        agent.facing = dx
        agent.hit_flash = max(0, agent.hit_flash - 1)

        old_dist = d

        if action == "move_toward":
            agent.pos["x"] = clamp(agent.pos["x"] + dx * MOVE_SPEED, 40, ARENA_W - 40)
            agent.state = "walk"
            new_dist = dist(agent.pos, enemy.pos)
            if ablation["reward_shaping"]:
                reward += (old_dist - new_dist) * 0.04   # potential shaping
            reward += 0.1

        elif action == "move_away":
            agent.pos["x"] = clamp(agent.pos["x"] - dx * MOVE_SPEED, 40, ARENA_W - 40)
            agent.state = "walk"
            reward -= 0.05

        elif action == "attack":
            agent.state = "attack"
            if d <= ATTACK_RANGE:
                # Blocked by defend?
                if enemy.state == "defend":
                    dmg = random.randint(3, 8)
                    enemy.hit_flash = 6
                    enemy.hp = clamp(enemy.hp - dmg, 0, MAX_HP)
                    reward += 0.3
                    self.events.append(f"{'PPO' if is_ppo else 'FSM'} attack BLOCKED — {dmg} dmg")
                else:
                    dodge_evade = (enemy.state == "dodge" and random.random() > 0.45)
                    if dodge_evade:
                        reward -= 0.1
                        self.events.append(f"{'PPO' if is_ppo else 'FSM'} attack DODGED!")
                    else:
                        dmg = random.randint(12, 22)
                        enemy.hit_flash = 8
                        enemy.hp = clamp(enemy.hp - dmg, 0, MAX_HP)
                        reward += 1.5 + (dmg / 20.0)
                        self.events.append(f"{'PPO' if is_ppo else 'FSM'} lands HIT — {dmg} dmg")
            else:
                reward -= 0.2   # whiff
                self.events.append(f"{'PPO' if is_ppo else 'FSM'} swings — out of range")

        elif action == "jump_attack":
            agent.state = "jump"
            if d <= ATTACK_RANGE + 30:
                dmg = random.randint(16, 28)
                if enemy.state != "defend":
                    enemy.hit_flash = 10
                    enemy.hp = clamp(enemy.hp - dmg, 0, MAX_HP)
                    reward += 2.0
                    self.events.append(f"{'PPO' if is_ppo else 'FSM'} JUMP ATTACK — {dmg} dmg!")
                else:
                    dmg2 = dmg // 3
                    enemy.hp = clamp(enemy.hp - dmg2, 0, MAX_HP)
                    reward += 0.5
            else:
                reward -= 0.1

        elif action == "dodge":
            agent.state = "dodge"
            agent.pos["x"] = clamp(agent.pos["x"] - dx * 20, 40, ARENA_W - 40)
            reward += 0.15

        elif action == "defend":
            agent.state = "defend"
            reward += 0.1

        # Survival reward
        reward += 0.05
        return round(reward, 3)

    def step_sim(self):
        if self.done:
            return self.to_dict()

        ppo, fsm = self.ppo, self.fsm
        self.events = []

        obs_ppo = self._obs(ppo, fsm)
        obs_fsm = self._obs(fsm, ppo)

        ppo_action, ppo_probs = ppo.select_action(obs_ppo)
        fsm_action, fsm_probs = fsm.select_action(obs_fsm)

        ppo_reward = self._apply_action(ppo, ppo_action, fsm, is_ppo=True)
        fsm_reward = self._apply_action(fsm, fsm_action, ppo, is_ppo=False)

        ppo.update_weights(ppo_reward)
        fsm.update_reward(fsm_reward)

        self.episode_rewards_ppo.append(ppo_reward)
        self.episode_rewards_fsm.append(fsm_reward)

        self.step += 1

        # Idle reset after action frame
        if ppo.anim_frame > 0:
            ppo.anim_frame -= 1
        else:
            if ppo.state not in ("walk",):
                ppo.state = "idle"
        if fsm.anim_frame > 0:
            fsm.anim_frame -= 1
        else:
            if fsm.state not in ("walk",):
                fsm.state = "idle"

        # Check termination
        if ppo.hp <= 0 or fsm.hp <= 0 or self.step >= MAX_STEPS:
            self.done = True
            if ppo.hp > fsm.hp:
                self.winner = "PPO"
            elif fsm.hp > ppo.hp:
                self.winner = "FSM"
            else:
                self.winner = "DRAW"
            self.events.append(f"Episode over — Winner: {self.winner}")

        return self.to_dict()

    def to_dict(self):
        ppo, fsm = self.ppo, self.fsm
        return {
            "step":    self.step,
            "done":    self.done,
            "winner":  self.winner,
            "events":  self.events[-4:],
            "ppo": {
                "hp":        ppo.hp,
                "x":         ppo.pos["x"],
                "y":         ppo.pos["y"],
                "facing":    ppo.facing,
                "state":     ppo.state,
                "hit_flash": ppo.hit_flash,
                "action":    ACTIONS[ppo.last_action_idx],
                "probs":     [round(p, 3) for p in ppo.last_probs],
                "entropy":   ppo.current_entropy(),
                "reward":    round(ppo.cumulative_reward, 2),
            },
            "fsm": {
                "hp":        fsm.hp,
                "x":         fsm.pos["x"],
                "y":         fsm.pos["y"],
                "facing":    fsm.facing,
                "state":     fsm.state,
                "hit_flash": fsm.hit_flash,
                "action":    fsm.last_action if fsm.last_action else "idle",
                "probs":     [round(p, 3) for p in (fsm.entropy_history and [0.02]*6 or [0.02]*6)],
                "entropy":   fsm.current_entropy(),
                "reward":    round(fsm.cumulative_reward, 2),
                "fsm_state": fsm.fsm_state,
            },
            "ablation": ablation,
        }


# ─────────────────────────────────────────────
#  GLOBAL GAME + METRICS STORE
# ─────────────────────────────────────────────
game = GameState()

metrics_store = {
    "episodes": [],        # list of episode summaries
    "ppo_wins": 0,
    "fsm_wins": 0,
    "draws":    0,
    "total":    0,
}

def record_episode(g: GameState):
    ep = {
        "ep_num":     metrics_store["total"] + 1,
        "winner":     g.winner,
        "steps":      g.step,
        "ppo_hp":     g.ppo.hp,
        "fsm_hp":     g.fsm.hp,
        "ppo_reward": round(g.ppo.cumulative_reward, 2),
        "fsm_reward": round(g.fsm.cumulative_reward, 2),
        "ppo_entropy": round(sum(g.ppo.entropy_history) / max(len(g.ppo.entropy_history), 1), 3),
        "fsm_entropy": round(sum(g.fsm.entropy_history) / max(len(g.fsm.entropy_history), 1), 3),
        "ppo_Gt":     discounted_return(g.episode_rewards_ppo),
        "fsm_Gt":     discounted_return(g.episode_rewards_fsm),
        "ppo_actions": g.ppo.action_counts,
        "fsm_actions": g.fsm.action_counts,
    }
    if g.winner == "PPO":   metrics_store["ppo_wins"] += 1
    elif g.winner == "FSM": metrics_store["fsm_wins"] += 1
    else:                   metrics_store["draws"]    += 1
    metrics_store["total"] += 1
    metrics_store["episodes"].append(ep)
    if len(metrics_store["episodes"]) > 200:
        metrics_store["episodes"].pop(0)


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────
@app.route("/api/reset", methods=["POST"])
def reset():
    global game
    if game.done and game.winner:
        record_episode(game)
    game = GameState()
    return jsonify(game.to_dict())


@app.route("/api/step", methods=["POST"])
def step():
    state = game.step_sim()
    if state["done"] and state["winner"]:
        record_episode(game)
    return jsonify(state)


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    eps = metrics_store["episodes"]

    # Win rate rolling window
    win_rate_history = []
    for i in range(len(eps)):
        window = eps[max(0, i-19):i+1]
        ppo_w = sum(1 for e in window if e["winner"] == "PPO") / len(window)
        win_rate_history.append({
            "ep": eps[i]["ep_num"],
            "ppo_wr": round(ppo_w * 100, 1),
            "fsm_wr": round((1 - ppo_w) * 100, 1),
            "ppo_entropy": eps[i]["ppo_entropy"],
            "fsm_entropy": eps[i]["fsm_entropy"],
            "ppo_reward":  eps[i]["ppo_reward"],
            "fsm_reward":  eps[i]["fsm_reward"],
            "steps":       eps[i]["steps"],
        })

    # Aggregate action counts
    agg_ppo = {a: 0 for a in ACTIONS}
    agg_fsm = {a: 0 for a in ACTIONS}
    for ep in eps:
        for a in ACTIONS:
            agg_ppo[a] += ep["ppo_actions"].get(a, 0)
            agg_fsm[a] += ep["fsm_actions"].get(a, 0)

    total_ppo = max(sum(agg_ppo.values()), 1)
    total_fsm = max(sum(agg_fsm.values()), 1)
    action_dist = [
        {"action": a,
         "PPO": round(agg_ppo[a] / total_ppo * 100, 1),
         "FSM": round(agg_fsm[a] / total_fsm * 100, 1)}
        for a in ACTIONS
    ]

    total = max(metrics_store["total"], 1)
    avg_ppo_ent = round(sum(e["ppo_entropy"] for e in eps) / max(len(eps), 1), 3)
    avg_fsm_ent = round(sum(e["fsm_entropy"] for e in eps) / max(len(eps), 1), 3)
    avg_steps   = round(sum(e["steps"] for e in eps) / max(len(eps), 1), 1)

    # Engagement Index: EI = 0.4*(D/Dmax) + 0.35*(H/Hmax) + 0.25*WR
    ppo_wr  = metrics_store["ppo_wins"] / total
    ei      = 0.4 * (avg_steps / MAX_STEPS) + 0.35 * (avg_ppo_ent / 2.58) + 0.25 * ppo_wr
    ei_fsm  = 0.4 * (avg_steps / MAX_STEPS) + 0.35 * (avg_fsm_ent / 2.58) + 0.25 * (metrics_store["fsm_wins"] / total)

    return jsonify({
        "summary": {
            "total_episodes": metrics_store["total"],
            "ppo_wins":  metrics_store["ppo_wins"],
            "fsm_wins":  metrics_store["fsm_wins"],
            "draws":     metrics_store["draws"],
            "ppo_wr":    round(ppo_wr * 100, 1),
            "avg_ppo_entropy": avg_ppo_ent,
            "avg_fsm_entropy": avg_fsm_ent,
            "avg_steps":       avg_steps,
            "ppo_ei":    round(ei, 3),
            "fsm_ei":    round(ei_fsm, 3),
        },
        "history":     win_rate_history,
        "action_dist": action_dist,
    })


@app.route("/api/batch", methods=["POST"])
def batch():
    """Run N silent episodes, record metrics, return results."""
    data = request.get_json() or {}
    n    = min(int(data.get("n", 20)), 100)

    for _ in range(n):
        g = GameState()
        while not g.done:
            g.step_sim()
        record_episode(g)

    return jsonify({
        "ran": n,
        "ppo_wins": metrics_store["ppo_wins"],
        "fsm_wins": metrics_store["fsm_wins"],
        "total":    metrics_store["total"],
    })

@app.route("/")
def index():
    return jsonify({"status": "ok", "message": "Deep RL Stickman Fight API is running."})
  
@app.route("/api/ablation", methods=["GET"])
def get_ablation():
    return jsonify(ablation)


@app.route("/api/ablation", methods=["POST"])
def set_ablation():
    data = request.get_json() or {}
    key  = data.get("key")
    if key in ablation:
        ablation[key] = not ablation[key]
    return jsonify(ablation)


@app.route("/api/equations", methods=["GET"])
def equations():
    """Return computed equation samples for the Math page."""
    rewards = [1.0, 0.5, -0.2, 1.5, 0.3]
    Gt = discounted_return(rewards)
    probs_ppo = [0.40, 0.28, 0.20, 0.07, 0.03, 0.02]
    probs_fsm = [0.02, 0.02, 0.90, 0.02, 0.02, 0.02]
    ratio, adv = 1.3, 0.8
    return jsonify({
        "Gt":          {"value": Gt, "rewards": rewards, "gamma": GAMMA},
        "ppo_clip":    {"ratio": ratio, "advantage": adv, "eps": EPSILON_CLIP,
                        "result": ppo_clip(ratio, adv)},
        "entropy_ppo": {"probs": probs_ppo, "value": entropy(probs_ppo)},
        "entropy_fsm": {"probs": probs_fsm, "value": entropy(probs_fsm)},
        "gae_delta":   {"r": 1.0, "V_next": 2.1, "V_curr": 1.8, "gamma": GAMMA,
                        "result": round(1.0 + GAMMA * 2.1 - 1.8, 4)},
        "engagement_index": {
            "ppo": round(0.4 * 0.71 + 0.35 * 0.97 + 0.25 * 0.783, 3),
            "fsm": round(0.4 * 0.49 + 0.35 * 0.28 + 0.25 * 0.521, 3),
        },
    })


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Deep RL Stickman Fight — Backend")
    print("  http://localhost:5009")
    print("=" * 50)
    app.run(debug=True, port=5009)
