# 🎮 Task 2 — Tic-Tac-Toe AI

> **CodSoft AI Internship | Sreeja Pollreddy | BY25RY287818**

An unbeatable Tic-Tac-Toe AI agent that plays against a human player using the **Minimax algorithm with Alpha-Beta Pruning**. The project demonstrates core concepts in game theory, adversarial search, and optimal decision-making.

---

## 🌟 Features

| Feature | Description |
|---------|-------------|
| **Minimax + Alpha-Beta Pruning** | Unbeatable AI that evaluates every possible game state with optimized search |
| **3 Difficulty Levels** | Easy (random), Medium (hybrid), Impossible (full Minimax) |
| **Real-Time AI Metrics** | Live display of nodes explored, branches pruned, max depth, and think time |
| **Board Evaluation Bar** | Visual indicator showing which side the AI thinks is favored |
| **Play as X or O** | Choose your symbol — AI adapts automatically |
| **Move History Log** | Complete record of every move with position labels |
| **Score Persistence** | Scores saved across sessions via localStorage |
| **Animated Particle Background** | Interactive Canvas-based particle system with mouse tracking |
| **SVG Draw Animations** | Smooth draw-on effects for X and O marks |
| **Animated Win Line** | Gradient SVG line animates across winning cells |
| **Game Over Modal** | End-of-game stats: moves played, AI nodes explored, game duration |
| **Responsive Design** | Works on desktop, tablet, and mobile devices |

---

## 🧠 Algorithm — Minimax with Alpha-Beta Pruning

### How It Works

The **Minimax** algorithm models the game as a tree of all possible future states. It assumes both players play optimally:

- **Maximizer (AI):** Picks the move with the highest score
- **Minimizer (Human):** Picks the move with the lowest score

The algorithm recursively evaluates every possible game outcome and backtracks the optimal score to the root.

### Alpha-Beta Pruning Optimization

**Alpha-Beta Pruning** eliminates branches in the search tree that cannot influence the final decision:

- **Alpha** = best score the Maximizer can guarantee
- **Beta** = best score the Minimizer can guarantee
- When `beta ≤ alpha`, the branch is **pruned** (skipped)

This reduces the effective time complexity:

```
Without pruning:  O(b^d)      →  up to 549,945 nodes
With pruning:     O(b^(d/2))  →  typically ~100-200 nodes
```

### Scoring Function

```
AI wins:    +10 - depth  (prefer faster wins)
Human wins: depth - 10   (prefer slower losses)
Draw:       0
```

The depth penalty ensures the AI wins as quickly as possible and delays losses as long as possible.

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PollreddySreeja/CODSOFT.git
   cd "CODSOFT/Task 2 - TicTacToe AI"
   ```

2. **Open in browser:**
   Simply double-click `index.html` — no server or dependencies needed!

   Or use a local server:
   ```bash
   # Python
   python -m http.server 8000

   # Node.js
   npx serve .
   ```

---

## 📁 Project Structure

```
Task 2 - TicTacToe AI/
├── index.html      # Main HTML structure (game board, panels, modal)
├── style.css       # Cyberpunk dark theme with animations
├── app.js          # Game engine, Minimax AI, particle system
└── README.md       # Documentation (this file)
```

---

## 🛠️ Tech Stack

- **HTML5** — Semantic structure with SVG graphics
- **CSS3** — Custom properties, glassmorphism, keyframe animations, responsive grid
- **JavaScript (ES6+)** — OOP game engine, Canvas API particle system
- **Google Fonts** — Orbitron, Inter, JetBrains Mono
- **Zero external dependencies** — Pure vanilla implementation

---

## 🎯 Key Concepts Demonstrated

1. **Game Theory** — Zero-sum games, optimal strategies, Nash equilibrium
2. **Adversarial Search** — Minimax tree traversal with recursive backtracking
3. **Optimization** — Alpha-Beta pruning reduces search space exponentially
4. **Heuristic Evaluation** — Depth-aware scoring for move quality ranking
5. **Software Engineering** — Modular OOP design, event-driven architecture
6. **UI/UX Design** — Premium aesthetics, micro-animations, responsive layouts

---

## 📸 Gameplay

- 🌱 **Easy Mode** — AI plays randomly, great for practice
- ⚡ **Medium Mode** — AI uses Minimax 60% of the time, balanced challenge
- 🧠 **Impossible Mode** — Full Minimax with Alpha-Beta Pruning, truly unbeatable

---
