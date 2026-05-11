// ============================================================
// TIC-TAC-TOE AI — Minimax with Alpha-Beta Pruning
// CodSoft Internship Task 2
// ============================================================

class TicTacToeAI {
    constructor() {
        // Game state
        this.board = Array(9).fill(null);
        this.humanSymbol = 'X';
        this.aiSymbol = 'O';
        this.currentPlayer = 'X'; // X always goes first
        this.gameActive = true;
        this.difficulty = 'impossible';
        this.moveHistory = [];
        this.gameStartTime = Date.now();

        // Scores
        this.scores = JSON.parse(localStorage.getItem('tttScores')) || { player: 0, ai: 0, draw: 0 };

        // AI stats
        this.nodesExplored = 0;
        this.nodesPruned = 0;
        this.maxDepthReached = 0;
        this.thinkTime = 0;
        this.totalNodesThisGame = 0;

        // Win combinations
        this.winCombos = [
            [0,1,2],[3,4,5],[6,7,8], // rows
            [0,3,6],[1,4,7],[2,5,8], // cols
            [0,4,8],[2,4,6]          // diagonals
        ];

        // Position names for move history
        this.posNames = [
            'Top-Left','Top-Mid','Top-Right',
            'Mid-Left','Center','Mid-Right',
            'Bot-Left','Bot-Mid','Bot-Right'
        ];

        // SVG gradient defs (injected once)
        this.svgDefs = `
            <defs>
                <linearGradient id="xGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#ff006e"/>
                    <stop offset="100%" style="stop-color:#ff4d94"/>
                </linearGradient>
                <linearGradient id="oGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#3a86ff"/>
                    <stop offset="100%" style="stop-color:#00f5d4"/>
                </linearGradient>
            </defs>`;

        this.init();
    }

    init() {
        this.bindEvents();
        this.updateScoreDisplay();
        this.initParticles();
        // If AI goes first
        if (this.humanSymbol === 'O') {
            this.makeAIMove();
        }
    }

    bindEvents() {
        // Cell clicks
        document.querySelectorAll('.cell').forEach(cell => {
            cell.addEventListener('click', () => this.handleCellClick(parseInt(cell.dataset.index)));
        });

        // Difficulty buttons
        document.querySelectorAll('.diff-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.diff-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.difficulty = btn.dataset.diff;
                this.updateAlgoInfo();
                this.resetGame();
            });
        });

        // Symbol buttons
        document.querySelectorAll('.sym-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.sym-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.humanSymbol = btn.dataset.symbol;
                this.aiSymbol = this.humanSymbol === 'X' ? 'O' : 'X';
                this.resetGame();
            });
        });
    }

    // ========================
    // GAME FLOW
    // ========================

    handleCellClick(index) {
        if (!this.gameActive || this.board[index] || this.currentPlayer !== this.humanSymbol) return;

        this.makeMove(index, this.humanSymbol);

        const result = this.checkGameEnd();
        if (result) return;

        // AI's turn
        this.currentPlayer = this.aiSymbol;
        this.setStatus('thinking', 'AI is analyzing the board...');

        setTimeout(() => {
            if (this.gameActive) this.makeAIMove();
        }, 400 + Math.random() * 300);
    }

    makeMove(index, symbol) {
        this.board[index] = symbol;
        this.renderCell(index, symbol);
        this.moveHistory.push({ symbol, index, moveNum: this.moveHistory.length + 1 });
        this.updateMoveHistory();
    }

    makeAIMove() {
        let bestMove;
        this.nodesExplored = 0;
        this.nodesPruned = 0;
        this.maxDepthReached = 0;

        const startTime = performance.now();

        switch (this.difficulty) {
            case 'easy':
                bestMove = this.getRandomMove();
                break;
            case 'medium':
                bestMove = Math.random() < 0.6 ? this.getBestMove() : this.getRandomMove();
                break;
            case 'impossible':
                bestMove = this.getBestMove();
                break;
        }

        this.thinkTime = (performance.now() - startTime).toFixed(1);
        this.totalNodesThisGame += this.nodesExplored;

        this.updateAIMetrics();

        if (bestMove !== undefined && bestMove !== null) {
            this.makeMove(bestMove, this.aiSymbol);
            this.currentPlayer = this.humanSymbol;

            const result = this.checkGameEnd();
            if (!result) {
                this.setStatus('player', 'Your turn — tap a cell to play');
            }
        }
    }

    // ========================
    // MINIMAX WITH ALPHA-BETA PRUNING
    // ========================

    getBestMove() {
        let bestScore = -Infinity;
        let bestMove = null;

        const availableMoves = this.getAvailableMoves();

        for (const move of availableMoves) {
            this.board[move] = this.aiSymbol;
            const score = this.minimax(this.board, 0, false, -Infinity, Infinity);
            this.board[move] = null;

            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }
        }

        // Update eval bar based on best score
        this.updateEvalBar(bestScore);

        return bestMove;
    }

    minimax(board, depth, isMaximizing, alpha, beta) {
        this.nodesExplored++;
        if (depth > this.maxDepthReached) this.maxDepthReached = depth;

        // Terminal state checks
        const winner = this.checkWinner(board);
        if (winner === this.aiSymbol) return 10 - depth;   // AI wins (prefer faster wins)
        if (winner === this.humanSymbol) return depth - 10;  // Human wins (prefer slower losses)
        if (this.isBoardFull(board)) return 0;               // Draw

        if (isMaximizing) {
            let maxEval = -Infinity;
            for (let i = 0; i < 9; i++) {
                if (board[i] === null) {
                    board[i] = this.aiSymbol;
                    const eval_ = this.minimax(board, depth + 1, false, alpha, beta);
                    board[i] = null;
                    maxEval = Math.max(maxEval, eval_);
                    alpha = Math.max(alpha, eval_);
                    if (beta <= alpha) {
                        this.nodesPruned++;
                        break; // Beta cutoff (pruning!)
                    }
                }
            }
            return maxEval;
        } else {
            let minEval = Infinity;
            for (let i = 0; i < 9; i++) {
                if (board[i] === null) {
                    board[i] = this.humanSymbol;
                    const eval_ = this.minimax(board, depth + 1, true, alpha, beta);
                    board[i] = null;
                    minEval = Math.min(minEval, eval_);
                    beta = Math.min(beta, eval_);
                    if (beta <= alpha) {
                        this.nodesPruned++;
                        break; // Alpha cutoff (pruning!)
                    }
                }
            }
            return minEval;
        }
    }

    getRandomMove() {
        const available = this.getAvailableMoves();
        return available[Math.floor(Math.random() * available.length)];
    }

    getAvailableMoves() {
        return this.board.reduce((moves, cell, i) => {
            if (cell === null) moves.push(i);
            return moves;
        }, []);
    }

    // ========================
    // WIN / DRAW DETECTION
    // ========================

    checkWinner(board) {
        for (const combo of this.winCombos) {
            const [a, b, c] = combo;
            if (board[a] && board[a] === board[b] && board[a] === board[c]) {
                return board[a];
            }
        }
        return null;
    }

    getWinningCombo(board) {
        for (const combo of this.winCombos) {
            const [a, b, c] = combo;
            if (board[a] && board[a] === board[b] && board[a] === board[c]) {
                return combo;
            }
        }
        return null;
    }

    isBoardFull(board) {
        return board.every(cell => cell !== null);
    }

    checkGameEnd() {
        const winner = this.checkWinner(this.board);
        const combo = this.getWinningCombo(this.board);

        if (winner) {
            this.gameActive = false;
            if (combo) this.highlightWin(combo);

            if (winner === this.humanSymbol) {
                this.scores.player++;
                this.setStatus('win', '🎉 You won! Impressive!');
                setTimeout(() => this.showModal('win'), 800);
            } else {
                this.scores.ai++;
                this.setStatus('lose', '🤖 AI wins! The machine prevails.');
                setTimeout(() => this.showModal('lose'), 800);
            }
            this.saveScores();
            this.updateScoreDisplay();
            return true;
        }

        if (this.isBoardFull(this.board)) {
            this.gameActive = false;
            this.scores.draw++;
            this.setStatus('draw', "🤝 It's a draw! Well played.");
            setTimeout(() => this.showModal('draw'), 800);
            this.saveScores();
            this.updateScoreDisplay();
            return true;
        }

        return false;
    }

    // ========================
    // RENDERING
    // ========================

    renderCell(index, symbol) {
        const cell = document.getElementById(`cell${index}`);
        const content = cell.querySelector('.cell-content');
        cell.classList.add('taken', symbol === 'X' ? 'x-cell' : 'o-cell');

        if (symbol === 'X') {
            content.innerHTML = `
                <svg class="x-mark" viewBox="0 0 50 50">
                    ${this.svgDefs}
                    <path d="M14 14L36 36" stroke="url(#xGradient)" stroke-width="5" stroke-linecap="round"/>
                    <path d="M36 14L14 36" stroke="url(#xGradient)" stroke-width="5" stroke-linecap="round" style="animation-delay:0.15s"/>
                </svg>`;
        } else {
            content.innerHTML = `
                <svg class="o-mark" viewBox="0 0 50 50">
                    ${this.svgDefs}
                    <circle cx="25" cy="25" r="14" stroke="url(#oGradient)" stroke-width="4" fill="none"/>
                </svg>`;
        }
    }

    highlightWin(combo) {
        combo.forEach(i => document.getElementById(`cell${i}`).classList.add('win-cell'));
        this.drawWinLine(combo);
    }

    drawWinLine(combo) {
        const line = document.getElementById('winLine');
        const boardEl = document.getElementById('gameBoard');
        const boardRect = boardEl.getBoundingClientRect();
        const svgEl = document.getElementById('winLineSvg');
        const svgRect = svgEl.getBoundingClientRect();

        const getCenter = (idx) => {
            const cell = document.getElementById(`cell${idx}`);
            const rect = cell.getBoundingClientRect();
            return {
                x: ((rect.left + rect.width / 2 - svgRect.left) / svgRect.width) * 330,
                y: ((rect.top + rect.height / 2 - svgRect.top) / svgRect.height) * 330
            };
        };

        const start = getCenter(combo[0]);
        const end = getCenter(combo[2]);

        line.setAttribute('x1', start.x);
        line.setAttribute('y1', start.y);
        line.setAttribute('x2', end.x);
        line.setAttribute('y2', end.y);

        requestAnimationFrame(() => line.classList.add('show'));
    }

    // ========================
    // STATUS & UI UPDATES
    // ========================

    setStatus(type, text) {
        const indicator = document.getElementById('statusIndicator');
        const textEl = document.getElementById('statusText');
        indicator.className = 'status-indicator ' + type;
        textEl.textContent = text;
    }

    updateScoreDisplay() {
        const total = this.scores.player + this.scores.ai + this.scores.draw || 1;
        document.getElementById('playerScore').textContent = this.scores.player;
        document.getElementById('aiScore').textContent = this.scores.ai;
        document.getElementById('drawScore').textContent = this.scores.draw;
        document.getElementById('playerFill').style.width = (this.scores.player / total * 100) + '%';
        document.getElementById('aiFill').style.width = (this.scores.ai / total * 100) + '%';
        document.getElementById('drawFill').style.width = (this.scores.draw / total * 100) + '%';
    }

    updateAIMetrics() {
        this.animateMetric('nodesExplored', this.nodesExplored);
        this.animateMetric('nodesPruned', this.nodesPruned);
        this.animateMetric('maxDepth', this.maxDepthReached);
        document.getElementById('thinkTime').textContent = this.thinkTime + 'ms';
        document.getElementById('thinkTime').classList.add('updated');
        setTimeout(() => document.getElementById('thinkTime').classList.remove('updated'), 400);
    }

    animateMetric(id, value) {
        const el = document.getElementById(id);
        el.textContent = value;
        el.classList.add('updated');
        setTimeout(() => el.classList.remove('updated'), 400);
    }

    updateEvalBar(score) {
        // Score ranges from -10 to 10. Map to 0-100%
        const pct = ((score + 10) / 20) * 100;
        document.getElementById('evalFill').style.width = pct + '%';
        document.getElementById('evalMarker').style.left = pct + '%';
    }

    updateMoveHistory() {
        const list = document.getElementById('moveList');
        list.innerHTML = '';
        this.moveHistory.forEach(m => {
            const entry = document.createElement('div');
            entry.className = `move-entry ${m.symbol === 'X' ? 'x-move' : 'o-move'}`;
            const isHuman = m.symbol === this.humanSymbol;
            entry.innerHTML = `
                <span class="move-num">#${m.moveNum}</span>
                <span class="move-player ${m.symbol === 'X' ? 'x-player' : 'o-player'}">${m.symbol}</span>
                <span class="move-pos">${this.posNames[m.index]}</span>
                <span style="font-size:0.65rem;color:var(--text-muted)">${isHuman ? 'You' : 'AI'}</span>`;
            list.appendChild(entry);
        });
        list.scrollTop = list.scrollHeight;
    }

    updateAlgoInfo() {
        const tag = document.getElementById('algoTag');
        const desc = document.querySelector('.algo-desc');
        switch (this.difficulty) {
            case 'easy':
                tag.textContent = 'RANDOM SELECTION';
                desc.innerHTML = 'The AI picks <strong>random</strong> empty cells. Great for beginners who want a relaxed game.';
                break;
            case 'medium':
                tag.textContent = 'HYBRID STRATEGY';
                desc.innerHTML = 'The AI uses Minimax <strong>60%</strong> of the time and plays randomly otherwise. A balanced challenge.';
                break;
            case 'impossible':
                tag.textContent = 'MINIMAX + α-β PRUNING';
                desc.innerHTML = 'The AI evaluates <strong>every possible</strong> future game state using recursive search. Alpha-Beta Pruning eliminates provably suboptimal branches, reducing computation by up to <strong>60%</strong>.';
                break;
        }
    }

    // ========================
    // MODAL
    // ========================

    showModal(result) {
        const modal = document.getElementById('gameOverModal');
        const icon = document.getElementById('modalIcon');
        const title = document.getElementById('modalTitle');
        const subtitle = document.getElementById('modalSubtitle');
        const elapsed = ((Date.now() - this.gameStartTime) / 1000).toFixed(1);

        document.getElementById('msMovesPlayed').textContent = this.moveHistory.length;
        document.getElementById('msTotalNodes').textContent = this.totalNodesThisGame;
        document.getElementById('msGameTime').textContent = elapsed + 's';

        switch (result) {
            case 'win':
                icon.textContent = '🏆';
                title.textContent = 'YOU WIN!';
                subtitle.textContent = 'Outstanding! You beat the AI!';
                break;
            case 'lose':
                icon.textContent = '🤖';
                title.textContent = 'AI WINS';
                subtitle.textContent = 'The machine intelligence prevails.';
                break;
            case 'draw':
                icon.textContent = '🤝';
                title.textContent = "IT'S A DRAW";
                subtitle.textContent = 'A perfectly balanced game.';
                break;
        }

        modal.classList.add('show');
    }

    hideModal() {
        document.getElementById('gameOverModal').classList.remove('show');
    }

    // ========================
    // GAME CONTROLS
    // ========================

    resetGame() {
        this.hideModal();
        this.board = Array(9).fill(null);
        this.currentPlayer = 'X';
        this.gameActive = true;
        this.moveHistory = [];
        this.gameStartTime = Date.now();
        this.totalNodesThisGame = 0;

        // Clear board
        document.querySelectorAll('.cell').forEach(cell => {
            cell.classList.remove('taken', 'x-cell', 'o-cell', 'win-cell');
            cell.querySelector('.cell-content').innerHTML = '';
        });

        // Clear win line
        const line = document.getElementById('winLine');
        line.classList.remove('show');
        line.setAttribute('x1', 0);
        line.setAttribute('y1', 0);
        line.setAttribute('x2', 0);
        line.setAttribute('y2', 0);

        // Reset UI
        this.setStatus('player', 'Your turn — tap a cell to play');
        this.updateMoveHistory();
        this.updateEvalBar(0);
        this.animateMetric('nodesExplored', 0);
        this.animateMetric('nodesPruned', 0);
        this.animateMetric('maxDepth', 0);
        document.getElementById('thinkTime').textContent = '0ms';

        document.getElementById('moveList').innerHTML = '<div class="move-empty">No moves yet — start playing!</div>';

        // If AI goes first
        if (this.humanSymbol === 'O') {
            this.currentPlayer = 'X'; // X goes first
            this.setStatus('thinking', 'AI is making the first move...');
            setTimeout(() => this.makeAIMove(), 500);
        }
    }

    resetScores() {
        this.scores = { player: 0, ai: 0, draw: 0 };
        this.saveScores();
        this.updateScoreDisplay();
        this.resetGame();
    }

    playAgain() {
        this.hideModal();
        setTimeout(() => this.resetGame(), 300);
    }

    saveScores() {
        localStorage.setItem('tttScores', JSON.stringify(this.scores));
    }

    // ========================
    // PARTICLE SYSTEM
    // ========================

    initParticles() {
        const canvas = document.getElementById('particleCanvas');
        const ctx = canvas.getContext('2d');
        let particles = [];
        let mouse = { x: null, y: null };

        const resize = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        };
        resize();
        window.addEventListener('resize', resize);

        window.addEventListener('mousemove', (e) => {
            mouse.x = e.clientX;
            mouse.y = e.clientY;
        });

        class Particle {
            constructor() {
                this.x = Math.random() * canvas.width;
                this.y = Math.random() * canvas.height;
                this.size = Math.random() * 2 + 0.5;
                this.speedX = (Math.random() - 0.5) * 0.5;
                this.speedY = (Math.random() - 0.5) * 0.5;
                this.opacity = Math.random() * 0.3 + 0.1;
                const colors = ['131,56,236', '58,134,255', '255,0,110', '0,245,212'];
                this.color = colors[Math.floor(Math.random() * colors.length)];
            }
            update() {
                this.x += this.speedX;
                this.y += this.speedY;
                if (this.x > canvas.width) this.x = 0;
                if (this.x < 0) this.x = canvas.width;
                if (this.y > canvas.height) this.y = 0;
                if (this.y < 0) this.y = canvas.height;

                // Mouse interaction
                if (mouse.x !== null) {
                    const dx = mouse.x - this.x;
                    const dy = mouse.y - this.y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 120) {
                        this.x -= dx * 0.01;
                        this.y -= dy * 0.01;
                    }
                }
            }
            draw() {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(${this.color}, ${this.opacity})`;
                ctx.fill();
            }
        }

        const particleCount = Math.min(80, Math.floor((canvas.width * canvas.height) / 15000));
        for (let i = 0; i < particleCount; i++) {
            particles.push(new Particle());
        }

        const connectParticles = () => {
            for (let a = 0; a < particles.length; a++) {
                for (let b = a + 1; b < particles.length; b++) {
                    const dx = particles[a].x - particles[b].x;
                    const dy = particles[a].y - particles[b].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 150) {
                        const opacity = (1 - dist / 150) * 0.08;
                        ctx.beginPath();
                        ctx.strokeStyle = `rgba(131, 56, 236, ${opacity})`;
                        ctx.lineWidth = 0.5;
                        ctx.moveTo(particles[a].x, particles[a].y);
                        ctx.lineTo(particles[b].x, particles[b].y);
                        ctx.stroke();
                    }
                }
            }
        };

        const animate = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            particles.forEach(p => { p.update(); p.draw(); });
            connectParticles();
            requestAnimationFrame(animate);
        };
        animate();
    }
}

// Initialize the game
const game = new TicTacToeAI();
