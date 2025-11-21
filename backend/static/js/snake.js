(function () {
  const connEl = document.getElementById("snake-conn");
  const lastEl = document.getElementById("snake-last");
  const scoreEl = document.getElementById("snake-score");
  const canvas = document.getElementById("snake-canvas");
  const ctx = canvas.getContext("2d");

  // Grid settings
  const COLS = 24;
  const ROWS = 24;
  const CELL = Math.floor(canvas.width / COLS);

  // Game state
  let snake = [{ x: 12, y: 12 }];
  let dir = { x: 1, y: 0 };          // moving right initially
  let pendingDir = dir;               // input buffering
  let food = spawnFood();
  let alive = true;
  let score = 0;
  let tickMs = 180;                   // game speed (lower = faster)

  //M: Websocket-connection
  let ws = null;


  // Prevent reversals
  function setDirection(cmd) {
    if (!cmd) return;
    if (cmd === "left"  && dir.x !== 1)  pendingDir = { x: -1, y:  0 };
    if (cmd === "right" && dir.x !== -1) pendingDir = { x:  1, y:  0 };
    if (cmd === "up"    && dir.y !== 1)  pendingDir = { x:  0, y: -1 };
    if (cmd === "down"  && dir.y !== -1) pendingDir = { x:  0, y:  1 };
  }

  // Debug: arrow keys
  document.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft")  setDirection("left");
    if (e.key === "ArrowRight") setDirection("right");
    if (e.key === "ArrowUp")    setDirection("up");
    if (e.key === "ArrowDown")  setDirection("down");
  });

  function connectWebSocket() {
    // connect Websocket to /wsMovement endpoint (in app.py)
    ws = new WebSocket('ws://localhost:8000/wsMovement');
    
    ws.onopen = function() {
        console.log('WebSocket verbunden zu /wsMovement');
        statusDiv.textContent = 'Connected';
        statusDiv.className = 'status status-connected';
    };
    
    ws.onmessage = function(event) {
        const message = event.data;
        console.log('Received:', message);
        
        // M: transform only relevant signals to events 
        if (message === "Right-command received") {
            setDirection("left");
            lastCmdDiv.textContent = 'RIGHT';
        } else if (message === "Left-command received") {
            setDirection("right");
            lastCmdDiv.textContent = 'LEFT';
        } else if (message === "Up-command received") {
            setDirection("up");
            lastCmdDiv.textContent = 'UP';
        } else if (message === "Down-command received") {
            setDirection("down");
            lastCmdDiv.textContent = 'DOWN';
        }
    };
    
    ws.onerror = function(error) {
        console.error('WebSocket Fehler:', error);
        statusDiv.textContent = 'Error';
        statusDiv.className = 'status status-error';
    };
    
    ws.onclose = function() {
        console.log('WebSocket geschlossen');
        statusDiv.textContent = 'Disconnected';
        statusDiv.className = 'status status-waiting';
        
        //M: Auto-Reconnect after 3 secs
        setTimeout(connectWebSocket, 3000);
    };
}

  // Game loop
  let lastTick = 0;
  function loop(ts) {
    if (!alive) return draw(); // stop updates, keep last frame
    if (ts - lastTick >= tickMs) {
      step();
      lastTick = ts;
    }
    draw();
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);

  function step() {
    // Apply buffered direction
    dir = pendingDir;

    // New head
    const head = { x: (snake[0].x + dir.x), y: (snake[0].y + dir.y) };

    // Wrap around edges (torus); change to collide if you prefer
    head.x = (head.x + COLS) % COLS;
    head.y = (head.y + ROWS) % ROWS;

    // Self-collision
    if (snake.some(s => s.x === head.x && s.y === head.y)) {
      alive = false;
      setConnStatus("Game over", false, true);
      return;
    }

    snake.unshift(head);

    // Eat
    if (head.x === food.x && head.y === food.y) {
      score += 1;
      scoreEl.textContent = String(score);
      food = spawnFood();
      // optional speed-up
      if (tickMs > 60) tickMs -= 1;
    } else {
      snake.pop();
    }
  }

  function draw() {
    // Clear
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Grid background (subtle)
    ctx.globalAlpha = 0.1;
    ctx.beginPath();
    for (let i = 0; i <= COLS; i++) {
      ctx.moveTo(i * CELL, 0);
      ctx.lineTo(i * CELL, canvas.height);
    }
    for (let j = 0; j <= ROWS; j++) {
      ctx.moveTo(0, j * CELL);
      ctx.lineTo(canvas.width, j * CELL);
    }
    ctx.stroke();
    ctx.globalAlpha = 1.0;

    // Food
    rect(food.x, food.y, "#a3e635");

    // Snake
    snake.forEach((seg, idx) => rect(seg.x, seg.y, idx === 0 ? "#60a5fa" : "#3b82f6"));
  }

  function rect(gx, gy, fillStyle) {
    ctx.fillStyle = fillStyle;
    ctx.fillRect(gx * CELL + 1, gy * CELL + 1, CELL - 2, CELL - 2);
  }

  function spawnFood() {
    while (true) {
      const f = { x: Math.floor(Math.random() * COLS), y: Math.floor(Math.random() * ROWS) };
      if (!snake.some(s => s.x === f.x && s.y === f.y)) return f;
    }
  }
})();
