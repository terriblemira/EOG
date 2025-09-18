(function () {
  // DOM
  const connEl  = document.getElementById("pong-conn");
  const lastEl  = document.getElementById("pong-last");
  const scoreEl = document.getElementById("pong-score");
  const canvas  = document.getElementById("pong-canvas");
  const ctx     = canvas.getContext("2d");

  // Field
  const W = canvas.width;
  const H = canvas.height;

  // Paddles
  const PADDLE_W = 14;
  const PADDLE_H = 90;
  const PADDLE_MARGIN = 24;
  const PADDLE_SPEED = 6; // base speed; “down/up” increments dy towards ±PADDLE_SPEED

  const left =  { x: PADDLE_MARGIN,        y: H/2 - PADDLE_H/2, dy: 0, score: 0 };
  const right = { x: W - PADDLE_MARGIN - PADDLE_W, y: H/2 - PADDLE_H/2, dy: 0, score: 0 };

  // Ball
  const BALL_R = 8;
  let ball = resetBall();

  // Game loop timing
  let running = true;
  let lastTs = 0;

  // EOG command handling
  function setConnStatus(text, ok=false, bad=false) {
    connEl.textContent = text;
    connEl.className = "status " + (ok ? "status-ok" : bad ? "status-fail" : "status-waiting");
  }

  function connectCmdWS() {
    setConnStatus("Connecting…");
    const ws = new WebSocket(`ws://${location.host}/ws/cmd`);
    ws.onopen = () => setConnStatus("Connected", true);
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg && msg.cmd) {
          lastEl.textContent = msg.cmd;
          handleCommand(msg.cmd);
        }
      } catch {
        // ignore text messages like "LSL:connected"
      }
    };
    ws.onclose = () => { setConnStatus("Disconnected — retrying…", false, true); setTimeout(connectCmdWS, 700); };
    ws.onerror = () => setConnStatus("WS error", false, true);
  }
  connectCmdWS();

  // Map commands to paddle movement:
  // - "up"/"down": move the LEFT paddle (you), continuous while new cmds arrive
  // - "left"/"right": optional nudge of ball spin or ignore (kept to symmetry)
  function handleCommand(cmd) {
    if (cmd === "up")    left.dy = -PADDLE_SPEED;
    if (cmd === "down")  left.dy =  PADDLE_SPEED;
    if (cmd === "left")  spinBall(-0.5);
    if (cmd === "right") spinBall( 0.5);
  }

  // Keyboard debug (ArrowUp/ArrowDown)
  document.addEventListener("keydown", (e) => {
    if (e.key === "ArrowUp")    { left.dy = -PADDLE_SPEED; }
    if (e.key === "ArrowDown")  { left.dy =  PADDLE_SPEED; }
  });
  document.addEventListener("keyup", (e) => {
    if (e.key === "ArrowUp" || e.key === "ArrowDown") { left.dy = 0; }
  });

  // Restart
  document.getElementById("pong-restart").addEventListener("click", () => {
    left.y = H/2 - PADDLE_H/2; left.dy = 0; left.score = 0;
    right.y = H/2 - PADDLE_H/2; right.dy = 0; right.score = 0;
    ball = resetBall(true);
    scoreEl.textContent = "0";
    running = true;
    setConnStatus("Connected", true);
  });

  // AI for right paddle (simple tracking)
  function aiUpdate() {
    const center = right.y + PADDLE_H/2;
    if (ball.y < center - 6) right.dy = -PADDLE_SPEED * 0.9;
    else if (ball.y > center + 6) right.dy =  PADDLE_SPEED * 0.9;
    else right.dy = 0;
  }

  // Ball helpers
  function resetBall(randomize = false) {
    const angle = (Math.random() * 0.6 - 0.3); // slight vertical angle
    const speed = 5;
    return {
      x: W/2, y: H/2,
      vx: (randomize ? (Math.random() < 0.5 ? -speed : speed) : speed),
      vy: speed * angle,
      s: speed
    };
  }
  function spinBall(amount) {
    // small vertical spin tweak
    ball.vy += amount;
  }

  // Physics step
  function update(dt) {
    if (!running) return;

    // Move paddles
    left.y += left.dy;
    right.y += right.dy;
    left.y = clamp(left.y, 0, H - PADDLE_H);
    right.y = clamp(right.y, 0, H - PADDLE_H);

    // Simple AI
    aiUpdate();

    // Move ball
    ball.x += ball.vx;
    ball.y += ball.vy;

    // Top/bottom bounce
    if (ball.y - BALL_R <= 0 && ball.vy < 0) { ball.y = BALL_R; ball.vy = -ball.vy; }
    if (ball.y + BALL_R >= H && ball.vy > 0) { ball.y = H - BALL_R; ball.vy = -ball.vy; }

    // Left paddle collision
    if (ball.x - BALL_R <= left.x + PADDLE_W &&
        ball.y >= left.y && ball.y <= left.y + PADDLE_H &&
        ball.vx < 0) {
      ball.x = left.x + PADDLE_W + BALL_R;
      ball.vx = -ball.vx * 1.03; // speed up slightly
      // add spin based on where it hit the paddle
      const hit = (ball.y - (left.y + PADDLE_H/2)) / (PADDLE_H/2);
      ball.vy += hit * 2.5;
    }

    // Right paddle collision
    if (ball.x + BALL_R >= right.x &&
        ball.y >= right.y && ball.y <= right.y + PADDLE_H &&
        ball.vx > 0) {
      ball.x = right.x - BALL_R;
      ball.vx = -ball.vx * 1.03;
      const hit = (ball.y - (right.y + PADDLE_H/2)) / (PADDLE_H/2);
      ball.vy += hit * 2.5;
    }

    // Score / reset
    if (ball.x < -BALL_R) {
      // right scores
      right.score += 1;
      scoreEl.textContent = String(left.score); // we show your (left) score
      ball = resetBall(true);
    } else if (ball.x > W + BALL_R) {
      // left scores
      left.score += 1;
      scoreEl.textContent = String(left.score);
      ball = resetBall(true);
    }
  }

  // Render
  function draw() {
    ctx.clearRect(0, 0, W, H);

    // Midline
    ctx.globalAlpha = 0.15;
    ctx.setLineDash([8, 12]);
    ctx.beginPath();
    ctx.moveTo(W/2, 0); ctx.lineTo(W/2, H);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;

    // Paddles
    ctx.fillStyle = "#fff";
    ctx.fillRect(left.x, left.y, PADDLE_W, PADDLE_H);
    ctx.fillRect(right.x, right.y, PADDLE_W, PADDLE_H);

    // Ball
    ctx.beginPath();
    ctx.arc(ball.x, ball.y, BALL_R, 0, Math.PI * 2);
    ctx.fill();

    // Score (big, top-center)
    ctx.globalAlpha = 0.9;
    ctx.font = "28px system-ui, Arial";
    ctx.textAlign = "center";
    ctx.fillText(String(left.score), W * 0.25, 40);
    ctx.fillText(String(right.score), W * 0.75, 40);
    ctx.globalAlpha = 1;
  }

  // Loop
  function loop(ts) {
    const dt = ts - lastTs; lastTs = ts;
    update(dt);
    draw();
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);

  // Utils
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
})();
