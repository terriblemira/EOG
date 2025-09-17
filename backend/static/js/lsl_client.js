(function () {
  const statusEl = document.getElementById("lsl-status");
  const rateEl = document.getElementById("lsl-rate");
  const chanEl = document.getElementById("lsl-chan");
  const canvas = document.getElementById("lsl-canvas");
  const ctx = canvas ? canvas.getContext("2d") : null;

  let ws;
  let lastSecond = performance.now();
  let msgCount = 0;
  let channels = 0;
  let mostRecentVals = [];

  // simple sparkline buffer (last 300 points of channel 0)
  const maxPoints = 300;
  const buf = [];

  function setStatus(text) {
    if (!statusEl) return;
    statusEl.textContent = text;
    if (text.includes("connected")) {
      statusEl.className = "status status-ok";
    } else if (text.startsWith("WS") || text.startsWith("Connecting")) {
      statusEl.className = "status status-waiting";
    } else if (text.startsWith("LSL:error") || text.includes("closed")) {
      statusEl.className = "status status-fail";
    }
  }

  function draw() {
    if (!ctx || !canvas) return;
    const w = canvas.width = canvas.clientWidth;
    const h = canvas.height = canvas.clientHeight;

    ctx.clearRect(0, 0, w, h);
    ctx.lineWidth = 2;
    ctx.beginPath();

    if (buf.length > 0) {
      const min = Math.min(...buf);
      const max = Math.max(...buf);
      const rng = (max - min) || 1e-6;
      buf.forEach((v, i) => {
        const x = (i / (buf.length - 1)) * (w - 2) + 1;
        const y = h - ((v - min) / rng) * (h - 2) - 1;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
    }
    ctx.stroke();
    requestAnimationFrame(draw);
  }

  function connect() {
    setStatus("Connecting…");
    ws = new WebSocket(`ws://${location.host}/ws/lsl`);

    ws.onopen = () => setStatus("WS connected (waiting for LSL) …");

    ws.onmessage = (e) => {
      const t = typeof e.data === "string" ? e.data : "";
      if (t === "WS:ready") return;
      if (t === "LSL:connected") {
        setStatus("LSL connected");
        return;
      }
      if (t.startsWith("LSL:error")) {
        setStatus(t);
        return;
      }
      try {
        const msg = JSON.parse(e.data);
        // msg.t: array of timestamps, msg.v: array of samples (each is [ch1, ch2, ...])
        if (Array.isArray(msg.v) && msg.v.length) {
          channels = msg.v[0].length || 1;
          chanEl && (chanEl.textContent = `${channels}`);
          // push first channel into sparkline buffer
          for (let i = 0; i < msg.v.length; i++) {
            const sample = msg.v[i];
            mostRecentVals = sample;
            buf.push(sample[0]);
            if (buf.length > maxPoints) buf.shift();
          }
          msgCount += 1;
          const now = performance.now();
          if (now - lastSecond >= 1000) {
            const rate = msgCount * 1000 / (now - lastSecond);
            rateEl && (rateEl.textContent = rate.toFixed(1) + " msg/s");
            msgCount = 0;
            lastSecond = now;
          }
        }
      } catch {
        // plain text already handled above
      }
    };

    ws.onclose = () => {
      setStatus("WS closed. Reconnecting…");
      setTimeout(connect, 1000);
    };
    ws.onerror = () => setStatus("WS error");
  }

  // kick off
  connect();
  requestAnimationFrame(draw);
})();
