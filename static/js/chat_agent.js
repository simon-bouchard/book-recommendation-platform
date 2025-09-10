import { mountBanners, showBanner } from "/static/js/flash_alert.js";
const CHAT_CONTAINER = document.getElementById("chatBox");

// ---- DOM helpers ----
function el(tag, className, text) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text != null) e.textContent = text;
  return e;
}

function appendMessage(container, role, text) {
  const msg = el("div", `message message--${role}`);
  const inner = el("div", "message-inner");
  if (role === "bot") {
    inner.innerHTML = renderMarkdown(text);  // render MD for bot
  } else {
    inner.textContent = text;                // plain text for user
  }
  msg.appendChild(inner);
  container.appendChild(msg);
  container.scrollTop = container.scrollHeight;
}

// ---- Rate-limit UX helpers ----
function setDisabled(on) {
  const sendBtn = document.getElementById("chatSend");
  const inputEl = document.getElementById("chatInput");
  const overlay = document.getElementById("chatBannerOverlay");
  if (sendBtn) sendBtn.disabled = on;
  if (inputEl) inputEl.disabled = on;
  if (overlay) overlay.style.display = on ? "block" : "none";
}

let countdownTimer = null;
function startCountdown(seconds, label) {
  clearInterval(countdownTimer);
  let remaining = Math.max(1, seconds | 0);
  const wrap = showBanner(
    "warning",
    `${label} — wait ${remaining}s`,
    { autoHideMs: 0, withClose: false, role: "status", container: CHAT_CONTAINER }
  );
  setDisabled(true);
  countdownTimer = setInterval(() => {
    remaining -= 1;
    if (remaining <= 0) {
      clearInterval(countdownTimer);
      wrap.remove();
      setDisabled(false);
    } else {
      wrap.querySelector(".alert > div").textContent = `${label} — wait ${remaining}s`;
    }
  }, 1000);
}

function showStopForDay(detail) {
  setDisabled(true);
  showBanner("warning", `${detail} Resets at midnight.`, { autoHideMs: 0, withClose: false, container: CHAT_CONTAINER });
}

function ensureUsageBar() {
  let bar = document.getElementById("rlUsage");
  if (bar) return bar;
  const chatBox = document.getElementById("chatBox");
  bar = el("div", "rl-usage");
  bar.id = "rlUsage";
  bar.style.marginTop = "8px";
  bar.style.fontSize = "12px";
  bar.style.opacity = "0.85";
  chatBox?.appendChild(bar);
  return bar;
}

function updateUsageFromHeaders(h) {
  const dayC = h.get("X-RateLimit-Day-Count");
  const dayL = h.get("X-RateLimit-Day-Limit");
  const minC = h.get("X-RateLimit-Min-Count");
  const minL = h.get("X-RateLimit-Min-Limit");
  const sysC = h.get("X-RateLimit-System-Day-Count");
  const sysL = h.get("X-RateLimit-System-Day-Limit");
  if (!dayC || !dayL || !minC || !minL || !sysC || !sysL) return;

  const bar = ensureUsageBar();
  bar.textContent = `Today: ${dayC}/${dayL} • Per minute: ${minC}/${minL} • System: ${sysC}/${sysL}`;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function renderMarkdown(md) {
  // 1) escape everything
  let html = escapeHtml(md);

  // 2) fenced code blocks ``` ```
  html = html.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code}</code></pre>`);

  // 3) inline code `code`
  html = html.replace(/`([^`\n]+)`/g, '<code>$1</code>');

  // 4) links [text](http://...)
  html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

  // 5) bold **text** and italic *text*
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');

  // 6) headings (# to ######) at line start
  html = html.replace(/^###### (.*)$/gm, '<h6>$1</h6>')
             .replace(/^##### (.*)$/gm,  '<h5>$1</h5>')
             .replace(/^#### (.*)$/gm,   '<h4>$1</h4>')
             .replace(/^### (.*)$/gm,    '<h3>$1</h3>')
             .replace(/^## (.*)$/gm,     '<h2>$1</h2>')
             .replace(/^# (.*)$/gm,      '<h1>$1</h1>');

  // 7) simple unordered lists: lines starting with - or *
  html = html.replace(/(?:^|\n)([-*] .+(?:\n[-*] .+)*)/g, (block) => {
    const items = block.trim().split(/\n/).map(line => line.replace(/^[-*] (.*)$/, '<li>$1</li>')).join('');
    return `\n<ul>${items}</ul>`;
  });

  // 8) line breaks for single newlines (outside of code/blocks)
  html = html.replace(/(?<!<\/h\d>|<\/li>|<\/ul>|<\/pre>)\n/g, '<br>');

  return html;
}

// ---- Single, definitive send function ----
async function sendChatMessage(text, messagesEl) {
  // optimistic UI
  const thinking = document.createElement("div");
  thinking.className = "message message--bot";
  thinking.appendChild(document.createElement("div")).className = "message-inner";
  thinking.firstChild.textContent = "Thinking…";
  messagesEl.appendChild(thinking);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  const useProfile = !!document.getElementById("useProfile")?.checked;
  const payload = { message: text, use_profile: useProfile, restrict_to_catalog: true };

  try {
    setDisabled(true);
    const resp = await fetch("/chat/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify(payload),
    });

    // Success → normal path; also show usage from headers
    if (resp.ok) {
      updateUsageFromHeaders(resp.headers);
      const data = await resp.json().catch(() => ({}));
      thinking.remove();
      const reply = data?.reply || "Sorry, I couldn't generate a response.";
      appendMessage(messagesEl, "bot", reply);
      if (Array.isArray(data?.books) && data.books.length) {
        const gridHost = document.getElementById("chatBookResults");
        if (gridHost) {
          // Avoid nested .book-grid .book-grid
          gridHost.classList.remove("book-grid");
          gridHost.innerHTML = ""; // let the shared renderer take over

          // Reuse the exact renderer used by the book page
          const mod = await import("/static/js/paginated_books.js");
          mod.setupPaginatedBookDisplay({
            books: data.books,
            containerId: "chatBookResults",
            // Chat view doesn't need buttons; render all at once with shared card markup
            manualPagination: true,
            scrollOnFirstRender: true
          });

          // Make sure users see the section
          gridHost.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      }
      setDisabled(false);
      return;
    }

    // Unauthorized (if login enabled)
    if (resp.status === 401) {
      thinking.remove();
      showBanner("warning", "Please log in to use the chatbot.", { autoHideMs: 7000, container: CHAT_CONTAINER });
      setDisabled(false);
      appendMessage(messagesEl, "bot", "Login required.");
      return;
    }

    // Rate limited → reason + Retry-After
    if (resp.status === 429) {
      thinking.remove();
      const reason = resp.headers.get("X-RateLimit-Block-Reason") || "unknown";
      const retryAfter = parseInt(resp.headers.get("Retry-After") || "60", 10);
      let detail = "Rate limit exceeded.";
      try {
        const j = await resp.json();
        if (j?.detail) detail = j.detail;
      } catch {}
      // minute vs day/system
      if (reason === "identity_minute") {
        startCountdown(retryAfter, "You are sending messages too quickly");
      } else if (reason === "identity_day" || reason === "system_day") {
        showStopForDay(detail);
      } else {
        // unknown → safe fallback
        startCountdown(retryAfter, detail);
      }
      appendMessage(messagesEl, "bot", detail);
      return;
    }

    // Other errors
    thinking.remove();
    showBanner("danger", `Error ${resp.status}: ${resp.statusText || "Something went wrong."}`, { autoHideMs: 7000, container: CHAT_CONTAINER });
    appendMessage(messagesEl, "bot", "Something went wrong while contacting the agent.");
    setDisabled(false);

  } catch (err) {
    thinking.remove();
    showBanner("danger", "Network error. Please try again.", { autoHideMs: 7000, container: CHAT_CONTAINER
     });
    appendMessage(messagesEl, "bot", "Something went wrong while contacting the agent.");
    setDisabled(false);
    console.error(err);
  }
}

// ---- Boot ----
window.addEventListener("DOMContentLoaded", () => {
  try { mountBanners({ autoHideMs: 0 }); } catch {}

  const messagesEl = document.getElementById("chatMessages");
  const inputEl    = document.getElementById("chatInput");
  const sendBtn    = document.getElementById("chatSend");

  // toggles (persist)
  const useProfileEl = document.getElementById("useProfile");
  const restrictEl   = document.getElementById("restrictToCatalog");

  if (useProfileEl) {
    const saved = localStorage.getItem("chat.useProfile");
    if (saved !== null) useProfileEl.checked = saved === "true";
    useProfileEl.addEventListener("change", () => {
      localStorage.setItem("chat.useProfile", String(useProfileEl.checked));
    });
  }
  if (restrictEl) {
    const saved = localStorage.getItem("chat.restrictToCatalog");
    if (saved !== null) restrictEl.checked = saved === "true";
    restrictEl.addEventListener("change", () => {
      localStorage.setItem("chat.restrictToCatalog", String(restrictEl.checked));
    });
  }

  if (!messagesEl || !inputEl || !sendBtn) return;

  const send = async () => {
    const text = (inputEl.value || "").trim();
    if (!text || sendBtn.disabled || inputEl.disabled) return;
    appendMessage(messagesEl, "user", text);
    inputEl.value = "";
    await sendChatMessage(text, messagesEl);
  };

  sendBtn.addEventListener("click", send);

  // Enter to send; Shift+Enter for newline
  inputEl.addEventListener("keydown", async (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      await send();
    }
  });
});
