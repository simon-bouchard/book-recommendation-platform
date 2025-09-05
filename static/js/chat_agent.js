// /static/js/chat_agent.js  (replace the whole file with this)
import { mountBanners, showBanner } from "/static/js/flash_alert.js";

// ---- DOM helpers ----
function el(tag, className, text) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text != null) e.textContent = text;
  return e;
}

function appendMessage(container, role, text) {
  const msg = el("div", `message message--${role}`);
  const inner = el("div", "message-inner", text);
  msg.appendChild(inner);
  container.appendChild(msg);
  container.scrollTop = container.scrollHeight;
}

function renderBooks(books = []) {
  const grid = document.getElementById("chatBookResults"); // optional section
  if (!grid) return;
  grid.innerHTML = "";
  if (!books.length) return;

  for (const b of books) {
    const a = document.createElement("a");
    a.href = `/book/${encodeURIComponent(b.item_idx)}`;
    a.className = "book-card";
    a.setAttribute("aria-label", b.title || "Book");

    const img = el("img", "book-cover-img");
    img.src = b.cover_url || "/static/placeholder.png";
    img.alt = b.title ? `Cover of ${b.title}` : "Book cover";
    a.appendChild(img);

    const h3 = el("h3", "book-title");
    h3.appendChild(el("span", "book-title-text", b.title || "Unknown title"));
    a.appendChild(h3);

    if (b.author) a.appendChild(el("p", "author", b.author));
    if (b.year)   a.appendChild(el("p", "year", String(b.year)));

    grid.appendChild(a);
  }
}

// ---- Single, definitive send function ----
async function sendChatMessage(text, messagesEl) {
  // “Thinking…” placeholder
  const thinking = el("div", "message message--bot");
  thinking.appendChild(el("div", "message-inner", "Thinking…"));
  messagesEl.appendChild(thinking);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  try {
    const resp = await fetch("/chat/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify({
        message: text,
        use_profile: true,            // toggles are optional; keep true by default
        restrict_to_catalog: true
      }),
    });

    // Handle auth/rate-limit FIRST (don’t try to parse as success)
    if (resp.status === 401) {
      showBanner("warning", "Please log in to use the chatbot.", {
        container: chatBannerOverlay || document.body
      });
      return;
    }
    if (resp.status === 429) {
      const data = await resp.json().catch(() => ({}));
      const msg = data?.detail || "Rate limit exceeded. Please try later.";
      showBanner("warning", msg, {
        container: chatBannerOverlay || document.body
      });
      return;
    }
    if (!resp.ok) {
      thinking.remove();
      showBanner("error", "Something went wrong. Please try again.");
      return;
    }

    // Success path
    const data = await resp.json().catch(() => ({}));
    thinking.remove();

    const reply = data?.reply || "Sorry, I couldn't generate a response.";
    appendMessage(messagesEl, "bot", reply);

    if (Array.isArray(data?.books) && data.books.length) {
      renderBooks(data.books);
      const grid = document.getElementById("chatBookResults");
      grid?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  } catch (err) {
    thinking.remove();
    appendMessage(messagesEl, "bot", "Something went wrong while contacting the agent.");
    console.error(err);
  }
}

// ---- Boot ----
window.addEventListener("DOMContentLoaded", () => {
  // Mount any server-rendered banners (auto-hide, close buttons, etc.)
  try { mountBanners({ autoHideMs: 0 }); } catch {}

  const messagesEl = document.getElementById("chatMessages");
  const inputEl    = document.getElementById("chatInput");
  const sendBtn    = document.getElementById("chatSend");
  const bannerOverlay = document.getElementById("chatBannerOverlay"); 

  if (!messagesEl || !inputEl || !sendBtn) return;

  // Click to send
  sendBtn.addEventListener("click", async () => {
    const text = (inputEl.value || "").trim();
    if (!text) return;
    appendMessage(messagesEl, "user", text);
    inputEl.value = "";
    await sendChatMessage(text, messagesEl);
  });

  // Enter to send; Shift+Enter for newline
  inputEl.addEventListener("keydown", async (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const text = (inputEl.value || "").trim();
      if (!text) return;
      appendMessage(messagesEl, "user", text);
      inputEl.value = "";
      await sendChatMessage(text, messagesEl);
    }
  });
});
