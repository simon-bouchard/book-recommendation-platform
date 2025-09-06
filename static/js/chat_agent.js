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
  const thinking = document.createElement("div");
  thinking.className = "message message--bot";
  thinking.appendChild(document.createElement("div")).className = "message-inner";
  thinking.firstChild.textContent = "Thinking…";
  messagesEl.appendChild(thinking);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  const useProfile = !!document.getElementById("useProfile")?.checked;

  const payload = {
    message: text,
    use_profile: useProfile,
    restrict_to_catalog: true
  };

  try {
    const resp = await fetch("/chat/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify(payload),
    });

    if (!resp.ok) {
      thinking.remove();
      appendMessage(messagesEl, "bot", "Something went wrong while contacting the agent.");
      return;
    }

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
  try { mountBanners({ autoHideMs: 0 }); } catch {}

  const messagesEl = document.getElementById("chatMessages");
  const inputEl    = document.getElementById("chatInput");
  const sendBtn    = document.getElementById("chatSend");
  const bannerOverlay = document.getElementById("chatBannerOverlay");

  // New: toggles
  const useProfileEl = document.getElementById("useProfile");
  const restrictEl   = document.getElementById("restrictToCatalog");

  // Restore persisted prefs
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

  const getFlags = () => ({
    useProfile: useProfileEl ? !!useProfileEl.checked : true,
    restrictToCatalog: restrictEl ? !!restrictEl.checked : true,
  });

  if (!messagesEl || !inputEl || !sendBtn) return;

  // Click to send
  sendBtn.addEventListener("click", async () => {
    const text = (inputEl.value || "").trim();
    if (!text) return;
    appendMessage(messagesEl, "user", text);
    inputEl.value = "";
    await sendChatMessage(text, messagesEl, getFlags());
  });

  // Enter to send; Shift+Enter for newline
  inputEl.addEventListener("keydown", async (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const text = (inputEl.value || "").trim();
      if (!text) return;
      appendMessage(messagesEl, "user", text);
      inputEl.value = "";
      await sendChatMessage(text, messagesEl, getFlags());
    }
  });
});
