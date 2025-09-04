// Minimal, fetch-based chat. Expects a backend POST /chat/agent that returns:
// { reply: "…assistant text…", books: [{item_idx, title, author, year, cover_url}, ...] }

function el(tag, className, text) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text) e.textContent = text;
  return e;
}

const chatMessages = document.getElementById("chatMessages");
const chatInput    = document.getElementById("chatInput");
const chatSend     = document.getElementById("chatSend");
const useProfile   = document.getElementById("useProfile");
const restrictToCatalog = document.getElementById("restrictToCatalog");
const resultsGrid  = document.getElementById("chatBookResults");

function appendMessage(role, text) {
  const msg = el("div", `message message--${role}`);
  const inner = el("div", "message-inner");
  inner.textContent = text;
  msg.appendChild(inner);
  chatMessages.appendChild(msg);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function renderBooks(books = []) {
  const grid = document.getElementById("chatBookResults");
  if (!grid) return;            // grid is optional on the chat demo page
  grid.innerHTML = "";
  if (!books.length) return;

  books.forEach(b => {
    const a = document.createElement("a");
    a.href = `/book/${encodeURIComponent(b.item_idx)}`;
    a.className = "book-card";
    a.setAttribute("aria-label", b.title || "Book");

    const img = el("img", "book-cover-img");
    img.src = b.cover_url || "/static/placeholder.png";
    img.alt = b.title ? `Cover of ${b.title}` : "Book cover";
    a.appendChild(img);

    const h3 = el("h3", "book-title");
    const span = el("span", "book-title-text", b.title || "Unknown title");
    h3.appendChild(span);
    a.appendChild(h3);

    if (b.author) a.appendChild(el("p", "author", b.author));
    if (b.year)   a.appendChild(el("p", "year", String(b.year)));

    grid.appendChild(a);
  });
}

async function sendMessage() {
  const text = chatInput.value.trim();
  if (!text) return;

  // show user message
  appendMessage("user", text);
  chatInput.value = "";
  chatInput.focus();

  // show placeholder while fetching
  const thinking = el("div", "message message--bot");
  const inner = el("div", "message-inner");
  inner.textContent = "Thinking…";
  thinking.appendChild(inner);
  chatMessages.appendChild(thinking);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  try {
    const payload = {
      message: text,
      use_profile: !!useProfile?.checked,
      restrict_to_catalog: !!restrictToCatalog?.checked
    };

    const resp = await fetch("/chat/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify(payload)
    });

    const data = await resp.json();
    thinking.remove();

    const reply = data?.reply || "Sorry, I couldn't generate a response.";
    appendMessage("bot", reply);

    // optional book suggestions array
    if (Array.isArray(data?.books)) {
      renderBooks(data.books);
      const grid = document.getElementById("chatBookResults");
      if (grid && data.books.length) {
        grid.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }

  } catch (err) {
    thinking.remove();
    appendMessage("bot", "Something went wrong while contacting the agent.");
    console.error(err);
  }
}

chatSend?.addEventListener("click", sendMessage);

chatInput?.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
