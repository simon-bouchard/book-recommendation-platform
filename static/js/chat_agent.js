// static/js/chat_agent.js
/**
 * Chat agent frontend — drives the chatbot UI.
 *
 * Sends messages to /chat/stream (SSE) and handles three chunk types:
 *   - status  → updates the in-progress "thinking" bubble in-place
 *   - token   → streams characters into the bot message bubble
 *   - complete → finalises the message, populates hover-card data from book objects
 *
 * Inline <book id="X"> references emitted by the LLM are rendered as
 * hoverable citation links; bookDataMap is populated from the complete chunk's
 * `books` array so hover cards show rich metadata (cover, author, year, etc.).
 */

import { mountBanners, showBanner } from "/static/js/flash_alert.js";

const CHAT_CONTAINER = document.getElementById("chatBox");

/**
 * Rich book metadata keyed by item_idx (string).
 * Populated from the `complete` chunk's `books` array.
 * @type {Record<string, object>}
 */
let bookDataMap = {};

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------

function el(tag, className, text) {
	const e = document.createElement(tag);
	if (className) e.className = className;
	if (text != null) e.textContent = text;
	return e;
}

function escapeHtml(s) {
	return String(s ?? "").replace(
		/[&<>"']/g,
		(c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
	);
}

// ---------------------------------------------------------------------------
// Message rendering
// ---------------------------------------------------------------------------

/**
 * Append a complete message bubble to the message list.
 * Bot messages are passed through renderMarkdown; user messages are plain-text.
 */
function appendMessage(container, role, text) {
	const msg = el("div", `message message--${role}`);
	const inner = el("div", "message-inner");
	if (role === "bot") {
		inner.innerHTML = renderMarkdown(text);
	} else {
		inner.textContent = text;
	}
	msg.appendChild(inner);
	container.appendChild(msg);
	container.scrollTop = container.scrollHeight;

	if (role === "bot") {
		attachBookLinkHandlers(msg);
	}

	return msg;
}

/**
 * Create an empty bot bubble and return handles for progressive updates.
 * @returns {{ bubble: HTMLElement, inner: HTMLElement }}
 */
function createBotBubble(container) {
	const bubble = el("div", "message message--bot");
	const inner = el("div", "message-inner");
	bubble.appendChild(inner);
	container.appendChild(bubble);
	container.scrollTop = container.scrollHeight;
	return { bubble, inner };
}

/**
 * Walk all .inline-book-ref[data-book-id] links inside a message element
 * and wire up hover-card show/hide events.
 */
function attachBookLinkHandlers(msgEl) {
	msgEl.querySelectorAll("a.inline-book-ref[data-book-id]").forEach((link) => {
		link.addEventListener("mouseenter", () => showBookHoverCard(link, link.dataset.bookId));
	});
}

// ---------------------------------------------------------------------------
// Markdown renderer
// ---------------------------------------------------------------------------

/**
 * Lightweight Markdown → HTML renderer that also handles the custom
 * <book id="X">Title</book> syntax emitted by the LLM.
 *
 * Processing order matters:
 *   1. Extract book refs (before escaping so angle-brackets are preserved)
 *   2. Escape all remaining HTML
 *   3. Apply Markdown rules
 *   4. Restore book refs as clickable anchor tags
 */
function renderMarkdown(md) {
	// 1) Extract inline book references before HTML-escaping
	const bookRefs = [];
	let html = md.replace(/<book\s+id="(\d+)">([^<]+)<\/book>/g, (_, id, title) => {
		const idx = bookRefs.length;
		bookRefs.push({ id: String(id), title: String(title) });
		return `\x00BOOKREF${idx}\x00`;
	});

	html = html.replace(/\[([^\]]+)\]\((\d+)\)/g, (_, title, id) => {
		const idx = bookRefs.length;
		bookRefs.push({ id: String(id), title: String(title) });
		return `\x00BOOKREF${idx}\x00`;
	});

	// 2) Escape everything else
	html = escapeHtml(html);

	// 3) Fenced code blocks
	html = html.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code}</code></pre>`);

	// 4) Inline code
	html = html.replace(/`([^`\n]+)`/g, "<code>$1</code>");

	// 5) Links
	html = html.replace(
		/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
		'<a href="$2" target="_blank" rel="noopener">$1</a>'
	);

	// 6) Bold / italic
	html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
	html = html.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");

	// 7) Headings
	for (let lvl = 6; lvl >= 1; lvl--) {
		const hashes = "#".repeat(lvl);
		html = html.replace(
			new RegExp(`^${hashes} (.*)$`, "gm"),
			`<h${lvl}>$1</h${lvl}>`
		);
	}

	// 8) Unordered lists (contiguous - / * lines)
	html = html.replace(/(?:^|\n)((?:[-*] .+\n?)+)/g, (_, block) => {
		const items = block
			.trim()
			.split(/\n/)
			.map((line) => line.replace(/^[-*] (.*)$/, "<li>$1</li>"))
			.join("");
		return `\n<ul>${items}</ul>`;
	});

	// 9) Single newlines → <br> (skip after block elements)
	html = html.replace(/(?<!<\/(?:h[1-6]|li|ul|pre)>)\n/g, "<br>");

	// 10) Restore book refs as hoverable citation links
	bookRefs.forEach(({ id, title }, index) => {
		const safeId = escapeHtml(id);
		const safeTitle = escapeHtml(title);
		const link = `<a href="/book/${safeId}" class="inline-book-ref" data-book-id="${safeId}">${safeTitle}</a>`;
		html = html.replace(`\x00BOOKREF${index}\x00`, link);
	});

	return html;
}

// ---------------------------------------------------------------------------
// Book hover card
// ---------------------------------------------------------------------------

function showBookHoverCard(link, bookId) {
	const book = bookDataMap[String(bookId)];
	if (!book) return;

	document.querySelector(".book-hover-card")?.remove();

	const card = document.createElement("div");
	card.className = "book-hover-card";

	let html = '<div class="book-hover-card-content">';

	if (book.cover_id) {
		const coverUrl = `https://covers.openlibrary.org/b/id/${book.cover_id}-M.jpg`;
		html += `<img src="${coverUrl}" alt="Cover" class="book-hover-cover" onerror="this.style.display='none'" />`;
	}
	html += `<div class="book-hover-title">${escapeHtml(book.title ?? "")}</div>`;
	if (book.author) {
		html += `<div class="book-hover-author">${escapeHtml(book.author)}</div>`;
	}
	if (book.year) {
		html += `<div class="book-hover-year">${escapeHtml(String(book.year))}</div>`;
	}
	if (book.num_ratings) {
		html += `<div class="book-hover-ratings">${book.num_ratings.toLocaleString()} ratings</div>`;
	}
	if (book.genre) {
		html += `<div class="book-hover-genre">${escapeHtml(book.genre)}</div>`;
	}

	html += "</div>";
	card.innerHTML = html;
	document.body.appendChild(card);

	// Position: prefer right of link, clamp to viewport
	const rect = link.getBoundingClientRect();
	const W = 220;
	let left = rect.right + 12;
	let top = rect.top;
	if (left + W > window.innerWidth - 10) left = rect.left - W - 12;
	if (top + card.offsetHeight > window.innerHeight - 10) {
		top = window.innerHeight - card.offsetHeight - 10;
	}
	card.style.left = `${left}px`;
	card.style.top = `${top}px`;

	const remove = () => card.remove();
	link.addEventListener("mouseleave", remove);
	card.addEventListener("mouseleave", remove);
}

// ---------------------------------------------------------------------------
// Rate-limit UX
// ---------------------------------------------------------------------------

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
	const wrap = showBanner("warning", `${label} — wait ${remaining}s`, {
		autoHideMs: 0,
		withClose: false,
		role: "status",
		container: CHAT_CONTAINER,
	});
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
	showBanner("warning", `${detail} Resets at midnight.`, {
		autoHideMs: 0,
		withClose: false,
		container: CHAT_CONTAINER,
	});
}

function ensureUsageBar() {
	let bar = document.getElementById("rlUsage");
	if (bar) return bar;
	bar = el("div", "rl-usage");
	bar.id = "rlUsage";
	bar.style.cssText = "margin-top:8px; font-size:12px; opacity:0.85;";
	document.getElementById("chatBox")?.appendChild(bar);
	return bar;
}

function updateUsageFromHeaders(headers) {
	const get = (k) => headers.get(k);
	const dayC = get("X-RateLimit-Day-Count");
	const dayL = get("X-RateLimit-Day-Limit");
	const minC = get("X-RateLimit-Min-Count");
	const minL = get("X-RateLimit-Min-Limit");
	const sysC = get("X-RateLimit-System-Day-Count");
	const sysL = get("X-RateLimit-System-Day-Limit");
	if (!dayC || !dayL || !minC || !minL || !sysC || !sysL) return;
	ensureUsageBar().textContent = `Today: ${dayC}/${dayL} • Per minute: ${minC}/${minL} • System: ${sysC}/${sysL}`;
}

// ---------------------------------------------------------------------------
// SSE parser
// ---------------------------------------------------------------------------

/**
 * Async generator that reads a fetch Response body as Server-Sent Events
 * and yields parsed JSON objects from `data: {...}` lines.
 *
 * @param {Response} response
 * @yields {object}
 */
async function* parseSseStream(response) {
	const reader = response.body.getReader();
	const decoder = new TextDecoder();
	let buffer = "";

	try {
		while (true) {
			const { done, value } = await reader.read();
			if (done) break;
			buffer += decoder.decode(value, { stream: true });

			// SSE lines are separated by \n; events by \n\n
			const lines = buffer.split("\n");
			// Keep the last (possibly incomplete) line in the buffer
			buffer = lines.pop() ?? "";

			for (const line of lines) {
				const trimmed = line.trim();
				if (!trimmed.startsWith("data:")) continue;
				const raw = trimmed.slice(5).trim();
				if (!raw || raw === "[DONE]") continue;
				try {
					yield JSON.parse(raw);
				} catch {
					// Malformed JSON — skip
				}
			}
		}
	} finally {
		reader.releaseLock();
	}
}

// ---------------------------------------------------------------------------
// Core send function
// ---------------------------------------------------------------------------

/**
 * Send a user message, stream the response, and update the UI progressively.
 *
 * Chunk type contract (from /chat/stream):
 *   { type: "status",   content: string }          — progress label
 *   { type: "token",    content: string }           — response characters
 *   { type: "complete", data: { book_ids, books, target, success, ... } }
 *
 * @param {string} text
 * @param {HTMLElement} messagesEl
 */
async function sendChatMessage(text, messagesEl) {
	// Optimistic bot bubble shown immediately as a status placeholder
	const { bubble, inner: bubbleInner } = createBotBubble(messagesEl);
	bubbleInner.classList.add("chat-status-text");
	bubbleInner.textContent = "Thinking…";

	const useProfile = !!document.getElementById("useProfile")?.checked;
	const payload = { user_text: text, use_profile: useProfile };

	try {
		setDisabled(true);

		const resp = await fetch("/chat/stream", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			credentials: "same-origin",
			body: JSON.stringify(payload),
		});

		// Rate-limit headers are on the initial response, before the body is read
		updateUsageFromHeaders(resp.headers);

		// --- Non-200 handling (no body to stream) ---
		if (resp.status === 401) {
			bubble.remove();
			showBanner("warning", "Please log in to use the chatbot.", {
				autoHideMs: 7000,
				container: CHAT_CONTAINER,
			});
			appendMessage(messagesEl, "bot", "Login required.");
			setDisabled(false);
			return;
		}

		if (resp.status === 429) {
			bubble.remove();
			const reason = resp.headers.get("X-RateLimit-Block-Reason") ?? "unknown";
			const retryAfter = parseInt(resp.headers.get("Retry-After") ?? "60", 10);
			let detail = "Rate limit exceeded.";
			try {
				const j = await resp.json();
				if (j?.detail) detail = j.detail;
			} catch { /* ignore */ }

			if (reason === "identity_minute") {
				startCountdown(retryAfter, "You are sending messages too quickly");
			} else if (reason === "identity_day" || reason === "system_day") {
				showStopForDay(detail);
			} else {
				startCountdown(retryAfter, detail);
			}
			appendMessage(messagesEl, "bot", detail);
			return;
		}

		if (!resp.ok) {
			bubble.remove();
			showBanner("danger", `Error ${resp.status}: ${resp.statusText || "Something went wrong."}`, {
				autoHideMs: 7000,
				container: CHAT_CONTAINER,
			});
			appendMessage(messagesEl, "bot", "Something went wrong while contacting the agent.");
			setDisabled(false);
			return;
		}

		// --- Stream processing ---
		let streamingStarted = false;
		let accumulatedText = "";

		for await (const chunk of parseSseStream(resp)) {
			if (chunk.type === "status") {
				// Update the placeholder in-place while LLM is still working
				bubbleInner.textContent = chunk.content ?? "Processing…";

			} else if (chunk.type === "token" && chunk.content) {
				if (!streamingStarted) {
					// Transition from status placeholder to streaming mode
					streamingStarted = true;
					bubbleInner.classList.remove("chat-status-text");
					bubbleInner.innerHTML = "";
					bubble.classList.add("is-streaming");
				}
				accumulatedText += chunk.content;
				// Re-render the whole accumulated text on each token so that
				// partial Markdown (e.g. an opening **) resolves cleanly once complete.
				// This is safe because renderMarkdown is pure and fast for chat-length text.
				bubbleInner.innerHTML = renderMarkdown(accumulatedText);
				messagesEl.scrollTop = messagesEl.scrollHeight;

			} else if (chunk.type === "complete") {
				// Finalise: remove streaming cursor, attach hover handlers
				bubble.classList.remove("is-streaming");

				// Ensure the final rendered text is in place (handles edge case where
				// no tokens arrived but a complete chunk was still emitted)
				if (accumulatedText) {
					bubbleInner.innerHTML = renderMarkdown(accumulatedText);
					attachBookLinkHandlers(bubble);
				}

				// Populate bookDataMap from full book objects included in the chunk
				const books = chunk.data?.books ?? [];
				books.forEach((book) => {
					if (book?.item_idx != null) {
						bookDataMap[String(book.item_idx)] = book;
					}
				});

				// Attach hover handlers now that bookDataMap is populated
				attachBookLinkHandlers(bubble);

				messagesEl.scrollTop = messagesEl.scrollHeight;
				setDisabled(false);
			}
		}

		// Guard: if the stream closed without a complete chunk (e.g. server crash)
		if (streamingStarted) {
			bubble.classList.remove("is-streaming");
			attachBookLinkHandlers(bubble);
		}
		setDisabled(false);

	} catch (err) {
		bubble.remove();
		showBanner("danger", "Network error. Please try again.", {
			autoHideMs: 7000,
			container: CHAT_CONTAINER,
		});
		appendMessage(messagesEl, "bot", "Something went wrong while contacting the agent.");
		setDisabled(false);
		console.error(err);
	}
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

window.addEventListener("DOMContentLoaded", () => {
	try { mountBanners({ autoHideMs: 0 }); } catch { /* non-fatal */ }

	const messagesEl = document.getElementById("chatMessages");
	const inputEl = document.getElementById("chatInput");
	const sendBtn = document.getElementById("chatSend");
	const useProfileEl = document.getElementById("useProfile");

	// Persist profile toggle across sessions
	if (useProfileEl) {
		const saved = localStorage.getItem("chat.useProfile");
		if (saved !== null) useProfileEl.checked = saved === "true";
		useProfileEl.addEventListener("change", () => {
			localStorage.setItem("chat.useProfile", String(useProfileEl.checked));
		});
	}

	if (!messagesEl || !inputEl || !sendBtn) return;

	const send = async () => {
		const text = (inputEl.value ?? "").trim();
		if (!text || sendBtn.disabled || inputEl.disabled) return;
		appendMessage(messagesEl, "user", text);
		inputEl.value = "";
		await sendChatMessage(text, messagesEl);
	};

	sendBtn.addEventListener("click", send);
	inputEl.addEventListener("keydown", async (e) => {
		if (e.key === "Enter" && !e.shiftKey) {
			e.preventDefault();
			await send();
		}
	});
});
