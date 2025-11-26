import { mountBanners, showBanner } from "/static/js/flash_alert.js";
const CHAT_CONTAINER = document.getElementById("chatBox");

// Store book data for hover cards (indexed by item_idx)
let bookDataMap = {};

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

	// Attach hover handlers to inline book links
	if (role === "bot") {
		const bookLinks = msg.querySelectorAll('a.inline-book-ref[data-book-id]');
		bookLinks.forEach(link => {
			link.addEventListener('mouseenter', (e) => {
				const bookId = link.getAttribute('data-book-id');
				showBookHoverCard(link, bookId);
			});
		});
	}
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
	return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// Show a book hover card near a link
function showBookHoverCard(link, bookId) {
	const book = bookDataMap[bookId];
	if (!book) return;

	// Remove any existing hover card
	const existing = document.querySelector('.book-hover-card');
	if (existing) existing.remove();

	// Create card
	const card = document.createElement('div');
	card.className = 'book-hover-card';

	let html = '<div class="book-hover-card-content">';

	// Cover
	if (book.cover_id) {
		const coverUrl = `https://covers.openlibrary.org/b/id/${book.cover_id}-M.jpg`;
		html += `<img src="${coverUrl}" alt="Cover" class="book-hover-cover" onerror="this.style.display='none'" />`;
	}

	// Title
	html += `<div class="book-hover-title">${escapeHtml(book.title || '')}</div>`;

	// Author
	if (book.author) {
		html += `<div class="book-hover-author">${escapeHtml(book.author)}</div>`;
	}

	// Year
	if (book.year) {
		html += `<div class="book-hover-year">${escapeHtml(String(book.year))}</div>`;
	}

	// Rating count (if available)
	if (book.num_ratings) {
		html += `<div class="book-hover-ratings">${book.num_ratings.toLocaleString()} ratings</div>`;
	}

	// Genre
	if (book.genre) {
		html += `<div class="book-hover-genre">${escapeHtml(book.genre)}</div>`;
	}

	html += '</div>';
	card.innerHTML = html;
	document.body.appendChild(card);

	// Position near link
	const rect = link.getBoundingClientRect();
	const cardWidth = 220;
	const cardHeight = card.offsetHeight;

	let left = rect.right + 12; // Right of link
	let top = rect.top;

	// Adjust if too far right
	if (left + cardWidth > window.innerWidth - 10) {
		left = rect.left - cardWidth - 12; // Left of link instead
	}

	// Adjust if too far down
	if (top + cardHeight > window.innerHeight - 10) {
		top = window.innerHeight - cardHeight - 10;
	}

	card.style.left = left + 'px';
	card.style.top = top + 'px';

	// Remove on mouse leave
	link.addEventListener('mouseleave', () => {
		card.remove();
	});

	card.addEventListener('mouseleave', () => {
		card.remove();
	});
}

function renderMarkdown(md) {
	// 1) Extract inline book references before escaping
	const bookRefs = [];
	let bookCounter = 0;
	let html = md.replace(/<book\s+id="(\d+)">([^<]+)<\/book>/g, (match, id, title) => {
		bookRefs.push({ id: String(id), title: String(title) });
		return `__BOOK_REF_${bookCounter++}__`;
	});

	// 2) escape everything else
	html = escapeHtml(html);

	// 3) fenced code blocks ``` ```
	html = html.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code}</code></pre>`);

	// 4) inline code `code`
	html = html.replace(/`([^`\n]+)`/g, '<code>$1</code>');

	// 5) links [text](http://...)
	html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

	// 6) bold **text** and italic *text*
	html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
	html = html.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');

	// 7) headings (# to ######) at line start
	html = html.replace(/^###### (.*)$/gm, '<h6>$1</h6>')
		.replace(/^##### (.*)$/gm, '<h5>$1</h5>')
		.replace(/^#### (.*)$/gm, '<h4>$1</h4>')
		.replace(/^### (.*)$/gm, '<h3>$1</h3>')
		.replace(/^## (.*)$/gm, '<h2>$1</h2>')
		.replace(/^# (.*)$/gm, '<h1>$1</h1>');

	// 8) simple unordered lists: lines starting with - or *
	html = html.replace(/(?:^|\n)([-*] .+(?:\n[-*] .+)*)/g, (block) => {
		const items = block.trim().split(/\n/).map(line => line.replace(/^[-*] (.*)$/, '<li>$1</li>')).join('');
		return `\n<ul>${items}</ul>`;
	});

	// 9) line breaks for single newlines (outside of code/blocks)
	html = html.replace(/(?<!<\/h\d>|<\/li>|<\/ul>|<\/pre>)\n/g, '<br>');

	// 10) Restore inline book references as clickable links
	bookRefs.forEach((ref, index) => {
		const bookLink = `<a href="/book/${escapeHtml(ref.id)}" class="inline-book-ref" data-book-id="${escapeHtml(ref.id)}">${escapeHtml(ref.title)}</a>`;
		html = html.replace(`__BOOK_REF_${index}__`, bookLink);
	});

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

	// FIX 1: use ChatIn schema → user_text
	const payload = { user_text: text, use_profile: useProfile };

	try {
		setDisabled(true);
		const resp = await fetch("/chat/agent", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			credentials: "same-origin",
			body: JSON.stringify(payload),
		});

		if (resp.ok) {
			updateUsageFromHeaders(resp.headers);
			const data = await resp.json().catch(() => ({}));

			thinking.remove();

			// FIX 2: ChatOut uses 'text' (not 'reply')
			const reply = (data && typeof data.text === "string" ? data.text : "Sorry, I couldn't generate a response.");
			appendMessage(messagesEl, "bot", reply);

			// Store book data for hover cards
			if (Array.isArray(data?.books) && data.books.length) {
				data.books.forEach(book => {
					bookDataMap[book.item_idx] = book;
				});
			}

			if (Array.isArray(data?.books) && data.books.length) {
				const gridHost = document.getElementById("chatBookResults");
				if (gridHost) {
					gridHost.classList.remove("book-grid");
					gridHost.innerHTML = "";

					const mod = await import("/static/js/paginated_books.js");
					mod.setupPaginatedBookDisplay({
						books: data.books,
						containerId: "chatBookResults",
						manualPagination: true,
						scrollOnFirstRender: true
					});

					gridHost.scrollIntoView({ behavior: "smooth", block: "start" });
				}
			}
			setDisabled(false);
			return;
		}

		if (resp.status === 401) {
			thinking.remove();
			showBanner("warning", "Please log in to use the chatbot.", { autoHideMs: 7000, container: CHAT_CONTAINER });
			setDisabled(false);
			appendMessage(messagesEl, "bot", "Login required.");
			return;
		}

		if (resp.status === 429) {
			thinking.remove();
			const reason = resp.headers.get("X-RateLimit-Block-Reason") || "unknown";
			const retryAfter = parseInt(resp.headers.get("Retry-After") || "60", 10);
			let detail = "Rate limit exceeded.";
			try {
				const j = await resp.json();
				if (j?.detail) detail = j.detail;
			} catch { }
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

		thinking.remove();
		showBanner("danger", `Error ${resp.status}: ${resp.statusText || "Something went wrong."}`, { autoHideMs: 7000, container: CHAT_CONTAINER });
		appendMessage(messagesEl, "bot", "Something went wrong while contacting the agent.");
		setDisabled(false);

	} catch (err) {
		thinking.remove();
		showBanner("danger", "Network error. Please try again.", { autoHideMs: 7000, container: CHAT_CONTAINER });
		appendMessage(messagesEl, "bot", "Something went wrong while contacting the agent.");
		setDisabled(false);
		console.error(err);
	}
}

// ---- Boot ----
window.addEventListener("DOMContentLoaded", () => {
	try { mountBanners({ autoHideMs: 0 }); } catch { }

	const messagesEl = document.getElementById("chatMessages");
	const inputEl = document.getElementById("chatInput");
	const sendBtn = document.getElementById("chatSend");

	// toggles (persist)
	const useProfileEl = document.getElementById("useProfile");
	const restrictEl = document.getElementById("restrictToCatalog");

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
