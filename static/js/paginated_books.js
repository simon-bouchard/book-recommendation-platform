function escapeHtml(unsafe) {
  if (unsafe == null) return "";
  return String(unsafe)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

/* Build a tidy SVG “No cover” image as a data URI */
function placeholderDataURI(titleText = "No cover") {
  const svg = `
  <svg xmlns="http://www.w3.org/2000/svg" width="150" height="220" viewBox="0 0 150 220">
    <defs>
      <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#f1f1f1"/>
        <stop offset="100%" stop-color="#e7e7e7"/>
      </linearGradient>
    </defs>
    <rect width="150" height="220" fill="url(#g)" />
    <rect x="10" y="10" width="130" height="200" rx="6" ry="6" fill="none" stroke="#cfcfcf" stroke-dasharray="6 4" />
    <text x="75" y="110" font-size="18" text-anchor="middle" fill="#7a7a7a" font-family="Arial, sans-serif">No cover</text>
    <text x="75" y="132" font-size="10" text-anchor="middle" fill="#9a9a9a" font-family="Arial, sans-serif">
      ${escapeHtml(titleText).slice(0, 26)}
    </text>
  </svg>`;
  return "data:image/svg+xml;utf8," + encodeURIComponent(svg);
}

/* Always render an <img>; if it 404s or there’s no id, swap to SVG placeholder */
function renderCoverImg(book) {
  const cid = book?.cover_id;
  const alt = `Cover for ${escapeHtml(book?.title || "")}`;
  const fallback = placeholderDataURI(book?.title || "No cover");

  // If no cover_id is provided (null/undefined/""/"0"), render placeholder immediately
  const hasCover = cid !== null && cid !== undefined && String(cid).trim() !== "" && String(cid) !== "0";
  if (!hasCover) {
    return `<img src="${fallback}" alt="${alt}" class="book-cover-img" />`;
  }

  // If there *is* a cover_id, try OpenLibrary first; on error, swap to the SVG placeholder
  const src = `https://covers.openlibrary.org/b/id/${cid}-M.jpg`;
  return `<img src="${src}" alt="${alt}" class="book-cover-img"
              onerror="this.onerror=null; this.src='${fallback}';" />`;
}

export function setupPaginatedBookDisplay({
  books,
  containerId,
  prevButtonId,
  nextButtonId,
  limit = 30,
  scrollOnFirstRender = false,
  manualPagination = false
}) {
  let offset = 0;
  let isFirstRender = true;

  const container = document.getElementById(containerId);
  const prevBtn = document.getElementById(prevButtonId);
  const nextBtn = document.getElementById(nextButtonId);

  function renderPage() {
    const page = manualPagination ? books : books.slice(offset, offset + limit);
    if (page.length === 0) {
      container.innerHTML = "<p>No results to display.</p>";
      return;
    }

    const html = page.map(book => `
      <a href="/book/${book.item_idx}">
        <div class="book-card">
          ${renderCoverImg(book)}
          <h3 class="book-title">
            <span class="book-title-text">${escapeHtml(book.title)}</span>
          </h3>
          ${book.author ? `<p class="author"><i>${escapeHtml(book.author)}</i></p>` : ""}
          ${book.year ? `<p class="year">${escapeHtml(String(book.year))}</p>` : ""}
          ${book.isbn ? `<p class="isbn">${escapeHtml(book.isbn)}</p>` : ""}
        </div>
      </a>
    `).join("");

    container.innerHTML = `<div class="book-grid">${html}</div>`;
    container.style.display = "block";

    if (!manualPagination) {
      prevBtn.style.display = offset > 0 ? "inline-block" : "none";
      nextBtn.style.display = offset + limit < books.length ? "inline-block" : "none";
    }

    if (!isFirstRender || scrollOnFirstRender) {
      container.scrollIntoView({ behavior: "smooth" });
    }
    isFirstRender = false;
  }

  if (!manualPagination) {
    prevBtn.onclick = () => { offset = Math.max(0, offset - limit); renderPage(); };
    nextBtn.onclick = () => { offset += limit; renderPage(); };
  }

  renderPage();
}
