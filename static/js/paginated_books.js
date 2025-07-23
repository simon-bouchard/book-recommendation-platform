/**
 * Paginated rendering of books into a container with buttons.
 * 
 * @param {Object} config
 * @param {Array} config.books - List of book objects to render
 * @param {string} config.containerId - ID of the div to render into
 * @param {string} config.prevButtonId - ID of the "Previous" button
 * @param {string} config.nextButtonId - ID of the "Next" button
 * @param {number} [config.limit=30] - Books per page
 * @param {boolean} [config.showSimilarity=false] - Show similarity score
 */
export function setupPaginatedBookDisplay({
    books,
    containerId,
    prevButtonId,
    nextButtonId,
    limit = 30,
    showSimilarity = false,
    scrollOnFirstRender = false
}) {
    let offset = 0;
    let isFirstRender = true;

    const container = document.getElementById(containerId);
    const prevBtn = document.getElementById(prevButtonId);
    const nextBtn = document.getElementById(nextButtonId);

    function renderPage() {
        const page = books.slice(offset, offset + limit);
        if (page.length === 0) {
            container.innerHTML = "<p>No results to display.</p>";
            return;
        }

        const html = page.map(book => `
            <a href="/book/${book.item_idx}">
                <div class="book-card">
                    <img src="${book.cover_id 
                        ? `https://covers.openlibrary.org/b/id/${book.cover_id}-M.jpg`
                        : 'https://via.placeholder.com/150x220?text=No+Cover'}"
                        alt="Cover for ${book.title}">
                    <h3>${book.title}</h3>
                    ${book.author ? `<p><strong>Author:</strong> ${book.author}</p>` : ''}
                    ${book.year ? `<p><strong>Year:</strong> ${book.year}</p>` : ''}
                    ${book.isbn ? `<p><strong>ISBN:</strong> ${book.isbn}</p>` : ''}
                    ${showSimilarity && book.score !== undefined 
                        ? `<p><strong>Similarity:</strong> ${book.score.toFixed(3)}</p>` 
                        : ''}
                </div>
            </a>
        `).join('');

        container.innerHTML = `<div class="book-grid">${html}</div>`;
        container.style.display = "block";

        prevBtn.style.display = offset > 0 ? "inline-block" : "none";
        nextBtn.style.display = offset + limit < books.length ? "inline-block" : "none";

        if (!isFirstRender || scrollOnFirstRender) {
            container.scrollIntoView({ behavior: "smooth" });
        }
        isFirstRender = false;
    }

    prevBtn.onclick = () => {
        offset = Math.max(0, offset - limit);
        renderPage();
    };

    nextBtn.onclick = () => {
        offset += limit;
        renderPage();
    };

    renderPage();
}
