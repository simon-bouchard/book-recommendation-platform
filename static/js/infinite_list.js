export function setupInfiniteList({
    listEl,
    errorEl,
    emptyEl,
    countEl,
    pageSize = 10,
    fetchPage,
    renderItem,
    rootEl
}) {
    if (!listEl) throw new Error("listEl required");
    const root = rootEl || listEl;

    const sentinel = document.createElement("div");
    sentinel.style.height = "1px";
    sentinel.dataset.role = "sentinel";
    listEl.appendChild(sentinel);

    let nextCursor = null;
    let isLoading = false;
    let totalCount = 0;
    let lastCursorRequested = "__none__";
    let io = null;

    async function load(initial = false) {
        if (isLoading) return;
        const requestedCursor = nextCursor ?? "__first__";
        if (!initial && requestedCursor === lastCursorRequested) return;
        lastCursorRequested = requestedCursor;

        isLoading = true;
        try {
            const { items, total_count, has_more, next_cursor } = await fetchPage({
                cursor: nextCursor,
                limit: pageSize
            });

            const arr = Array.isArray(items) ? items : [];

            if (initial && arr.length === 0) {
                if (emptyEl) emptyEl.style.display = "block";
                nextCursor = null;
                if (io) io.disconnect();
                return;
            }

            if (typeof total_count === "number") {
                totalCount = total_count;
                if (countEl && totalCount) countEl.textContent = `(${totalCount})`;
            }

            for (const it of arr) {
                listEl.insertBefore(renderItem(it), sentinel);
            }

            if (has_more === false || arr.length === 0) {
                nextCursor = null;
                if (io) io.disconnect();
            } else {
                nextCursor = next_cursor ?? null;
            }
        } catch (e) {
            if (errorEl) errorEl.style.display = "block";
            nextCursor = null;
            if (io) io.disconnect();
        } finally {
            isLoading = false;
        }
    }

    function start() {
        io = new IntersectionObserver((entries) => {
            if (entries.some(e => e.isIntersecting)) {
                if (!isLoading && nextCursor) load(false);
            }
        }, {
            root,
            rootMargin: "200px",
            threshold: 0.01
        });
        io.observe(sentinel);
        load(true);
    }

    function destroy() {
        if (io) io.disconnect();
    }

    function reset() {
        destroy();
        nextCursor = null;
        isLoading = false;
        lastCursorRequested = "__none__";
        listEl.innerHTML = "";
        listEl.appendChild(sentinel);
        if (emptyEl) emptyEl.style.display = "none";
        if (errorEl) errorEl.style.display = "none";
        if (countEl) countEl.textContent = "";
        start();
    }

    return { start, destroy, reset };
}

