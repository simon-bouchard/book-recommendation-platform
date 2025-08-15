// /static/js/flash_banner.js
function setupOne(alertEl, autoHideMs) {
  const wrap = alertEl.closest(".page-banner") || alertEl;
  let timer = null;

  const startTimer = () => {
    if (!autoHideMs) return;
    clearTimeout(timer);
    timer = setTimeout(() => wrap.remove(), autoHideMs);
  };

  const stopTimer = () => { if (timer) clearTimeout(timer); };

  alertEl.addEventListener("mouseenter", stopTimer);
  alertEl.addEventListener("mouseleave", startTimer);

  const closeBtn = alertEl.querySelector(".alert__close");
  if (closeBtn) {
    closeBtn.addEventListener("click", () => wrap.remove(), { once: true });
  }

  startTimer();
}

export function mountBanners({ root = document, autoHideMs = 5000 } = {}) {
  const alerts = root.querySelectorAll(".page-banner .alert");
  alerts.forEach((el) => setupOne(el, Number(el.dataset.autohide ?? autoHideMs)));
}

export function showBanner(
  type,
  text,
  {
    container = document.body,
    autoHideMs = 5000,
    withClose = true,
    role = "alert"
  } = {}
) {
  const wrap = document.createElement("div");
  wrap.className = "page-banner";

  const alert = document.createElement("div");
  alert.className = `alert alert--${type}`;
  alert.setAttribute("role", role);
  alert.textContent = "";
  const msg = document.createElement("div");
  msg.textContent = text;
  alert.appendChild(msg);

  if (withClose) {
    const btn = document.createElement("button");
    btn.className = "alert__close";
    btn.type = "button";
    btn.setAttribute("aria-label", "Close");
    btn.textContent = "✕";
    alert.appendChild(btn);
  }

  wrap.appendChild(alert);
  container.prepend(wrap);
  setupOne(alert, autoHideMs);
  return wrap;
}
