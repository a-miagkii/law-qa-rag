(function () {
  function ensureOverlay() {
    let overlay = document.getElementById("loadingOverlay");
    if (overlay) {
      return overlay;
    }

    overlay = document.createElement("div");
    overlay.id = "loadingOverlay";
    overlay.className = "loading-overlay";
    overlay.hidden = true;
    overlay.setAttribute("aria-hidden", "true");
    overlay.innerHTML = [
      '<div class="loading-card" role="status" aria-live="polite">',
      '  <span class="spinner" aria-hidden="true"></span>',
      "  <div>",
      "    <strong>Формируем ответ</strong>",
      "    <p>Идет поиск по корпусу и подготовка цитат.</p>",
      "  </div>",
      "</div>",
    ].join("");
    document.body.appendChild(overlay);
    return overlay;
  }

  function showLoading(form) {
    const overlay = ensureOverlay();

    document.body.classList.add("is-loading");
    overlay.hidden = false;
    overlay.setAttribute("aria-hidden", "false");

    const buttons = document.querySelectorAll("button");
    buttons.forEach((button) => {
      button.disabled = true;
    });

    const submitButton = form.querySelector("button[type='submit']");
    if (submitButton) {
      const label = submitButton.getAttribute("data-loading-label");
      const textNode = submitButton.querySelector("span");
      if (label && textNode) {
        textNode.textContent = label;
      }
    }
  }

  document.querySelectorAll("form[data-ask-form]").forEach((form) => {
    form.addEventListener("submit", () => {
      if (!form.checkValidity()) {
        return;
      }
      showLoading(form);
    });
  });
})();
