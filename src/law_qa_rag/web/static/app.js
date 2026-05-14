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

  const authModal = document.querySelector("[data-auth-modal]");

  function setAuthMode(mode) {
    if (!authModal) {
      return;
    }
    const safeMode = mode === "register" ? "register" : "login";
    authModal.dataset.mode = safeMode;
    authModal.querySelectorAll("[data-auth-tab]").forEach((tab) => {
      tab.setAttribute("aria-selected", tab.dataset.authTab === safeMode ? "true" : "false");
    });
    authModal.querySelectorAll("[data-auth-panel]").forEach((panel) => {
      panel.hidden = panel.dataset.authPanel !== safeMode;
    });
  }

  function openAuth(mode) {
    if (!authModal) {
      return;
    }
    setAuthMode(mode || authModal.dataset.mode || "login");
    authModal.classList.add("is-open");
    authModal.setAttribute("aria-hidden", "false");
    document.body.classList.add("has-modal");
    const firstInput = authModal.querySelector("[data-auth-panel]:not([hidden]) input:not([type='hidden'])");
    if (firstInput) {
      firstInput.focus();
    }
  }

  function closeAuth() {
    if (!authModal) {
      return;
    }
    authModal.classList.remove("is-open");
    authModal.setAttribute("aria-hidden", "true");
    document.body.classList.remove("has-modal");
  }

  document.querySelectorAll("[data-auth-open]").forEach((trigger) => {
    trigger.addEventListener("click", (event) => {
      event.preventDefault();
      const exampleQuestion = trigger.getAttribute("data-example-question");
      const questionInput = document.querySelector(".question-input");
      if (exampleQuestion && questionInput) {
        questionInput.value = exampleQuestion;
      }
      openAuth(trigger.getAttribute("data-auth-open"));
    });
  });

  if (authModal) {
    authModal.querySelectorAll("[data-auth-tab]").forEach((tab) => {
      tab.addEventListener("click", () => setAuthMode(tab.dataset.authTab));
    });
    authModal.querySelectorAll("[data-auth-close]").forEach((trigger) => {
      trigger.addEventListener("click", closeAuth);
    });
    if (authModal.dataset.open === "true") {
      openAuth(authModal.dataset.mode);
    } else {
      setAuthMode(authModal.dataset.mode);
    }
  }

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeAuth();
    }
  });
})();
