for (const chip of document.querySelectorAll("[data-question]")) {
  chip.addEventListener("click", () => {
    const input = document.querySelector("#question");
    input.value = chip.dataset.question;
    input.focus();
  });
}

const menuToggle = document.querySelector(".menu-toggle");
const workspaceNavigation = document.querySelector("#workspace-navigation");

if (menuToggle && workspaceNavigation) {
  const closeMenu = () => {
    menuToggle.setAttribute("aria-expanded", "false");
    workspaceNavigation.classList.remove("is-open");
  };

  menuToggle.addEventListener("click", () => {
    const isOpen = menuToggle.getAttribute("aria-expanded") === "true";
    menuToggle.setAttribute("aria-expanded", String(!isOpen));
    workspaceNavigation.classList.toggle("is-open", !isOpen);
  });

  workspaceNavigation.addEventListener("click", (event) => {
    if (event.target.closest("a")) closeMenu();
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeMenu();
  });
}
