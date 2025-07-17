export function initSubjectPicker({
    inputId,
    suggestionsBoxId,
    pillsContainerId,
    hiddenInputId,
    maxSubjects = 5
}) {
    const input = document.getElementById(inputId);
    const suggestionsBox = document.getElementById(suggestionsBoxId);
    const pillsContainer = document.getElementById(pillsContainerId);
    const hiddenInput = document.getElementById(hiddenInputId);
    
    const form = input.closest("form");

    let selectedSubjects = [];
    let allSuggestions = [];
    
    if (form) {
        form.addEventListener("submit", () => {
            input.value = "";
            selectedSubjects = [...new Set(selectedSubjects.map(s => s.trim()).filter(Boolean))];
            updateHiddenField();
        });
    }

    async function fetchSubjects(query = "") {
        const res = await fetch(`/subjects/suggestions?q=${encodeURIComponent(query)}`);
        const data = await res.json();
        return data.subjects;
    }

    async function showSuggestions(query = "") {
        suggestionsBox.innerHTML = "";

        if (selectedSubjects.length >= maxSubjects) return;

        const suggestions = await fetchSubjects(query);

        suggestions.forEach(s => {
            const subject = typeof s === "string" ? s : s.subject;
            const count = typeof s === "object" ? s.count : null;

            if (!selectedSubjects.includes(subject)) {
                const item = document.createElement("div");
                item.classList.add("suggestion");
                item.innerHTML = `
                    <span class="subject-name">${subject}</span>
                    ${count !== null ? `<span class="subject-count">(${count})</span>` : ""}
                `;
                item.onclick = () => addSubject(subject);
                suggestionsBox.appendChild(item);
            }
        });

        suggestionsBox.style.display = suggestionsBox.children.length ? "block" : "none";
    }

    function addSubject(subject) {
        if (selectedSubjects.length >= maxSubjects || selectedSubjects.includes(subject)) return;

        selectedSubjects.push(subject);
        updateHiddenField();

        const pill = document.createElement("div");
        pill.classList.add("pill");
        pill.textContent = subject;

        const x = document.createElement("span");
        x.textContent = "×";
        x.classList.add("pill-remove");
        x.onclick = () => {
            selectedSubjects = selectedSubjects.filter(s => s !== subject);
            pill.remove();
            updateHiddenField();
        };

        pill.appendChild(x);
        pillsContainer.appendChild(pill);
        input.value = "";
        suggestionsBox.innerHTML = "";
    }

    function updateHiddenField() {
        hiddenInput.value = selectedSubjects.join(",");
    }

    input.addEventListener("input", async function () {
        const query = this.value.trim();
        await showSuggestions(query);
    });

    input.addEventListener("focus", async function () {
        if (input.value.trim() === "") {
            await showSuggestions("");
        }
    });

    document.addEventListener("click", function (e) {
        if (!suggestionsBox.contains(e.target) && e.target !== input) {
            suggestionsBox.style.display = "none";
        }
    });

    if (hiddenInput.value) {
        const initialSubjects = hiddenInput.value.split(",").map(s => s.trim()).filter(Boolean);
        initialSubjects.forEach(subject => addSubject(subject));
    }

    input.addEventListener("keydown", function (e) {
        if (e.key === "Enter") {
            e.preventDefault();
        }
    });

}