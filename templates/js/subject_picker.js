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

    let selectedSubjects = [];

    async function fetchSubjects(query = "") {
        const res = await fetch(`/subjects/suggestions?q=${encodeURIComponent(query)}`);
        const data = await res.json();
        return data.subjects;
    }

    input.addEventListener("input", async function () {
        const query = this.value.trim();
        suggestionsBox.innerHTML = "";

        if (selectedSubjects.length >= maxSubjects) return;

        const subjects = await fetchSubjects(query);
        subjects.forEach(subject => {
            if (!selectedSubjects.includes(subject)) {
                const item = document.createElement("div");
                item.classList.add("suggestion");
                item.textContent = subject;
                item.onclick = () => addSubject(subject);
                suggestionsBox.appendChild(item);
            }
        });
    });

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
    }

    function updateHiddenField() {
        hiddenInput.value = selectedSubjects.join(",");
    }
}