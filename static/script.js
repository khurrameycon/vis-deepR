document.addEventListener("DOMContentLoaded", () => {
    // UI Element References
    const providerInput = document.getElementById("providerInput");
    const runAgentBtn = document.getElementById("runAgentBtn");
    const btnText = document.getElementById("btn-text");
    const spinner = document.getElementById("spinner");
    const streamImg = document.getElementById("stream-img");
    const streamPlaceholder = document.getElementById("stream-placeholder");
    const logBox = document.getElementById("log-box");
    const finalResultBox = document.getElementById("final-result-box");
    const taskInput = document.getElementById("taskInput");

    let ws;

    // Fetch providers and populate dropdown
    fetch("/api/providers")
        .then(response => response.json())
        .then(providers => {
            for (const [key, value] of Object.entries(providers)) {
                const option = document.createElement("option");
                option.value = key;
                option.textContent = value;
                providerInput.appendChild(option);
            }
        });

    function setUILoading(isLoading) {
        if (isLoading) {
            runAgentBtn.disabled = true;
            btnText.textContent = "Running...";
            spinner.classList.remove("d-none");
        } else {
            runAgentBtn.disabled = false;
            btnText.textContent = "Run Agent";
            spinner.classList.add("d-none");
        }
    }

    function addLog(message) {
        if (logBox.querySelector(".text-muted")) {
            logBox.innerHTML = ""; // Clear initial message
        }
        const logEntry = document.createElement("p");
        logEntry.className = "mb-1";
        logEntry.textContent = `[LOG] ${message}`;
        logBox.appendChild(logEntry);
        logBox.scrollTop = logBox.scrollHeight; // Auto-scroll
    }

    function connectWebSocket() {
        const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws/stream`);

        ws.onopen = () => {
            console.log("WebSocket connection established.");
            streamPlaceholder.textContent = "Stream connected. Waiting for video...";
        };

        ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            switch (message.type) {
                case "stream":
                    streamPlaceholder.style.display = "none";
                    streamImg.style.display = "block";
                    streamImg.src = `data:image/jpeg;base64,${message.data}`;
                    break;
                case "log":
                    addLog(message.data);
                    break;
                case "result":
                    finalResultBox.innerHTML = `<p>${message.data}</p>`;
                    setUILoading(false); // Task is done
                    break;
                case "error":
                    addLog(`[ERROR] ${message.data}`);
                    finalResultBox.innerHTML = `<p class="text-danger">An error occurred: ${message.data}</p>`;
                    setUILoading(false);
                    break;
            }
        };

        ws.onclose = () => {
            console.log("WebSocket connection closed.");
            streamPlaceholder.textContent = "Stream ended. Ready for new task.";
            streamPlaceholder.style.display = "block";
            streamImg.style.display = "none";
            setUILoading(false);
        };

        ws.onerror = (error) => {
            console.error("WebSocket error:", error);
            addLog("Error connecting to stream.");
            setUILoading(false);
        };
    }

    runAgentBtn.addEventListener("click", async () => {
        const task = taskInput.value;
        const provider = providerInput.value;
        if (!task) {
            alert("Please enter a task.");
            return;
        }

        // Reset UI for new run
        logBox.innerHTML = '<p class="text-muted">Agent thoughts and actions will be logged here...</p>';
        finalResultBox.innerHTML = '<p class="text-muted">Result will appear here...</p>';
        streamPlaceholder.textContent = "Requesting agent start...";
        setUILoading(true);

        const response = await fetch("/agent/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ task: task, llm_provider: provider }),
        });

        const data = await response.json();
        console.log(data.status);

        if (response.ok && data.status.includes("started")) {
            if (ws) { ws.close(); }
            connectWebSocket();
        } else {
            alert("Failed to start agent: " + data.status);
            setUILoading(false);
        }
    });
});