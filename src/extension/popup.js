document.addEventListener('DOMContentLoaded', () => {
  const urlInput = document.getElementById('urlInput');
  const scanBtn = document.getElementById('scanBtn');
  const resultBox = document.getElementById('result');
  const statusBadge = document.getElementById('statusBadge');
  const confidenceEl = document.getElementById('confidence');
  const latencyEl = document.getElementById('latency');

  // Auto-fill active tab URL
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0] && tabs[0].url) {
      urlInput.value = tabs[0].url;
    }
  });

  scanBtn.addEventListener('click', async () => {
    const url = urlInput.value.trim();
    if (!url) return;

    scanBtn.innerText = "Analyzing...";
    
    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url })
      });

      if (response.ok) {
        const data = await response.json();
        resultBox.classList.remove('hidden');

        if (data.is_phishing) {
          statusBadge.innerText = "🚨 PHISHING DETECTED";
          statusBadge.className = "badge phishing";
        } else {
          statusBadge.innerText = "✅ SAFE URL";
          statusBadge.className = "badge safe";
        }

        confidenceEl.innerText = (data.confidence_score * 100).toFixed(2) + "%";
        latencyEl.innerText = data.latency_ms.toFixed(2) + " ms";
      } else {
        alert("API Error: Backend server returned non-200 status.");
      }
    } catch (err) {
      alert("Error connecting to local FastAPI server on http://127.0.0.1:8000");
    } finally {
      scanBtn.innerText = "Scan URL";
    }
  });
});
