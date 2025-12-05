// stats.js
document.addEventListener("DOMContentLoaded", function () {
  const sentimentLink = document.getElementById("sentimentAnalysisLink");
  const sentimentSection = document.getElementById("sentimentAnalysisSection");

  sentimentLink.addEventListener("click", function (e) {
    e.preventDefault();

    // Hide all sections
    document.querySelectorAll("section").forEach(sec => sec.classList.add("d-none"));

    // Load once
    if (!sentimentSection.dataset.loaded) {
      fetch("/stats")
        .then(response => response.json())
        .then(data => {
          renderInfoCards(data);
          createEmotionChart(data.emotion_counts);
          createAvgToxicityByEmotionChart(data.avg_toxicity_by_emotion);
          createHashtagChart(data.top_hashtags);
          createEmotionHashtagChart(data.emotion_hashtag_analysis);

          if (data.hashtag_sentiment_ratio) {
            createHashtagRatioChart(data.hashtag_sentiment_ratio);
          }
          if (data.toxicity_histogram) {
            createToxicityHistChart(data.toxicity_histogram);
          }

          sentimentSection.dataset.loaded = "true";
        })
        .catch(err => console.error("Error fetching /stats:", err));
    }

    sentimentSection.classList.remove("d-none");
    sentimentSection.scrollIntoView({ behavior: "smooth" });
  });
});

/* =========================
   Info Cards
========================= */
function renderInfoCards(data) {
  const container = document.getElementById("infoCardsContainer");
  container.innerHTML = "";

  const sentiment = data.sentiment_counts || {};
  const total = data.total || 0;

  const cardData = [
    { title: "Total Tweets", value: total },
    { title: "Positive Tweets", value: sentiment["POSITIVE"] || sentiment["Positive"] || 0 },
    { title: "Negative Tweets", value: sentiment["NEGATIVE"] || sentiment["Negative"] || 0 }
  ];

  const row = document.createElement("div");
  row.className = "row justify-content-center gap-3";

  cardData.forEach(card => {
    const col = document.createElement("div");
    col.className = "col-md-3 col-sm-5 mb-3";

    col.innerHTML = `
      <div class="card shadow border-0">
        <div class="card-body text-center">
          <h6 class="card-title text-muted">${card.title}</h6>
          <h4 class="card-text fw-bold">${card.value}</h4>
        </div>
      </div>
    `;

    row.appendChild(col);
  });

  container.appendChild(row);
}

/* =========================
   Emotion Distribution
========================= */
function createEmotionChart(data) {
  if (!data) return;
  const ctx = document.getElementById("sentimentEmotionChart").getContext("2d");
  const labels = Object.keys(data);
  const values = Object.values(data);

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: "Emotion Count",
        data: values,
        backgroundColor: "#4bc0c0"
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          beginAtZero: true,
          title: { display: true, text: 'Tweet Count' }
        }
      }
    }
  });
}

/* =========================
   Avg Toxicity by Emotion
========================= */
function createAvgToxicityByEmotionChart(data) {
  if (!data) return;
  const ctx = document.getElementById("toxicityChart").getContext("2d");
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: Object.keys(data),
      datasets: [{
        label: "Avg Toxicity Score",
        data: Object.values(data),
        backgroundColor: "#ff6384"
      }]
    },
    options: {
      indexAxis: 'y',
      scales: {
        x: { beginAtZero: true, max: 1 }
      }
    }
  });
}

/* =========================
   Top Hashtags (global)
========================= */
function createHashtagChart(items) {
  if (!items || !items.length) return;
  const labels = items.map(item => item[0]);
  const values = items.map(item => item[1]);

  const ctx = document.getElementById("hashtagChart").getContext("2d");
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: "Hashtag Frequency",
        data: values,
        backgroundColor: "#9966ff"
      }]
    },
    options: {
      indexAxis: 'y',
      scales: {
        x: { beginAtZero: true }
      }
    }
  });
}

/* =========================
   Emotion vs Hashtags (stacked)
========================= */
function createEmotionHashtagChart(data) {
  if (!data) return;

  const allHashtags = new Set();
  const emotions = Object.keys(data);

  emotions.forEach(emotion => {
    (data[emotion] || []).forEach(([tag]) => allHashtags.add(tag));
  });

  const labels = Array.from(allHashtags);
  const hueStep = emotions.length ? 360 / emotions.length : 0;

  const datasets = emotions.map((emotion, index) => {
    const hue = (index * hueStep) % 360;
    const color = `hsl(${hue}, 70%, 50%)`;
    const tagMap = Object.fromEntries(data[emotion] || []);
    const values = labels.map(tag => tagMap[tag] || 0);
    return {
      label: emotion,
      data: values,
      backgroundColor: color
    };
  });

  const ctx = document.getElementById("emotionHashtagChart").getContext("2d");
  new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: {
        legend: { position: 'top' },
        tooltip: { mode: 'index', intersect: false }
      },
      interaction: { mode: 'nearest', axis: 'y', intersect: false },
      scales: {
        x: { stacked: true, title: { display: true } },
        y: { stacked: true, title: { display: true, text: 'Hashtags' } }
      }
    }
  });
}

/* =========================
   Hashtag Sentiment Ratio
   items: [{hashtag, pos, neg, pos_ratio}]
========================= */
function createHashtagRatioChart(items) {
  if (!items || !items.length) return;
  const labels = items.map(x => x.hashtag);
  const ratios = items.map(x => (x.pos_ratio * 100).toFixed(1));

  const basePalette = [
  "#10AC84"
  ];
  const colors = labels.map((_, i) => basePalette[i % basePalette.length]);

  const ctx = document.getElementById("hashtagRatioChart").getContext("2d");
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: "Positive ratio (%)",
        data: ratios,
        backgroundColor: colors,
        borderColor: colors,
        borderWidth: 1
      }]
    },
    options: {
      indexAxis: 'y',
      scales: {
        x: {
          beginAtZero: true,
          max: 100,
          ticks: { callback: v => v + "%" }
        }
      }
    }
  });
}

/* =========================
   Toxicity Histogram
   hist: { bins:["0–0.2",...], counts:[...] }
========================= */
function createToxicityHistChart(hist) {
  if (!hist || !hist.bins || !hist.counts) return;

  const ctx = document.getElementById("toxHistChart").getContext("2d");
  const orange = "#FF9800"; // σταθερό πορτοκαλί για όλα τα bars

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: hist.bins,
      datasets: [{
        label: "Count",
        data: hist.counts,
        backgroundColor: orange,   // ένα χρώμα για όλα
        borderColor: orange,
        borderWidth: 1
      }]
    },
    options: {
      scales: {
        y: { beginAtZero: true }
      }
    }
  });
}


