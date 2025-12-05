let liveTweets = [];
let lastFetchTime = 0;
let currentBatch = 1;
let isLoading = false;
const displayedTweetIds = new Set(); 



document.addEventListener("DOMContentLoaded", function () {
  const hamburger = document.querySelector('.toggle-btn');
  const toggler = document.querySelector('#icon');

  hamburger?.addEventListener('click', () => {
    document.querySelector('#sidebar').classList.toggle('expand');
    toggler?.classList.toggle("bx-chevrons-right");
    toggler?.classList.toggle("bx-chevrons-left");
  });

  const sections = {
    welcome: document.getElementById("welcomeSection"),
    userInput: document.getElementById("userInputSection"),
    liveFeed: document.getElementById("liveFeedSection"),
    sentimentAnalysis: document.getElementById("sentimentAnalysisSection"),
    trendingTopics: document.getElementById("trendingTopicsSection"),
    distilbert: document.getElementById("distilbertSection"),
    trainingStats: document.getElementById("trainingStatsSection")
  };

  const links = {
    home: document.getElementById("homeLink"),
    realTimeX: document.getElementById("realTimeXToggle"),
    sentimentAnalysis: document.getElementById("sentimentAnalysisLink"),
    trendingTopics: document.getElementById("trendingTopicsLink"),
    distilbert: document.getElementById("distilbertLink"),
    trainingStats: document.getElementById("trainingStatsLink")
  };

  function showOnly(sectionKey) {
    Object.values(sections).forEach(sec => sec?.classList.add("d-none"));
    sections[sectionKey]?.classList.remove("d-none");
  }

  function showHome() {
    Object.values(sections).forEach(sec => sec?.classList.add("d-none"));
    sections.welcome?.classList.remove("d-none");
    sections.userInput?.classList.remove("d-none");
    sections.liveFeed?.classList.add("d-none");
  }

  links.home?.addEventListener("click", (e) => {
    e.preventDefault();
    showHome();
  });

  links.realTimeX?.addEventListener("click", (e) => {
    e.preventDefault();
    const section = sections.liveFeed;
    const now = Date.now();
    const elapsed = now - lastFetchTime;

    const wasHidden = section?.classList.contains("d-none");
    section?.classList.toggle("d-none");

    // reload μόνο αν μόλις άνοιξε
    if (wasHidden) {
      if (elapsed > 60000 || liveTweets.length === 0) {
        loadLiveTweets(true);  // RESET μόνο όταν χρειάζεται
        lastFetchTime = now;
      }
    }
  });


  links.sentimentAnalysis?.addEventListener("click", (e) => {
    e.preventDefault();
    showOnly("sentimentAnalysis");
  });

  links.trendingTopics?.addEventListener("click", (e) => {
    e.preventDefault();
    showOnly("trendingTopics");
  });

  links.distilbert?.addEventListener("click", (e) => {
    e.preventDefault();
    showOnly("distilbert");
  });

  links.trainingStats?.addEventListener("click", (e) => {
    e.preventDefault();
    showOnly("trainingStats");
  });

  showHome();
});


function typeTextPlain(element, text, delay = 40, callback = null) {
  element.innerHTML = "";
  let i = 0;

  function typeChar() {
    if (i < text.length) {
      const char = text.charAt(i);
      if (char === "\n") {
        element.appendChild(document.createElement("br"));
      } else {
        element.appendChild(document.createTextNode(char));
      }
      i++;
      setTimeout(typeChar, delay);
    } else if (callback) {
      callback(); 
    }
  }

  typeChar();
}



// === Analyze Sentiment ===
async function analyzeSentiment() {
  const tweet = document.getElementById("tweetInput").value.trim();
  if (!tweet) {
    alert("Please enter a tweet!");
    return;
  }

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tweet })
    });

    const data = await response.json();
    const botResponse = document.getElementById("bot-response");
    const chatBox = document.getElementById("chat-response");

    // Καθαρισμός προηγούμενου περιεχομένου
    botResponse.innerHTML = "";
    botResponse.style.textAlign = "left"; // Στοίχιση αριστερά

    const fullResponse = `Sentiment: ${data.sentiment}\nReason: ${data.reason}`;

    // Εμφάνιση Sentiment + Reason
    typeTextPlain(botResponse, fullResponse, 40, () => {

      // === TOXICITY SCORE ===
      if (data.toxicity !== undefined) {
        const toxPercent = (data.toxicity * 100).toFixed(1);
        const toxicityBox = document.createElement("p");
        toxicityBox.className = "mt-2";
        toxicityBox.style.marginTop = "10px";
        toxicityBox.style.color = data.toxicity > 0.5 ? "#d32f2f" : "#888";
        toxicityBox.innerHTML = `Toxicity Score: <strong>${toxPercent}%</strong>`
        botResponse.appendChild(toxicityBox);
      }

      // === WARNINGS ===
      if (data.warnings && Array.isArray(data.warnings)) {
        data.warnings.forEach(warning => {
          const warnBox = document.createElement("p");
          warnBox.className = "mt-2";
          warnBox.style.marginTop = "8px";
          warnBox.style.color = "#ff9800";
          warnBox.innerHTML = `<em>${warning}</em>`;
          botResponse.appendChild(warnBox);
        });
      }
    });

    botResponse.style.color =
      data.sentiment === "Positive"
        ? "#4caf50"
        : data.sentiment === "Negative"
        ? "#f44336"
        : "#ff9800";

    chatBox.classList.remove("d-none");

    // === EMOTION CHART ===
    const emotionContainer = document.getElementById("emotionChartContainer");
    const emotionCanvas = document.getElementById("emotionChart");

    const labels = data.emotion_scores.map(e => e.label);
    const scores = data.emotion_scores.map(e => (e.score * 100).toFixed(1));

    const colors = [
      '#f7b731', '#fc5c65', '#fd9644',
      '#26de81', '#2bcbba', '#a55eea', '#778ca3'
    ];

    if (window.emotionChartInstance) {
      window.emotionChartInstance.destroy();
    }

    const ctx = emotionCanvas.getContext("2d");
    window.emotionChartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Emotion Confidence (%)',
          data: scores,
          backgroundColor: colors,
          borderColor: colors,
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            ticks: {
              callback: function (value) {
                return value + '%';
              }
            }
          }
        }
      }
    });

    emotionContainer.classList.remove("d-none");

  } catch (err) {
    console.error("Error during sentiment analysis:", err);
    alert("Something went wrong. Please try again.");
  }
}


// --- Load Tweets ---
async function loadLiveTweets(forceReload = false) {
  const container = document.getElementById("tweetsContainer");
  const loadMoreBtn = document.getElementById("loadMoreBtn");
  if (isLoading) return;
  isLoading = true;

  if (forceReload) {
    currentBatch = 1;        // ένα counter μόνο για UI/logs
    liveTweets = [];
    displayedTweetIds.clear();
    container.innerHTML = "";
  }

  try {
    const res = await fetch(`/live-feed?_=${Date.now()}`);
    const data = await res.json();

    if (data.tweets && data.tweets.length > 0) {
      const newTweets = data.tweets.filter(t => !displayedTweetIds.has(t.id));
      newTweets.forEach(t => displayedTweetIds.add(t.id));

      if (newTweets.length > 0) {
        const startIndex = liveTweets.length;
        liveTweets = liveTweets.concat(newTweets);
        // roll-in animation για αυτό το batch
        renderTweets(container, newTweets, startIndex, true);
        currentBatch++; // μόνο για debug/μετρητή
      }
    }

    if (!data.hasMore) {
      loadMoreBtn?.classList.add("d-none");
    } else {
      loadMoreBtn?.classList.remove("d-none");
    }
  } catch (e) {
    container.innerHTML += "<p>Error loading tweets.</p>";
    console.error(e);
  }

  isLoading = false;
}


// --- Load More ---
document.addEventListener("DOMContentLoaded", () => {
  const loadMoreBtn = document.getElementById("loadMoreBtn");
  loadMoreBtn?.addEventListener("click", () => loadLiveTweets());
});



// --- Render Tweets ---
function renderTweets(container, tweets, startIndex = 0, animate = false) {
  tweets.forEach((tweet, index) => {
    const tweetIndex = startIndex + index;

    const card = document.createElement("div");
    card.id = `tweet-${tweet.id}`;
    card.className = "tweet-card";
    if (animate) {
      card.classList.add("roll-in");
      card.style.animationDelay = `${index * 40}ms`; // μικρό stagger
    }

    const text = document.createElement("p");
    text.className = "tweet-text";
    text.innerHTML = `<strong>@${tweet.username}</strong>: ${tweet.text}`;
    card.appendChild(text);

    const meta = document.createElement("div");
    meta.className = "tweet-meta";

    const analyzeBtn = document.createElement("button");
    analyzeBtn.className = "btn-mini-analyze ms-2";
    analyzeBtn.textContent = "Analyze";
    analyzeBtn.onclick = () => analyzeTweetSentiment(tweet.text, tweetIndex);
    meta.appendChild(analyzeBtn);
    card.appendChild(meta);

    const responseBox = document.createElement("div");
    responseBox.id = `response-${tweetIndex}`;
    responseBox.className = "chat-box mt-2 d-none small-chat-box";
    responseBox.innerHTML = `
      <div class="bot-msg small-bot-msg">
        <span class="small-label">SentimentX:</span>
        <p id="response-text-${tweetIndex}" class="response-text"></p>
        <div id="response-extra-${tweetIndex}" class="mt-2"></div>
        <div class="chart-wrapper">
          <canvas id="emotionChart-${tweetIndex}"></canvas>
        </div>
      </div>
    `;
    card.appendChild(responseBox);

    container.appendChild(card);
  });
}



// --- Analyze Sentiment on Live Feed ---
async function analyzeTweetSentiment(tweet, index) {
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tweet })
    });

    const data = await response.json();
    const box = document.getElementById(`response-${index}`);
    const text = document.getElementById(`response-text-${index}`);
    const extraDiv = document.getElementById(`response-extra-${index}`);
    const canvas = document.getElementById(`emotionChart-${index}`);

    text.innerHTML = "";
    const fullResponse = `Sentiment: ${data.sentiment}\nReason: ${data.reason}`;

    typeTextPlain(text, fullResponse, 40, () => {
      // Toxicity
      if (data.toxicity !== undefined) {
        const toxPercent = (data.toxicity * 100).toFixed(1);
        const toxicityBox = document.createElement("p");
        toxicityBox.className = "mt-2";
        toxicityBox.style.marginTop = "10px";
        toxicityBox.style.color = data.toxicity > 0.5 ? "#d32f2f" : "#888";
        toxicityBox.innerHTML = `Toxicity Score: <strong>${toxPercent}%</strong>`;
        extraDiv.appendChild(toxicityBox);
      }

      // Warnings
      if (data.warnings && Array.isArray(data.warnings)) {
        data.warnings.forEach(warning => {
          const warnBox = document.createElement("p");
          warnBox.className = "mt-2";
          warnBox.style.marginTop = "8px";
          warnBox.style.color = "#ff9800";
          warnBox.innerHTML = `<em>${warning}</em>`;
          extraDiv.appendChild(warnBox);
        });
      }
    });

    text.style.color = data.sentiment === "Positive"
      ? "#4caf50"
      : data.sentiment === "Negative"
        ? "#f44336"
        : "#ff9800";

    box.classList.remove("d-none");

    // Emotion Chart
    if (data.emotion_scores && Array.isArray(data.emotion_scores)) {
      const labels = data.emotion_scores.map(e => e.label);
      const scores = data.emotion_scores.map(e => (e.score * 100).toFixed(1));
      const colors = [
        '#f7b731', '#fc5c65', '#fd9644',
        '#26de81', '#2bcbba', '#a55eea', '#778ca3'
      ];

      const ctx = canvas.getContext("2d");

      if (canvas.chartInstance) {
        canvas.chartInstance.destroy();
      }

      canvas.chartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{
            label: 'Emotion Confidence (%)',
            data: scores,
            backgroundColor: colors,
            borderColor: colors,
            borderWidth: 1
          }]
        },
        options: {
          responsive: true,
          scales: {
            y: {
              beginAtZero: true,
              max: 100,
              ticks: {
                callback: function (value) {
                  return value + '%';
                }
              }
            }
          }
        }
      });
    }

  } catch (err) {
    console.error("Error analyzing tweet:", err);
    alert("Something went wrong analyzing the tweet.");
  }
}
