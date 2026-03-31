// Goal board used for reset and initial page load.
const GOAL = [1, 2, 3, 4, 5, 6, 7, 8, 0];

// Mutable board state currently shown in the editor.
let boardState = [...GOAL];

// Predefined demo cases for quick testing and presentations.
const SAMPLE_CASES = {
  easy: [1, 2, 3, 4, 0, 6, 7, 5, 8],
  medium: [1, 2, 3, 5, 0, 6, 4, 7, 8],
  hard: [8, 6, 7, 2, 5, 4, 3, 0, 1]
};

// Most recent algorithm results returned by backend.
let lastResults = [];

// Maps algorithm label -> { states, moves } for path playback.
let pathStateMap = {};

// Currently selected algorithm in the path viewer dropdown.
let currentPathAlgorithm = "";

// Main board and path viewer board containers.
const editorBoard = document.getElementById("editor-board");
const pathBoard = document.getElementById("path-board");

// Status and form elements used across interactions.
const statusEl = document.getElementById("status");
const stateInput = document.getElementById("state-input");
const resultsSection = document.getElementById("results");
const summaryCards = document.getElementById("summary-cards");
const pathViewer = document.getElementById("path-viewer");
const pathAlgorithmSelect = document.getElementById("path-algorithm");
const pathStepInput = document.getElementById("path-step");
const pathMeta = document.getElementById("path-meta");

const algorithmSelect = document.getElementById("algorithm-select");
const dfsDepthInput = document.getElementById("dfs-depth");
const dfsExpansionsInput = document.getElementById("dfs-expansions");

// Primary action buttons.
const solveBtn = document.getElementById("solve-btn");
const mlPredictBtn = document.getElementById("ml-predict-btn");
const randomBtn = document.getElementById("random-btn");
const goalBtn = document.getElementById("goal-btn");
const applyStateBtn = document.getElementById("apply-state-btn");

// Preset sample buttons (easy / medium / hard).
const sampleButtons = document.querySelectorAll("[data-sample]");

// UI elements that display ML model output.
const mlPanel = document.getElementById("ml-panel");
const mlPredictionEl = document.getElementById("ml-prediction");
const mlConfidenceEl = document.getElementById("ml-confidence");
const mlExpertEl = document.getElementById("ml-expert");

function stateToString(state) {
  // Convert [1,2,3,...] into a human-editable input string.
  return state.join(" ");
}

function parseStateText(text) {
  // Accept commas or spaces and split into non-empty tokens.
  const tokens = text
    .replace(/,/g, " ")
    .split(/\s+/)
    .map((t) => t.trim())
    .filter(Boolean);

  // 8-puzzle requires exactly 9 entries.
  if (tokens.length !== 9) {
    throw new Error("Enter exactly 9 numbers.");
  }

  // Convert to numbers and validate integer-only input.
  const values = tokens.map((token) => Number(token));
  if (values.some((value) => Number.isNaN(value) || !Number.isInteger(value))) {
    throw new Error("All values must be integers.");
  }

  // Validate that digits 0..8 appear exactly once.
  const expected = new Set([0, 1, 2, 3, 4, 5, 6, 7, 8]);
  const got = new Set(values);
  if (got.size !== 9 || [...expected].some((n) => !got.has(n))) {
    throw new Error("Use each digit 0-8 exactly once.");
  }

  // Return validated board array.
  return values;
}

function isAdjacent(a, b) {
  // Convert flat indices to row/col coordinates.
  const r1 = Math.floor(a / 3);
  const c1 = a % 3;
  const r2 = Math.floor(b / 3);
  const c2 = b % 3;

  // Manhattan distance of 1 means direct adjacency.
  return Math.abs(r1 - r2) + Math.abs(c1 - c2) === 1;
}

function renderBoard(container, state, onTileClick = null, options = {}) {
  // Optional animation for smoother visual transitions.
  const { animate = false } = options;

  // Rebuild board from scratch each render for simplicity.
  container.innerHTML = "";

  state.forEach((value, index) => {
    // Each tile is a button so it can be interactive in editor mode.
    const tile = document.createElement("button");
    tile.type = "button";

    // Blank tile gets an alternate class and no text.
    tile.className = `tile ${value === 0 ? "blank" : ""}`;
    tile.textContent = value === 0 ? "" : String(value);

    // Add transition class when requested.
    if (animate) {
      tile.classList.add("tile-animate");
    }

    // In editor mode, tiles are clickable and send their index.
    if (onTileClick) {
      tile.addEventListener("click", () => onTileClick(index));
    } else {
      // In read-only mode (path viewer), disable tile buttons.
      tile.disabled = true;
    }

    // Append tile to board grid.
    container.appendChild(tile);
  });
}

function renderEditor(animate = false) {
  // Render editor board with click behavior for legal tile moves.
  renderBoard(editorBoard, boardState, (clickedIndex) => {
    // Find blank tile and ensure clicked tile is adjacent.
    const blankIndex = boardState.indexOf(0);
    if (!isAdjacent(clickedIndex, blankIndex)) {
      // Ignore illegal clicks to keep board valid.
      return;
    }

    // Swap clicked tile with blank to perform move.
    [boardState[clickedIndex], boardState[blankIndex]] = [boardState[blankIndex], boardState[clickedIndex]];

    // Keep text input synchronized with board state.
    stateInput.value = stateToString(boardState);

    // Re-render editor with animation after each move.
    renderEditor(true);
  }, { animate });
}

function setStatus(message, isError = false) {
  // Status line gives user feedback for actions and errors.
  statusEl.textContent = message;

  // Color status based on success vs error context.
  statusEl.style.color = isError ? "#ff9d85" : "#9ec5d7";
}

function resetMlPanel() {
  // Hide stale ML output when board state changes.
  mlPanel.classList.add("hidden");
  mlPredictionEl.textContent = "";
  mlConfidenceEl.textContent = "";
  mlExpertEl.textContent = "";
}

function formatNumber(value, decimals = 0) {
  // Return dash when value is missing or invalid.
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "-";
  }

  // Use fixed decimal output for clean metric display.
  return value.toFixed(decimals);
}

function renderSummaryCards(results) {
  // Clear previous cards before rendering new run.
  summaryCards.innerHTML = "";

  results.forEach((result) => {
    // Build one result card per algorithm.
    const card = document.createElement("article");
    card.className = "summary-card";
    card.innerHTML = `
      <h4>${result.algorithm}</h4>
      <p>Solved: ${result.found ? "Yes" : "No"}</p>
      <p>Depth: ${result.solutionDepth ?? "-"}</p>
      <p>Expanded: ${result.expandedNodes}</p>
      <p>Frontier: ${result.maxFrontierSize}</p>
      <p>Time: ${formatNumber(result.elapsedTime, 6)} s</p>
    `;

    // Add card to results area.
    summaryCards.appendChild(card);
  });
}

function renderBars(containerId, results, valueFn, valueFormatter) {
  // Find chart container and clear previous bars.
  const container = document.getElementById(containerId);
  container.innerHTML = "";

  // Compute values and normalize widths by max value.
  const values = results.map(valueFn);
  const maxValue = Math.max(...values, 1);

  results.forEach((result, idx) => {
    const rawValue = values[idx];

    // Convert metric into percent width for horizontal bar fill.
    const width = (rawValue / maxValue) * 100;

    // Construct one bar row with label, bar, and value text.
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <span class="bar-label">${result.algorithm}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${width.toFixed(2)}%"></span></span>
      <span class="bar-value">${valueFormatter(rawValue, result)}</span>
    `;

    // Append row into selected chart container.
    container.appendChild(row);
  });
}

function updatePathViewer() {
  // Lookup path data for currently selected algorithm.
  const pathData = pathStateMap[currentPathAlgorithm];
  if (!pathData) {
    // Hide viewer when no path is available.
    pathViewer.classList.add("hidden");
    return;
  }

  // Ensure viewer is visible when valid path exists.
  pathViewer.classList.remove("hidden");

  // Configure slider bounds from path length.
  const maxIndex = Math.max(0, pathData.states.length - 1);
  pathStepInput.max = String(maxIndex);

  // Clamp selected step to safe bounds.
  let step = Number(pathStepInput.value || "0");
  if (step > maxIndex) {
    step = maxIndex;
    pathStepInput.value = String(step);
  }

  // Render board snapshot for current step.
  const stepState = pathData.states[step];
  renderBoard(pathBoard, stepState, null, { animate: true });

  // Show move metadata. Step 0 is the start state.
  const moveLabel = step === 0 ? "(start)" : pathData.moves[step - 1] || "?";
  pathMeta.textContent = `Step ${step} / ${maxIndex} | Move: ${moveLabel}`;
}

function renderPathOptions(results) {
  // Reset old path data and dropdown options.
  pathStateMap = {};
  pathAlgorithmSelect.innerHTML = "";

  // Keep only solved results that include state sequences.
  results
    .filter((result) => result.found && Array.isArray(result.states) && result.states.length > 0)
    .forEach((result, idx) => {
      // Store path data keyed by algorithm label.
      pathStateMap[result.algorithm] = {
        states: result.states,
        moves: result.moves || []
      };

      // Add dropdown option for this algorithm path.
      const option = document.createElement("option");
      option.value = result.algorithm;
      option.textContent = result.algorithm;
      pathAlgorithmSelect.appendChild(option);

      // Set first available algorithm as default selection.
      if (idx === 0) {
        currentPathAlgorithm = result.algorithm;
      }
    });

  // Hide viewer entirely if no solved path is present.
  if (Object.keys(pathStateMap).length === 0) {
    pathViewer.classList.add("hidden");
    return;
  }

  // Initialize dropdown and slider to first step of chosen path.
  pathAlgorithmSelect.value = currentPathAlgorithm;
  pathStepInput.value = "0";
  updatePathViewer();
}

async function solveCurrentState() {
  // Show immediate status and prevent repeated clicks.
  setStatus("Solving...");
  solveBtn.disabled = true;

  try {
    // Build API payload from current UI state and settings.
    const body = {
      start: boardState,
      algorithm: algorithmSelect.value,
      dfsDepthLimit: Number(dfsDepthInput.value),
      dfsMaxExpansions: Number(dfsExpansionsInput.value),
      includePath: true
    };

    // Ask backend to solve puzzle with selected algorithm(s).
    const response = await fetch("/api/solve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });

    // Parse JSON and surface backend errors cleanly.
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || "Failed to solve puzzle.");
    }

    // Save results and reveal results section.
    lastResults = payload.results || [];
    resultsSection.classList.remove("hidden");

    // Render cards, metric bars, and path controls.
    renderSummaryCards(lastResults);
    renderBars("expanded-bars", lastResults, (r) => r.expandedNodes, (v) => String(v));
    renderBars("depth-bars", lastResults, (r) => (r.solutionDepth ?? 0), (v, r) => (r.solutionDepth == null ? "N/A" : String(v)));
    renderBars("time-bars", lastResults, (r) => r.elapsedTime, (v) => Number(v).toFixed(6));
    renderPathOptions(lastResults);

    // Notify user that solve operation completed.
    setStatus("Solved. Results updated.");
  } catch (error) {
    // Display API/network/validation errors in status line.
    setStatus(error.message || "Error while solving.", true);
  } finally {
    // Re-enable solve button in both success and error paths.
    solveBtn.disabled = false;
  }
}

async function randomizeState() {
  // Avoid duplicate random requests while in flight.
  randomBtn.disabled = true;
  setStatus("Generating random solvable state...");

  try {
    // Request a solvable random board generated by backend.
    const response = await fetch("/api/random?steps=40");
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || "Failed to generate state.");
    }

    // Apply random board and refresh editor view.
    boardState = payload.state;
    stateInput.value = stateToString(boardState);
    renderEditor(true);
  resetMlPanel();
    setStatus("Random solvable state loaded.");
  } catch (error) {
    // Show random-generation failure details.
    setStatus(error.message || "Error while randomizing state.", true);
  } finally {
    // Always re-enable random button.
    randomBtn.disabled = false;
  }
}

async function predictWithMl() {
  // Disable button while ML request is in flight.
  mlPredictBtn.disabled = true;
  setStatus("Getting ML prediction...");

  try {
    // Ask backend model for the best next move on current board.
    const response = await fetch("/api/ml/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ start: boardState })
    });

    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || "Failed to get ML prediction.");
    }

    // Reveal ML panel and render prediction details.
    mlPanel.classList.remove("hidden");
    mlPredictionEl.textContent = `Predicted next move: ${payload.predictedMove}`;
    mlConfidenceEl.textContent = `Confidence: ${formatNumber(payload.confidence * 100, 2)}% | Training accuracy: ${formatNumber((payload.trainingMetrics?.accuracy || 0) * 100, 2)}%`;
    mlExpertEl.textContent = payload.expertMove
      ? `A* expert move: ${payload.expertMove} | Match: ${payload.matchesExpert ? "Yes" : "No"}`
      : "A* expert move unavailable for this state.";

    setStatus("ML prediction ready.");
  } catch (error) {
    resetMlPanel();
    setStatus(error.message || "Error while getting ML prediction.", true);
  } finally {
    // Always restore button state.
    mlPredictBtn.disabled = false;
  }
}

// Solve action button.
solveBtn.addEventListener("click", solveCurrentState);

// ML prediction button.
mlPredictBtn.addEventListener("click", predictWithMl);

// Random board button.
randomBtn.addEventListener("click", randomizeState);

// Reset editor board back to solved goal state.
goalBtn.addEventListener("click", () => {
  boardState = [...GOAL];
  stateInput.value = stateToString(boardState);
  renderEditor(true);
  resetMlPanel();
  setStatus("Board reset to goal state.");
});

// Apply manually typed board input to editor state.
applyStateBtn.addEventListener("click", () => {
  try {
    // Parse and validate user-entered board text.
    boardState = parseStateText(stateInput.value);
    renderEditor(true);
    resetMlPanel();
    setStatus("State applied.");
  } catch (error) {
    // Show parse/validation error.
    setStatus(error.message || "Invalid state.", true);
  }
});

// Load one of the predefined sample boards.
sampleButtons.forEach((button) => {
  button.addEventListener("click", () => {
    // Read sample key from button data attribute.
    const key = button.dataset.sample;
    const sample = SAMPLE_CASES[key];

    // Ignore unknown sample keys defensively.
    if (!sample) {
      return;
    }

    // Apply selected sample and refresh editor.
    boardState = [...sample];
    stateInput.value = stateToString(boardState);
    renderEditor(true);
    resetMlPanel();
    setStatus(`Loaded ${key} sample.`);
  });
});

// Switch path viewer to another algorithm's solution path.
pathAlgorithmSelect.addEventListener("change", () => {
  currentPathAlgorithm = pathAlgorithmSelect.value;

  // Reset slider to first step whenever algorithm changes.
  pathStepInput.value = "0";
  updatePathViewer();
});

// Move through solution steps with slider.
pathStepInput.addEventListener("input", updatePathViewer);

// Initialize UI on first page load.
stateInput.value = stateToString(boardState);
renderEditor();
resetMlPanel();
setStatus("Ready. Build a board and click Solve Puzzle.");
