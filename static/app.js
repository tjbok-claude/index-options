/**
 * SPX Hedge Ranker — frontend
 *
 * Data flow:
 *   Param change  → debounce 400ms → /api/options (server re-scores cached CBOE data)
 *   Filter change → debounce 120ms → client-side array filter → table.replaceData()
 *   Column click  → Tabulator native sort, no server call
 */
'use strict';
console.log('app.js v20260322b');

// ---------------------------------------------------------------------------
// Global error handler — catches uncaught JS errors and shows them visibly
// ---------------------------------------------------------------------------
window.addEventListener('error', (evt) => {
  showError(`JS error: ${evt.message} (${evt.filename}:${evt.lineno})`);
});

const API_URL         = '/api/options';
const DEBOUNCE_PARAM  = 400;
const DEBOUNCE_FILTER = 120;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  allRows:      [],
  filteredRows: [],
  meta:         null,
  loading:      false,
};

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
function numVal(elemId, fallback) {
  const v = parseFloat($(elemId)?.value);
  return isNaN(v) ? fallback : v;
}

// Default crash-beta by symbol (crash-weighted, not market beta)
const SYMBOL_BETA_DEFAULTS = { SPX:1.0, NDX:1.3, RUT:0.85 };

function getParams() {
  return {
    p_crash:         numVal('pCrash',   0.55),
    horizon:         numVal('horizon',  18),
    roth_multiplier: numVal('rothMult', 1.25),
    contracts:       numVal('contracts',0),
    model:           $('modelSelect')?.value  || 'garch_ep',
    symbol:          $('symbolSelect')?.value || 'SPX',
    index_beta:      numVal('indexBeta', 1.0),
  };
}

function getFilters() {
  return {
    dte_min:        numVal('dteMin',        0),
    dte_max:        numVal('dteMax',        9999),
    oi_min:         numVal('oiMin',         0),
    vol_min:        numVal('volMin',        0),
    moneyness_min:  numVal('moneynessMin', -100),
    moneyness_max:  numVal('moneynessMax',  0),
    session:        $('sessionFilter')?.value || 'ALL',
  };
}

// ---------------------------------------------------------------------------
// Formatters
// ---------------------------------------------------------------------------
function na() { return '<span class="null-val">—</span>'; }

function fmtDollar0(cell) {
  const v = cell.getValue();
  if (v == null) return na();
  return '$' + Math.round(v).toLocaleString();
}

function fmtDollar2(cell) {
  const v = cell.getValue();
  return v == null ? na() : v.toFixed(2);
}

function fmtPct1(cell) {
  const v = cell.getValue();
  return v == null ? na() : v.toFixed(1) + '%';
}

function fmtIV(cell) {
  const v = cell.getValue();
  return v == null ? na() : (v * 100).toFixed(1) + '%';
}

function fmtNum2(cell) {
  const v = cell.getValue();
  return v == null ? na() : v.toFixed(2);
}

function fmtNum3(cell) {
  const v = cell.getValue();
  return v == null ? na() : v.toFixed(3);
}

function fmtMoneyness(cell) {
  const v = cell.getValue();
  return v == null ? na() : v.toFixed(1) + '%';
}

function fmtSpread(cell) {
  const v = cell.getValue();
  if (v == null) return na();
  const el = cell.getElement();
  el.classList.toggle('cell-warn', v > 20);
  return v.toFixed(1) + '%';
}

function fmtEPR(cell) {
  const v = cell.getValue();
  if (v == null) return na();
  const el = cell.getElement();
  el.classList.remove('cell-epr-high', 'cell-epr-mid', 'cell-epr-low');
  if      (v >= 2.0) el.classList.add('cell-epr-high');
  else if (v >= 1.0) el.classList.add('cell-epr-mid');
  else               el.classList.add('cell-epr-low');
  return v.toFixed(3);
}

function fmtCrashEff(cell) {
  const v = cell.getValue();
  if (v == null) return na();
  const el = cell.getElement();
  el.classList.remove('cell-epr-high', 'cell-epr-mid', 'cell-epr-low');
  if      (v >= 2.0) el.classList.add('cell-epr-high');
  else if (v >= 1.0) el.classList.add('cell-epr-mid');
  else               el.classList.add('cell-epr-low');
  return v.toFixed(3);
}

function fmtTheo(cell) {
  const v = cell.getValue();
  if (v == null) return na();
  if (v > 10)  return `<span style="color:var(--danger)">${v.toFixed(1)}%</span>`;
  if (v < -10) return `<span style="color:var(--good);font-weight:600">${v.toFixed(1)}%</span>`;
  return v.toFixed(1) + '%';
}

function rowFormatter(row) {}

// ---------------------------------------------------------------------------
// Column definitions
// ---------------------------------------------------------------------------
// width  = relative weight used by fitColumns to distribute available space
// minWidth = hard floor — columns won't go below this before horizontal scroll kicks in
const COLUMNS = [
  {
    title: 'Expiry', field: 'expiry', sorter: 'string', width: 105, minWidth: 90, frozen: true,
    tooltip: 'Option expiration date (YYYY-MM-DD).',
  },
  {
    title: 'Ses', field: 'session', sorter: 'string', width: 52, minWidth: 52, frozen: true,
    tooltip: 'AM = standard SPX (cash-settled at open on expiry day). PM = SPXW weekly (settled at close). AM has lower gamma risk for multi-month holds.',
  },
  {
    title: 'DTE', field: 'dte', sorter: 'number', width: 60, minWidth: 60,
    tooltip: 'Days to expiration. Filter shows only 180–540 DTE by default (6–18 months for set-and-forget).',
  },
  {
    title: 'Strike', field: 'strike', sorter: 'number', width: 78, minWidth: 72,
    tooltip: 'Option strike price in SPX index points.',
  },
  {
    title: 'OTM%', field: 'moneyness_pct', sorter: 'number', width: 76, minWidth: 70, formatter: fmtMoneyness,
    tooltip: 'How far out-of-the-money: (strike / SPX_spot − 1) × 100. −20% means the strike is 20% below current SPX. The 20%-portfolio-drawdown protection zone is roughly −19% to −22%.',
  },
  {
    title: 'Mid', field: 'mid', sorter: 'number', width: 72, minWidth: 66, formatter: fmtDollar2,
    tooltip: 'Mid-market price = (bid + ask) / 2, in SPX index points. Multiply by 100 for dollar cost per contract.',
  },
  {
    title: 'Sprd%', field: 'spread_pct', sorter: 'number', width: 72, minWidth: 66, formatter: fmtSpread,
    tooltip: 'Bid-ask spread as % of mid. Above 20% (shown amber) means mid is unreliable — expect to pay more than mid to fill.',
  },
  {
    title: 'IV', field: 'iv', sorter: 'number', width: 62, minWidth: 56, formatter: fmtIV,
    tooltip: 'Implied volatility from CBOE model. Higher IV = more expensive option. Compare across strikes/expiries to find relatively cheap contracts.',
  },
  {
    title: 'Delta', field: 'delta', sorter: 'number', width: 72, minWidth: 66, formatter: fmtNum2,
    tooltip: 'Option delta: $ change in option price per $1 move in SPX. Negative for puts. −0.15 delta ≈ 15% probability of finishing in-the-money (rough approximation).',
  },
  {
    title: 'Theta/d', field: 'theta_daily', sorter: 'number', width: 84, minWidth: 84, formatter: fmtNum2,
    tooltip: 'Daily time decay: dollars lost per contract per calendar day from time passing alone. Negative. LEAPS have slow theta — a key advantage for set-and-forget.',
  },
  {
    title: 'OI', field: 'open_interest', sorter: 'number', width: 72, minWidth: 66,
    tooltip: 'Open interest: total open contracts. Higher OI = better liquidity, tighter markets. Filter minimum is 100 by default.',
  },
  {
    title: 'Vol', field: 'volume', sorter: 'number', width: 66, minWidth: 60,
    tooltip: "Today's trading volume. Low volume doesn't disqualify a contract (LEAPS are thinly traded) but confirms whether mid price reflects real activity.",
  },
  {
    title: 'Cost 1c', field: 'cost_1c', sorter: 'number', width: 88, minWidth: 82, formatter: fmtDollar0,
    tooltip: 'Total premium cost for one contract = mid × 100. Each SPX contract covers 100 index units.',
  },
  {
    title: 'Cost Nc', field: 'cost_Nc', sorter: 'number', width: 88, minWidth: 82, formatter: fmtDollar0,
    tooltip: 'Total cost for N contracts (your full position). N is auto-computed from portfolio beta unless overridden.',
  },
  {
    title: 'Crash Payoffs (per contract)',
    columns: [
      {
        title: 'SPX −25%', field: 'payoff_crash_25pct_1c', sorter: 'number', width: 96, minWidth: 96, formatter: fmtDollar0,
        tooltip: 'Gross payoff if SPX drops 25% from today. Your $5M portfolio would drop ~22% (beta-adjusted). This is the "mild crash" scenario.',
      },
      {
        title: 'SPX −40%', field: 'payoff_crash_40pct_1c', sorter: 'number', width: 96, minWidth: 96, formatter: fmtDollar0,
        tooltip: 'Gross payoff if SPX drops 40%. Portfolio ~−35%. "Deep crash" scenario (2008-level drawdown).',
      },
      {
        title: 'SPX −55%', field: 'payoff_crash_55pct_1c', sorter: 'number', width: 96, minWidth: 96, formatter: fmtDollar0,
        tooltip: 'Gross payoff if SPX drops 55%. Portfolio ~−49%. "Severe crash" scenario (Great Depression analog).',
      },
    ],
  },
  {
    title: 'E[Pay]', field: 'e_payoff_roth_1c', sorter: 'number', width: 82, minWidth: 76, formatter: fmtDollar0,
    tooltip: 'Expected payoff = Σ(probability_i × payoff_i) × Roth multiplier. Weighted across all 6 scenarios (bull, flat, bear, crash mild/deep/severe). The 1.25× Roth multiplier reflects that gains in a Roth IRA are tax-free (vs taxable account).',
  },
  {
    title: 'E[Net]', field: 'e_net_1c', sorter: 'number', width: 82, minWidth: 76, formatter: fmtDollar0,
    tooltip: 'E[Pay] minus Cost 1c. Positive = positive expected value at your assumed crash probability. Most market puts have negative E[Net] at "market-implied" crash probabilities — positive means you believe crashes are underpriced.',
  },
  {
    title: 'EPR', field: 'EPR', sorter: 'number', width: 66, minWidth: 60, formatter: fmtEPR,
    tooltip: 'Expected Payoff Ratio = E[Pay] / Cost. PRIMARY SORT KEY.\n\n>2.0 (green): excellent value at your crash assumptions\n1.0–2.0 (yellow): break-even to modestly positive\n<1.0: expensive but may still be worth buying for tail protection\n\nEPR > 1.0 means the put is "cheap" relative to your subjective crash probability. Lower P(crash) → lower EPR everywhere.',
  },
  {
    title: 'CrEff', field: 'crash_efficiency', sorter: 'number', width: 72, minWidth: 66, formatter: fmtCrashEff,
    tooltip: 'Crash Efficiency = crash-scenario expected payoff (Roth-adjusted) / cost. Same as EPR but counting only the three crash scenarios, ignoring bull/flat/bear. Shows how efficiently this put buys tail protection specifically.',
  },
  {
    title: 'Ann%', field: 'annual_cost_pct', sorter: 'number', width: 68, minWidth: 62, formatter: fmtPct1,
    tooltip: 'Annualized cost of the full position (N contracts) as % of $5M portfolio = (Cost Nc / portfolio) × (365 / DTE) × 100. Good insurance is typically 0.5–1.5% per year. Above 2% is expensive.',
  },
  {
    title: 'vs Theo%', field: 'theo_vs_mid_pct', sorter: 'number', width: 92, minWidth: 88, formatter: fmtTheo,
    tooltip: "How much you're paying vs CBOE's own theoretical model price. (mid − theo) / theo × 100. Red (>+10%): overpaying. Green (<−10%): cheaper than CBOE model, rare but worth noting. Near 0 is normal for liquid strikes.",
  },
];

// ---------------------------------------------------------------------------
// Tabulator init
// ---------------------------------------------------------------------------
let table = null;

function initTable() {
  // Set an initial explicit height immediately — virtual rendering requires this.
  // setGridHeight() will refine it once the DOM is fully painted.
  const initialH = Math.max(300, window.innerHeight - 300);

  table = new Tabulator('#grid', {
    data:                 [],
    columns:              COLUMNS,
    layout:               'fitColumns',
    renderVertical:       'virtual',
    height:               initialH + 'px',
    rowFormatter:         rowFormatter,
    initialSort:          [{ column: 'EPR', dir: 'desc' }],
    placeholder:          "Click 'Fetch Live Data' to load the SPX options chain.",
    columnHeaderSortMulti: false,
    movableColumns:        true,
  });

  table.on('dataSorted', (sorters) => {
    if (!sorters.length) return;
    const s = sorters[0];
    const indicator = $('sortIndicator');
    if (!indicator) return;
    const arrow = s.dir === 'desc' ? '↓' : '↑';
    indicator.textContent = s.field === 'EPR' ? '' : `sorted by ${s.field} ${arrow}`;
  });
}

// Measure where the grid div starts vertically, fill to just above the legend.
function setGridHeight() {
  requestAnimationFrame(() => {
    if (!table) return;
    const gridEl = $('grid');
    if (!gridEl) return;
    const top     = gridEl.getBoundingClientRect().top;
    const legendH = (document.querySelector('.legend') || {}).offsetHeight || 26;
    const h       = Math.max(150, window.innerHeight - top - legendH - 2);
    table.setHeight(h + 'px');
  });
}

// ---------------------------------------------------------------------------
// API fetch
// ---------------------------------------------------------------------------
// Sequence counter: each fetch gets an ID; stale responses (superseded by a
// newer fetch) are silently discarded so symbol switches never clobber results.
let _fetchSeq = 0;

async function fetchOptions(forceRefresh = false) {
  if (state.loading) return;
  setLoading(true, 'Connecting to server…');

  const mySeq = ++_fetchSeq;   // tag this fetch; anything with a lower seq is stale

  // Show overlay while a forced refresh loads (gives clear visual feedback)
  if (forceRefresh) showGridOverlay(true);

  const p  = getParams();
  const qs = new URLSearchParams({
    p_crash:         p.p_crash,
    horizon:         p.horizon,
    roth_multiplier: p.roth_multiplier,
    contracts:       p.contracts,
    model:           p.model,
    symbol:          p.symbol,
    index_beta:      p.index_beta,
    force_refresh:   forceRefresh,
  });

  try {
    setStatus(`Fetching ${p.symbol} chain from CBOE…`);
    const resp = await fetch(`${API_URL}?${qs}`);
    const json = await resp.json();

    // Discard if a newer fetch has already started (e.g. user switched symbol
    // while the GARCH auto-refetch was still in flight)
    if (mySeq !== _fetchSeq) return;

    if (!resp.ok || json.error) {
      throw new Error(json.detail || json.error || `HTTP ${resp.status}`);
    }

    setStatus('Scoring…');
    state.allRows = json.rows  || [];
    state.meta    = json.meta  || null;

    // Sync dropdown if server fell back to a different model (e.g. survival when GARCH not ready)
    const actualKey = state.meta?.model_key;
    const sel = $('modelSelect');
    if (actualKey && sel && sel.value !== actualKey &&
        sel.querySelector(`option[value="${actualKey}"]`)) {
      sel.value = actualKey;
      updateGarchDisabledState();
    }

    clearError();
    applyFilters();
    updateMeta();
    setStatus('');

    // Hide overlay:
    // - On initial load with garch_ep: only once garch_ep data is truly ready
    // - After that (or for other models): always hide after fetch completes
    const model      = $('modelSelect')?.value;
    const garchReady = state.meta && !state.meta.garch_loading && state.meta.garch_ep_meta != null;
    if (_garchGridRevealed || model !== 'garch_ep' || garchReady) {
      showGridOverlay(false);
    }
  } catch (err) {
    if (mySeq === _fetchSeq) {   // only surface errors from the current fetch
      showError(err.message);
      showGridOverlay(false);
    }
    setStatus('');
  } finally {
    setLoading(false);
  }
}

// ---------------------------------------------------------------------------
// Client-side filtering  (instant — no server call)
// ---------------------------------------------------------------------------
function applyFilters() {
  const f = getFilters();

  state.filteredRows = state.allRows.filter(row => {
    if (row.dte           < f.dte_min)                                  return false;
    if (row.dte           > f.dte_max)                                  return false;
    if (row.open_interest != null && row.open_interest < f.oi_min)      return false;
    if (f.vol_min > 0 && row.volume != null && row.volume < f.vol_min)  return false;
    if (row.moneyness_pct < f.moneyness_min)                            return false;
    if (row.moneyness_pct > f.moneyness_max)                            return false;
    if (f.session !== 'ALL' && row.session !== f.session)               return false;
    return true;
  });

  if (table) table.replaceData(state.filteredRows);
  updateRowCount();
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------
function setLoading(loading, msg = '') {
  state.loading = loading;
  const btn = $('fetchBtn');
  if (btn) btn.disabled = loading;
  const sp = $('spinner');
  if (sp) sp.style.display = loading ? 'inline-block' : 'none';
  if (!loading) setStatus('');
  else if (msg) setStatus(msg);
}

function setStatus(msg) {
  const el = $('statusMsg');
  if (!el) return;
  el.textContent = msg;
  el.style.display = msg ? 'inline' : 'none';
}

function showHelp() { $('helpModal').style.display = 'flex'; }
function hideHelp() { $('helpModal').style.display = 'none'; }

function updateGarchDisabledState() {
  const isGarch = $('modelSelect')?.value === 'garch_ep';
  ['pCrash', 'horizon'].forEach(id => {
    const el = $(id);
    if (!el) return;
    el.closest('.ctrl').classList.toggle('ctrl-disabled', isGarch);
  });
}

let _garchGridRevealed = false;  // true once grid is first shown with real garch_ep data

function showGridOverlay(show) {
  const overlay = $('gridLoadOverlay');
  const grid    = $('grid');
  if (!overlay || !grid) return;
  overlay.style.display = show ? 'flex' : 'none';
  grid.style.display    = show ? 'none' : '';
  if (!show) { _garchGridRevealed = true; setGridHeight(); }
}

function showError(msg) {
  const el = $('errorBanner');
  if (!el) return;
  el.textContent = '⚠ ' + msg;
  el.style.display = 'block';
  setGridHeight(); // re-measure since error banner now visible
}

function clearError() {
  const el = $('errorBanner');
  if (!el) return;
  el.style.display = 'none';
}

function updateMeta() {
  const m = state.meta;
  if (!m) return;

  setText('metaSymbolLabel', m.symbol || 'SPX');
  setText('metaSpot',      m.spx_spot      ? '$' + Number(m.spx_spot).toLocaleString()         : '—');
  setText('metaIV30',      m.iv30          ? m.iv30.toFixed(1) + '%'                           : '—');
  setText('metaN',         m.n_contracts   ? m.n_contracts + ' contracts'                       : '—');
  setText('metaTimestamp', m.cboe_timestamp || '—');
  setText('metaFiltered',  m.n_filtered != null ? m.n_filtered.toLocaleString() + ' filtered out' : '—');

  // Staleness badge
  if (m.fetched_at) {
    const ageMin = Math.floor((Date.now() - new Date(m.fetched_at)) / 60_000);
    const badge  = $('staleBadge');
    if (badge) {
      if (ageMin > 20) {
        badge.textContent   = `${ageMin}m old — refresh?`;
        badge.className     = 'stale-badge ' + (ageMin > 60 ? 'stale-red' : 'stale-yellow');
        badge.style.display = 'inline-block';
      } else {
        badge.style.display = 'none';
      }
    }
  }
}

function updateRowCount() {
  const total   = state.allRows.length;
  const visible = state.filteredRows.length;
  setText('rowCount', visible === total
    ? `${total} puts`
    : `${visible} / ${total} puts`);
}

function setText(id, val) {
  const el = $(id);
  if (el) el.textContent = val;
}

// ---------------------------------------------------------------------------
// Panel collapse/expand
// ---------------------------------------------------------------------------
function togglePanel(panelId, btnId) {
  const panel  = $(panelId);
  const btn    = $(btnId);
  if (!panel || !btn) return;
  const body   = panel.querySelector('.panel-body');
  if (!body) return;
  const hidden = body.style.display === 'none';
  body.style.display = hidden ? '' : 'none';
  btn.textContent    = hidden ? '▲' : '▼';
  setGridHeight();
}

// ---------------------------------------------------------------------------
// Debounce
// ---------------------------------------------------------------------------
function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

const debouncedFetch  = debounce(() => fetchOptions(false), DEBOUNCE_PARAM);
const debouncedFilter = debounce(applyFilters, DEBOUNCE_FILTER);

// ===========================================================================
// TAB SWITCHING
// ===========================================================================
function switchTab(name) {
  const isOptions = name === 'options';
  $('tabContentOptions').style.display = isOptions ? '' : 'none';
  $('tabContentModel').style.display   = isOptions ? 'none' : '';
  $('tabOptions').classList.toggle('tab-active', isOptions);
  $('tabModel').classList.toggle('tab-active', !isOptions);
  if (!isOptions) mbOnShow();
  else setGridHeight();
}

// ===========================================================================
// MODEL BUILDER
// ===========================================================================

const DEFAULT_BUCKETS = [
  [3475, 0.10],
  [4170, 0.25],
  [4860, 0.50],
  [5560, 0.70],
  [6250, 0.90],
];

let pathChart = null;
let termChart = null;
let mbPreviewTimer = null;
let mbLastPreview  = null;

// ── Bucket table ─────────────────────────────────────────────────────────

function mbMovePct(level) {
  const spot = mbLastPreview?.current_price;
  if (!spot || isNaN(level)) return '—';
  const pct = (level / spot - 1) * 100;
  return (pct >= 0 ? '+' : '') + pct.toFixed(1) + '%';
}

function mbRenderBuckets(rows) {
  const tbody = $('bucketBody');
  tbody.innerHTML = '';
  rows.forEach((row, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><input type="number" class="bk-level" data-i="${i}" value="${row[0]}" step="25"
                 oninput="mbUpdateMove(this); mbOnBucketChange()" onblur="mbSortBuckets()"></td>
      <td class="bk-move">${mbMovePct(row[0])}</td>
      <td><input type="number" class="bk-prob"  data-i="${i}" value="${row[1]}" step="0.05" min="0" max="1"
                 oninput="mbOnBucketChange()"></td>
      <td><button class="del-btn" onclick="mbDeleteRow(${i})">×</button></td>`;
    tbody.appendChild(tr);
  });
}

function mbUpdateMove(input) {
  const td = input.closest('tr').querySelector('.bk-move');
  if (td) td.textContent = mbMovePct(parseFloat(input.value));
}

function mbRefreshMoveCells() {
  document.querySelectorAll('#bucketBody tr').forEach(tr => {
    const input = tr.querySelector('.bk-level');
    const td    = tr.querySelector('.bk-move');
    if (input && td) td.textContent = mbMovePct(parseFloat(input.value));
  });
}

function mbGetBuckets() {
  const levels = [...document.querySelectorAll('.bk-level')].map(el => parseFloat(el.value));
  const probs  = [...document.querySelectorAll('.bk-prob')].map(el => parseFloat(el.value));
  return levels.map((l, i) => [l, probs[i]]).filter(([l, p]) => !isNaN(l) && !isNaN(p));
}

function mbAddRow() {
  const buckets = mbGetBuckets();
  buckets.push([Math.round(buckets.length ? buckets[buckets.length - 1][0] + 500 : 4000), 0.5]);
  mbRenderBuckets(buckets);
}

function mbDeleteRow(i) {
  const buckets = mbGetBuckets();
  buckets.splice(i, 1);
  mbRenderBuckets(buckets);
  mbSchedulePreview();
}

function mbSortBuckets() {
  const buckets = mbGetBuckets().sort((a, b) => a[0] - b[0]);
  mbRenderBuckets(buckets);
}

function mbResetBuckets() {
  mbRenderBuckets(DEFAULT_BUCKETS);
  mbSchedulePreview();
}

function mbOnBucketChange() { mbSchedulePreview(); }

// ── Init status polling ──────────────────────────────────────────────────

let mbStatusTimer      = null;
let mbGarchReadyFetched = false;  // ensure we only auto-refetch once per load

function mbPollStatus() {
  fetch('/api/model/status').then(r => r.json()).then(s => {
    const el = $('mbInitStatus');
    if (s.loading) {
      el.className = 'mb-init-status loading';
      el.textContent = `Simulating ${s.n_paths ? (s.n_paths/1000).toFixed(0)+'K' : '100K'} paths… (may take 1–3 min on first load)`;
      mbStatusTimer = setTimeout(mbPollStatus, 1500);
    } else if (s.error) {
      el.className = 'mb-init-status error';
      el.textContent = '⚠ ' + s.error;
      // GARCH failed — reveal grid anyway (server falls back to survival model)
      if (!mbGarchReadyFetched) {
        mbGarchReadyFetched = true;
        showGridOverlay(false);
        fetchOptions(false);
      }
    } else if (s.paths_ready) {
      el.className = 'mb-init-status ready';
      el.textContent = `Ready  ·  SPX @ ${Math.round(s.spx_at_init).toLocaleString()}`;
      if (s.active_model) el.textContent += `  ·  model applied ${s.active_model.configured_at}`;
      // Re-fetch the grid now that GARCH is ready, if it's the active model
      if (!mbGarchReadyFetched && $('modelSelect')?.value === 'garch_ep') {
        mbGarchReadyFetched = true;
        fetchOptions(false);
      }
      // Show fitted drift as placeholder hint
      if (s.fitted_annual_drift != null) {
        const driftEl = $('mbFittedDrift');
        if (driftEl) driftEl.textContent = `fitted: ${s.fitted_annual_drift.toFixed(1)}%/yr`;
        const inp = $('mbDriftInput');
        if (inp && !inp.value) inp.placeholder = `fitted (${s.fitted_annual_drift.toFixed(1)}%)`;
      }
      if (!mbLastPreview) mbPreview();  // auto-preview once ready
    } else {
      el.className = 'mb-init-status loading';
      el.textContent = 'Waiting for simulation to start…';
      mbStatusTimer = setTimeout(mbPollStatus, 2000);
    }
  }).catch(() => {
    mbStatusTimer = setTimeout(mbPollStatus, 3000);
  });
}

function mbReinit() {
  const driftVal = $('mbDriftInput')?.value;
  const body = driftVal ? { annual_drift_pct: parseFloat(driftVal) } : {};
  fetch('/api/model/reinit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(() => {
    $('mbInitStatus').className = 'mb-init-status loading';
    $('mbInitStatus').textContent = 'Restarting simulation…';
    mbLastPreview = null;
    mbGarchReadyFetched = false;
    setTimeout(mbPollStatus, 1000);
  });
}

// ── Preview ──────────────────────────────────────────────────────────────

function mbSchedulePreview() {
  clearTimeout(mbPreviewTimer);
  mbPreviewTimer = setTimeout(mbPreview, 600);
}

async function mbPreview() {
  clearTimeout(mbPreviewTimer);
  const buckets    = mbGetBuckets();
  const confidence = parseFloat($('mbConfidence').value);
  const queryDte   = parseInt($('mbQueryDte').value);

  $('mbPreviewSpinner').style.display = 'inline-block';
  const statusEl = $('mbPreviewStatus');
  if (statusEl) statusEl.textContent = 'Updating…';
  $('mbError').style.display = 'none';

  try {
    const resp = await fetch('/api/model/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ buckets, confidence, query_dte: queryDte }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
    mbLastPreview = data;
    mbRefreshMoveCells();
    mbRenderPreview(data, buckets);
    if (statusEl) statusEl.textContent = '';
  } catch (err) {
    $('mbError').textContent = '⚠ ' + err.message;
    $('mbError').style.display = 'block';
    if (statusEl) statusEl.textContent = '';
  } finally {
    $('mbPreviewSpinner').style.display = 'none';
  }
}

// ── Commit ───────────────────────────────────────────────────────────────

async function mbCommit() {
  const buckets    = mbGetBuckets();
  const confidence = parseFloat($('mbConfidence').value);
  if (!buckets.length) {
    $('mbError').textContent = '⚠ Add at least one bucket row before applying.';
    $('mbError').style.display = 'block';
    return;
  }
  $('mbCommitSpinner').style.display = 'inline-block';
  $('mbCommitBtn').disabled = true;
  $('mbCommitStatus').style.display = 'none';
  $('mbError').style.display = 'none';

  try {
    const resp = await fetch('/api/model/commit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ buckets, confidence }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
    $('mbCommitStatus').textContent =
      `✓ Applied to grid at ${data.meta.configured_at} — switch to Options Grid tab and select GARCH/EP model`;
    $('mbCommitStatus').style.display = 'block';
    // Auto-select garch_ep in options tab
    const sel = $('modelSelect');
    if (sel) { sel.value = 'garch_ep'; debouncedFetch(); }
  } catch (err) {
    $('mbError').textContent = '⚠ ' + err.message;
    $('mbError').style.display = 'block';
  } finally {
    $('mbCommitSpinner').style.display = 'none';
    $('mbCommitBtn').disabled = false;
  }
}

// ── Render preview results ────────────────────────────────────────────────

const QUINTILE_COLORS = [
  '#f47067', '#db9b56', '#c1c346', '#56c445', '#45b8c4',
];

function mbRenderPreview(data, buckets) {
  mbRenderPriorTable(data.prior_stats, buckets);
  mbRenderPathChart(data.quintile_paths, data.quintile_groups, data.quintile_stats, data.current_price);
  mbRenderTermChart(data.terminal);
  mbRenderQuintileStats(data.quintile_stats);
}

function mbRenderPriorTable(stats, buckets) {
  // Build a CDF interpolator from user buckets
  const bkSorted = [...buckets].sort((a, b) => a[0] - b[0]);
  function viewCDF(level) {
    if (!bkSorted.length) return null;
    if (level <= bkSorted[0][0]) return bkSorted[0][1];
    if (level >= bkSorted[bkSorted.length - 1][0]) return bkSorted[bkSorted.length - 1][1];
    for (let i = 0; i < bkSorted.length - 1; i++) {
      const [l0, p0] = bkSorted[i], [l1, p1] = bkSorted[i + 1];
      if (level >= l0 && level <= l1) {
        return p0 + (p1 - p0) * (level - l0) / (l1 - l0);
      }
    }
    return null;
  }

  const tbody = $('priorBody');
  tbody.innerHTML = '';
  stats.forEach(s => {
    const vp = viewCDF(s.level);
    const viewStr = vp !== null ? (vp * 100).toFixed(0) + '%' : '—';
    const diff = vp !== null ? vp - s.prob : null;
    const viewClass = diff === null ? '' : diff > 0.02 ? 'view-bearish' : diff < -0.02 ? 'view-bullish' : '';
    const tr = document.createElement('tr');
    if (s.is_tilt_threshold) tr.className = 'tilt-row';
    tr.innerHTML = `
      <td>>−${(s.drawdown * 100).toFixed(0)}%</td>
      <td>${s.level.toLocaleString()}</td>
      <td>${(s.prob * 100).toFixed(0)}%</td>
      <td class="${viewClass}">${viewStr}</td>`;
    tbody.appendChild(tr);
  });
}

function mbRenderPathChart(paths, groups, stats, currentPrice) {
  const ctx = $('pathChart').getContext('2d');
  const n = paths[0].length;
  // x-axis labels: every ~13 points ≈ 3 months (step=5 trading days per point)
  const labels = Array.from({ length: n }, (_, i) => {
    const months = Math.round(i * 5 / 21);
    return i === 0 ? 'Now' : (i % 13 === 0 ? `${months}m` : '');
  });

  const datasets = paths.map((path, i) => {
    const q = groups[i];
    return {
      label: `Q${q + 1}`,
      data: path,
      borderColor: QUINTILE_COLORS[q],
      borderWidth: 1,
      pointRadius: 0,
      tension: 0.2,
    };
  });

  if (pathChart) pathChart.destroy();
  pathChart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'nearest', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => items[0].label,
            label: (item) => ` Q${groups[item.datasetIndex] + 1}: ${Math.round(item.raw).toLocaleString()}`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: '#4a5568', font: { size: 10 }, maxRotation: 0 },
          grid:  { color: '#1e2535' },
        },
        y: {
          ticks: { color: '#768390', font: { size: 10 }, callback: v => v.toLocaleString() },
          grid:  { color: '#1e2535' },
        },
      },
    },
  });
}

function mbRenderTermChart(terminal) {
  const ctx = $('termChart').getContext('2d');
  const labels = terminal.levels.map(l => Math.round(l).toLocaleString());

  if (termChart) termChart.destroy();
  termChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'GARCH prior',
          data: terminal.prior_probs,
          backgroundColor: 'rgba(118,131,144,0.4)',
          borderColor: 'rgba(118,131,144,0.0)',
          borderWidth: 0,
          barPercentage: 1.0,
          categoryPercentage: 1.0,
        },
        {
          label: 'EP posterior',
          data: terminal.posterior_probs,
          backgroundColor: 'rgba(77,148,255,0.5)',
          borderColor: 'rgba(77,148,255,0.0)',
          borderWidth: 0,
          barPercentage: 1.0,
          categoryPercentage: 1.0,
        },
      ],
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          labels: { color: '#768390', font: { size: 10 }, boxWidth: 12, padding: 8 },
        },
        tooltip: { mode: 'index', intersect: false },
      },
      scales: {
        x: {
          ticks: {
            color: '#4a5568', font: { size: 9 }, maxRotation: 45,
            maxTicksLimit: 12,
          },
          grid: { display: false },
        },
        y: {
          ticks: { color: '#768390', font: { size: 10 } },
          grid:  { color: '#1e2535' },
        },
      },
    },
  });
}

function mbRenderQuintileStats(stats) {
  const el = $('decileWeights');
  el.innerHTML = stats.map((s, i) => {
    const c = QUINTILE_COLORS[i];
    return `<span class="dw-chip" style="background:${c}22;color:${c};border-color:${c}44"
      title="avg bottom: ${s.drawdown_pct}%  |  avg terminal: ${s.v_terminal?.toLocaleString()}">
      Q${i + 1}  ${s.drawdown_pct}%</span>`;
  }).join('');
}

function mbOnShow() {
  // Start polling if not already running
  if (mbStatusTimer === null) {
    mbStatusTimer = 0; // sentinel — prevents double-start
    mbPollStatus();
  }
  if (!mbLastPreview) {
    mbRenderBuckets(DEFAULT_BUCKETS);
  }
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') hideHelp();
});

document.addEventListener('DOMContentLoaded', () => {
  // Verify Tabulator loaded
  if (typeof Tabulator === 'undefined') {
    showError('Tabulator failed to load from CDN. Check your internet connection and reload.');
    return;
  }

  initTable();

  // Height: set after first paint, then keep updated on resize
  requestAnimationFrame(setGridHeight);
  window.addEventListener('resize', debounce(() => {
    setGridHeight();
    if (table) table.redraw(true);
  }, 100));

  // Fetch button
  $('fetchBtn').addEventListener('click', () => fetchOptions(true));

  // Param inputs → re-score server-side
  ['pCrash', 'horizon', 'rothMult', 'contracts'].forEach(id => {
    $(id)?.addEventListener('input', debouncedFetch);
  });
  $('modelSelect')?.addEventListener('change', () => {
    updateGarchDisabledState();
    if ($('modelSelect').value !== 'garch_ep') showGridOverlay(false);
    debouncedFetch();
  });
  updateGarchDisabledState();

  // Symbol change: auto-fill crash-beta default, show overlay immediately, force-fetch.
  // Reset state.loading so a concurrent fetch never silently blocks the symbol switch.
  $('symbolSelect')?.addEventListener('change', () => {
    const sym    = $('symbolSelect').value;
    const betaEl = $('indexBeta');
    if (betaEl) betaEl.value = SYMBOL_BETA_DEFAULTS[sym] ?? 1.0;
    state.loading = false;
    showGridOverlay(true);
    fetchOptions(true);
  });
  $('indexBeta')?.addEventListener('input', debouncedFetch);

  // Filter inputs → instant client-side filter
  ['dteMin', 'dteMax', 'oiMin', 'volMin', 'moneynessMin', 'moneynessMax'].forEach(id => {
    $(id)?.addEventListener('input', debouncedFilter);
  });
  $('sessionFilter')?.addEventListener('change', debouncedFilter);

  // Panel toggles
  $('toggleParams')?.addEventListener('click', () => togglePanel('paramPanel', 'toggleParams'));
  $('toggleFilters')?.addEventListener('click', () => togglePanel('filterPanel', 'toggleFilters'));

  // Model builder: render default buckets and start GARCH status polling
  // immediately (not just when the tab is opened) so the grid auto-updates
  // once GARCH finishes loading, even if the user never visits that tab.
  mbRenderBuckets(DEFAULT_BUCKETS);
  mbStatusTimer = 0;  // sentinel
  mbPollStatus();

  // Show overlay if GARCH/EP is active — grid will appear once model is ready
  if ($('modelSelect')?.value === 'garch_ep') showGridOverlay(true);

  // Safety valve: if GARCH hasn't finished in 5 minutes, show grid anyway
  setTimeout(() => {
    if (!_garchGridRevealed) {
      console.warn('GARCH 5-min timeout — revealing grid with fallback model');
      showGridOverlay(false);
      if (!mbGarchReadyFetched) { mbGarchReadyFetched = true; fetchOptions(false); }
    }
  }, 300_000);

  // Auto-fetch on load (warms CBOE cache; grid revealed once GARCH is ready)
  fetchOptions(true);

  // Heartbeat: ping server every 30 s so it knows the browser is open.
  // Server auto-shuts-down after 90 s of silence (e.g. browser closed).
  function sendHeartbeat() { fetch('/api/heartbeat', { method: 'POST' }).catch(() => {}); }
  sendHeartbeat();
  setInterval(sendHeartbeat, 30_000);
});
