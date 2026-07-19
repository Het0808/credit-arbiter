const API_BASE = 'http://localhost:8000/api';

// DOM Elements
const appEl = document.getElementById('app');
const loginForm = document.getElementById('login-form');
const registerForm = document.getElementById('register-form');
const dashboardView = document.getElementById('dashboard-view');
const showRegisterLink = document.getElementById('show-register');
const showLoginLink = document.getElementById('show-login');
const alertBox = document.getElementById('alert-box');
const logoutBtn = document.getElementById('logout-btn');
const userInfo = document.getElementById('user-info');

// Queue / detail / assess elements
const queueView = document.getElementById('queue-view');
const queueRows = document.getElementById('queue-rows');
const detailView = document.getElementById('detail-view');
const backToQueueBtn = document.getElementById('back-to-queue-btn');
const detailExternalId = document.getElementById('detail-external-id');
const detailIncompleteBadge = document.getElementById('detail-incomplete-badge');
const detailMissingFields = document.getElementById('detail-missing-fields');
const detailProfile = document.getElementById('detail-profile');
const assessBtn = document.getElementById('assess-btn');
const assessmentResult = document.getElementById('assessment-result');
const escalationBanner = document.getElementById('escalation-banner');
const evRisk = document.getElementById('ev-risk');
const evPolicy = document.getElementById('ev-policy');
const evDocs = document.getElementById('ev-docs');
const evRegulatory = document.getElementById('ev-regulatory');
const evFairness = document.getElementById('ev-fairness');
const resultRecommendation = document.getElementById('result-recommendation');
const decisionControls = document.getElementById('decision-controls');
const acceptBtn = document.getElementById('accept-btn');
const showOverrideBtn = document.getElementById('show-override-btn');
const overrideForm = document.getElementById('override-form');
const overrideReason = document.getElementById('override-reason');
const overrideReasonCode = document.getElementById('override-reason-code');
const submitOverrideBtn = document.getElementById('submit-override-btn');
const decisionConfirmation = document.getElementById('decision-confirmation');
const clauseModal = document.getElementById('clause-modal');
const clauseModalBody = document.getElementById('clause-modal-body');
const clauseModalClose = document.getElementById('clause-modal-close');

const escapeHtml = (s) => String(s ?? '').replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

// State
let token = localStorage.getItem('halcyon_token');
let currentApplicationId = null;
let currentAssessmentId = null;

// Initialize
function init() {
  if (token) {
    fetchUserDetails();
  } else {
    showLogin();
  }
}

// Navigation
function showLogin() {
  appEl.classList.remove('dashboard-wide');
  loginForm.classList.remove('hidden');
  registerForm.classList.add('hidden');
  dashboardView.classList.add('hidden');
  hideAlert();
}

function showRegister() {
  appEl.classList.remove('dashboard-wide');
  loginForm.classList.add('hidden');
  registerForm.classList.remove('hidden');
  dashboardView.classList.add('hidden');
  hideAlert();
}

function showDashboard(email) {
  appEl.classList.add('dashboard-wide');
  loginForm.classList.add('hidden');
  registerForm.classList.add('hidden');
  dashboardView.classList.remove('hidden');
  userInfo.textContent = `Logged in as: ${email}`;
  hideAlert();
  showQueue();
  fetchQueue();
}

function showQueue() {
  queueView.classList.remove('hidden');
  detailView.classList.add('hidden');
}

function showDetail() {
  queueView.classList.add('hidden');
  detailView.classList.remove('hidden');
}

// Alerts
function showAlert(message, isError = true) {
  alertBox.textContent = message;
  alertBox.classList.remove('hidden');
  alertBox.style.color = isError ? 'var(--error)' : 'var(--primary)';
  alertBox.style.borderColor = isError ? 'var(--error)' : 'var(--primary)';
  alertBox.style.background = isError ? 'rgba(255, 75, 75, 0.1)' : 'rgba(102, 252, 241, 0.1)';
}

function hideAlert() {
  alertBox.classList.add('hidden');
}

function formatCurrency(value) {
  if (value === null || value === undefined) return '-';
  return `$${Number(value).toLocaleString()}`;
}

// API Calls
async function handleLogin(e) {
  e.preventDefault();
  const email = document.getElementById('login-email').value;
  const password = document.getElementById('login-password').value;

  try {
    const formData = new URLSearchParams();
    formData.append('username', email);
    formData.append('password', password);

    const res = await fetch(`${API_BASE}/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
      },
      body: formData
    });

    if (!res.ok) {
      const errorData = await res.json();
      throw new Error(errorData.detail || 'Login failed');
    }

    const data = await res.json();
    token = data.access_token;
    localStorage.setItem('halcyon_token', token);
    fetchUserDetails();
  } catch (error) {
    showAlert(error.message);
  }
}

async function handleRegister(e) {
  e.preventDefault();
  const email = document.getElementById('reg-email').value;
  const password = document.getElementById('reg-password').value;

  try {
    const res = await fetch(`${API_BASE}/auth/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ email, password })
    });

    if (!res.ok) {
      const errorData = await res.json();
      throw new Error(errorData.detail || 'Registration failed');
    }

    const data = await res.json();
    token = data.access_token;
    localStorage.setItem('halcyon_token', token);
    fetchUserDetails();
  } catch (error) {
    showAlert(error.message);
  }
}

async function fetchUserDetails() {
  try {
    const res = await fetch(`${API_BASE}/users/me`, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });

    if (!res.ok) {
      throw new Error('Session expired');
    }

    const user = await res.json();
    showDashboard(user.email);
  } catch (error) {
    token = null;
    localStorage.removeItem('halcyon_token');
    showLogin();
  }
}

async function fetchQueue() {
  try {
    const res = await fetch(`${API_BASE}/applications`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    if (!res.ok) throw new Error('Failed to load application queue');
    const applications = await res.json();
    renderQueue(applications);
  } catch (error) {
    showAlert(error.message);
  }
}

function renderQueue(applications) {
  queueRows.innerHTML = '';
  applications.forEach((application) => {
    const row = document.createElement('tr');
    row.className = 'queue-row';
    row.innerHTML = `
      <td>${application.external_id}</td>
      <td>${application.name_contract_type || '-'}</td>
      <td>${formatCurrency(application.amt_income_total)}</td>
      <td>${formatCurrency(application.amt_credit)}</td>
      <td>${application.status}</td>
    `;
    row.addEventListener('click', () => openApplication(application.id));
    queueRows.appendChild(row);
  });
}

async function openApplication(id) {
  try {
    const res = await fetch(`${API_BASE}/applications/${id}`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    if (!res.ok) throw new Error('Failed to load application detail');
    const application = await res.json();

    currentApplicationId = application.id;
    currentAssessmentId = null;

    renderDetail(application);
    assessmentResult.classList.add('hidden');
    showDetail();
  } catch (error) {
    showAlert(error.message);
  }
}

function renderDetail(application) {
  detailExternalId.textContent = application.external_id;

  if (application.status === 'INCOMPLETE') {
    detailIncompleteBadge.classList.remove('hidden');
    detailMissingFields.textContent = application.missing_fields || '';
  } else {
    detailIncompleteBadge.classList.add('hidden');
  }

  const fields = [
    ['Contract Type', application.name_contract_type],
    ['Annual Income', formatCurrency(application.amt_income_total)],
    ['Credit Amount', formatCurrency(application.amt_credit)],
    ['Annuity', formatCurrency(application.amt_annuity)],
    ['Employment (days)', application.days_employed],
    ['Education', application.name_education_type],
    ['Family Status', application.name_family_status],
    ['Region Rating', application.region_rating_client],
    ['Occupation', application.occupation_type],
  ];

  detailProfile.innerHTML = fields
    .map(([label, value]) => `
      <div class="profile-field">
        <label>${label}</label>
        <span>${value === null || value === undefined || value === '' ? '-' : value}</span>
      </div>
    `)
    .join('');
}

async function handleAssess() {
  try {
    const res = await fetch(`${API_BASE}/assess`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({ application_id: currentApplicationId })
    });

    if (!res.ok) {
      const errorData = await res.json();
      throw new Error(errorData.detail || 'Assessment failed');
    }

    const decision = await res.json();
    currentAssessmentId = decision.id;
    renderAssessment(decision);
  } catch (error) {
    showAlert(error.message);
  }
}

function renderAssessment(decision) {
  assessmentResult.classList.remove('hidden');
  decisionConfirmation.classList.add('hidden');
  overrideForm.classList.add('hidden');
  decisionControls.classList.remove('hidden');

  const ev = decision.evidence_chain || {};
  const ts = decision.created_at ? new Date(decision.created_at).toLocaleString() : '';
  ['ev-risk-time', 'ev-policy-time', 'ev-docs-time', 'ev-reg-time', 'ev-fair-time']
    .forEach((id) => { document.getElementById(id).textContent = ts; });

  if (decision.escalation_flag) {
    const code = decision.escalation_reason_code ? ` — ${decision.escalation_reason_code.replace(/_/g, ' ')}` : '';
    escalationBanner.textContent = `Escalated for human review${code}`;
    escalationBanner.classList.remove('hidden');
  } else {
    escalationBanner.classList.add('hidden');
  }

  // 1 · Risk & factors (SHAP-style contributors)
  const bandClass = decision.risk_band ? `badge-${decision.risk_band.toLowerCase()}` : '';
  let riskHtml = decision.risk_score != null
    ? `<div><span class="badge ${bandClass}">${decision.risk_band}</span> &nbsp;${(decision.risk_score * 100).toFixed(1)}% PD`
      + (ev.model_confidence != null ? ` · confidence ${(ev.model_confidence * 100).toFixed(0)}%` : '') + `</div>`
    : '<div>No risk score available</div>';
  if (Array.isArray(ev.risk_factors) && ev.risk_factors.length) {
    riskHtml += '<ul class="factor-list">' + ev.risk_factors.map((f) =>
      `<li><span class="factor-dir factor-${f.direction === 'increases_risk' ? 'up' : 'down'}">${f.direction === 'increases_risk' ? '▲' : '▼'}</span> ${escapeHtml(f.label)} <span class="factor-val">${f.contribution}</span></li>`
    ).join('') + '</ul>';
  }
  evRisk.innerHTML = riskHtml;

  // 2 · Policy clauses (clickable citations -> source_id + version)
  const clauses = ev.policy_clauses || [];
  if (clauses.length) {
    evPolicy.innerHTML = clauses.map((c) =>
      `<button class="citation" data-clause="${escapeHtml(c.clause_id)}" data-version="${escapeHtml(c.corpus_version || '')}">${escapeHtml(c.clause_id)} <span class="citation-ver">${escapeHtml(c.corpus_version || '')}</span></button>`
    ).join('')
    + `<div class="evidence-meta">Adherence: ${((ev.policy_adherence ?? 1) * 100).toFixed(0)}%`
    + (ev.policy_failed_rules && ev.policy_failed_rules.length ? ` · failed: ${ev.policy_failed_rules.map((r) => r.clause_id).join(', ')}` : '') + `</div>`;
  } else {
    evPolicy.innerHTML = '<span class="evidence-empty">No policy clause retrieved (retrieval failed)</span>';
  }

  // 3 · Documents
  const docs = ev.document_findings;
  if (docs) {
    const missing = docs.missing_information || [];
    const findings = docs.consistency_findings || [];
    evDocs.innerHTML =
      `<div>${docs.verified ? '<span class="badge badge-low">Verified</span>' : '<span class="badge badge-medium">Needs review</span>'} · ${docs.document_count ?? 0} document(s)</div>`
      + (missing.length ? `<div class="evidence-meta">Missing: ${missing.map(escapeHtml).join(', ')}</div>` : '')
      + (findings.length ? `<div class="evidence-meta">Findings: ${findings.map((f) => escapeHtml(f.type)).join(', ')}</div>` : '');
  } else {
    evDocs.innerHTML = '<span class="evidence-empty">No document report</span>';
  }

  // 4 · Regulatory (per-check breakdown)
  const checks = ev.regulatory_checks || [];
  evRegulatory.innerHTML =
    `<div><span class="badge badge-${(decision.regulatory_status || '').toLowerCase() === 'pass' ? 'low' : 'high'}">${decision.regulatory_status || '-'}</span></div>`
    + (checks.length ? '<div class="check-grid">' + checks.map((c) =>
        `<span class="check-chip check-${c.status.toLowerCase().includes('pass') ? 'ok' : 'bad'}">${escapeHtml(c.check)}: ${escapeHtml(c.status)}</span>`
      ).join('') + '</div>' : '')
    + (ev.regulatory_reason ? `<div class="evidence-meta">${escapeHtml(ev.regulatory_reason)}</div>` : '');

  // 5 · Fairness
  const fair = ev.fairness_result || {};
  evFairness.innerHTML = fair.scheme_paused
    ? `<span class="badge badge-high">Scheme paused (fairness hard-block)</span>`
    : `<span class="badge badge-low">No active fairness block</span> <span class="evidence-meta">${escapeHtml(fair.scheme || '')}</span>`;

  const recommendationClass = `badge-${decision.recommendation.toLowerCase()}`;
  resultRecommendation.innerHTML = `<span class="badge ${recommendationClass}">${decision.recommendation}</span>`;

  evPolicy.querySelectorAll('.citation').forEach((btn) => {
    btn.addEventListener('click', () => openClause(btn.dataset.clause, btn.dataset.version));
  });
}

async function openClause(clauseId, version) {
  try {
    const q = version ? `?corpus_version=${encodeURIComponent(version)}` : '';
    const res = await fetch(`${API_BASE}/policy/clause/${encodeURIComponent(clauseId)}${q}`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    if (!res.ok) throw new Error('Clause not found');
    const clause = await res.json();
    clauseModalBody.innerHTML =
      `<div class="clause-title">${escapeHtml(clause.clause_id)} <span class="citation-ver">${escapeHtml(clause.corpus_version)}</span></div>`
      + `<div class="clause-scheme">${escapeHtml(clause.scheme)} — ${escapeHtml(clause.title)}</div>`
      + `<div class="clause-text">${escapeHtml(clause.text)}</div>`;
    clauseModal.classList.remove('hidden');
  } catch (error) {
    showAlert(error.message);
  }
}

async function handleDecision(action, reason, reasonCode) {
  try {
    const res = await fetch(`${API_BASE}/assessments/${currentAssessmentId}/decision`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({ action, reason: reason || null, reason_code: reasonCode || null })
    });

    if (!res.ok) {
      const errorData = await res.json();
      throw new Error(errorData.detail || 'Failed to record decision');
    }

    const record = await res.json();
    decisionControls.classList.add('hidden');
    overrideForm.classList.add('hidden');
    decisionConfirmation.classList.remove('hidden');
    decisionConfirmation.textContent =
      `Recorded: ${record.underwriter_action} at ${new Date(record.underwriter_action_at).toLocaleString()}`;
    decisionConfirmation.style.color = 'var(--primary)';
    decisionConfirmation.style.borderColor = 'var(--primary)';
    decisionConfirmation.style.background = 'rgba(102, 252, 241, 0.1)';
  } catch (error) {
    showAlert(error.message);
  }
}

// Event Listeners
loginForm.addEventListener('submit', handleLogin);
registerForm.addEventListener('submit', handleRegister);
showRegisterLink.addEventListener('click', (e) => { e.preventDefault(); showRegister(); });
showLoginLink.addEventListener('click', (e) => { e.preventDefault(); showLogin(); });
logoutBtn.addEventListener('click', () => {
  token = null;
  localStorage.removeItem('halcyon_token');
  showLogin();
});

backToQueueBtn.addEventListener('click', () => { showQueue(); fetchQueue(); });
assessBtn.addEventListener('click', handleAssess);
acceptBtn.addEventListener('click', () => handleDecision('accept'));
showOverrideBtn.addEventListener('click', () => {
  overrideForm.classList.remove('hidden');
  decisionControls.classList.add('hidden');
});
submitOverrideBtn.addEventListener('click', () => {
  const reason = overrideReason.value.trim();
  const reasonCode = overrideReasonCode.value;
  if (!reasonCode) {
    showAlert('A reason code is required to override a recommendation');
    return;
  }
  if (!reason) {
    showAlert('A reason is required to override a recommendation');
    return;
  }
  handleDecision('override', reason, reasonCode);
});

clauseModalClose.addEventListener('click', () => clauseModal.classList.add('hidden'));
clauseModal.addEventListener('click', (e) => {
  if (e.target === clauseModal) clauseModal.classList.add('hidden');
});

// --- Ops dashboard (US-407) ---
const opsToggleBtn = document.getElementById('ops-toggle-btn');
const opsPanel = document.getElementById('ops-panel');
const opsKpis = document.getElementById('ops-kpis');
const opsAlerts = document.getElementById('ops-alerts');
const killSwitchInput = document.getElementById('kill-switch-input');

const fmtPct = (v) => (v == null ? '—' : `${(v * 100).toFixed(0)}%`);
const fmtUsd = (v) => (v == null ? '—' : `$${Number(v).toFixed(4)}`);
const fmtNum = (v, unit = '') => (v == null ? '—' : `${v}${unit}`);

async function loadDashboard() {
  try {
    const [dashRes, ksRes] = await Promise.all([
      fetch(`${API_BASE}/ops/dashboard`, { headers: { 'Authorization': `Bearer ${token}` } }),
      fetch(`${API_BASE}/ops/kill-switch`, { headers: { 'Authorization': `Bearer ${token}` } }),
    ]);
    if (!dashRes.ok) throw new Error('Failed to load dashboard');
    const dash = await dashRes.json();
    const ks = await ksRes.json();
    killSwitchInput.checked = !!ks.active;

    const k = dash.kpis;
    const tiles = [
      ['Throughput', fmtNum(k.throughput_decisions), `${k.window_days}d`],
      ['P95 latency', fmtNum(k.p95_latency_s, ' s'), `≤ ${dash.thresholds.p95_latency_s}s`],
      ['Cost / app', fmtUsd(k.cost_per_app_usd), `≤ $${dash.thresholds.cost_per_app_usd}`],
      ['Acceptance', fmtPct(k.acceptance_rate), `≥ ${fmtPct(dash.thresholds.acceptance_rate_min)}`],
      ['Override rate', fmtPct(k.override_rate), ''],
      ['Fairness gap', fmtNum(k.fairness_gap_pp, ' pp'), `≤ ${dash.thresholds.fairness_gap_pp}pp`],
    ];
    opsKpis.innerHTML = tiles.map(([label, val, sub]) =>
      `<div class="kpi-tile"><div class="kpi-label">${label}</div><div class="kpi-value">${val}</div><div class="kpi-sub">${sub}</div></div>`
    ).join('');

    if (dash.alerts && dash.alerts.length) {
      opsAlerts.innerHTML = dash.alerts.map((a) => `<div>⚠ ${escapeHtml(a)}</div>`).join('');
      opsAlerts.classList.remove('hidden');
    } else {
      opsAlerts.classList.add('hidden');
    }
  } catch (error) {
    showAlert(error.message);
  }
}

opsToggleBtn.addEventListener('click', () => {
  opsPanel.classList.toggle('hidden');
  if (!opsPanel.classList.contains('hidden')) loadDashboard();
});

killSwitchInput.addEventListener('change', async () => {
  try {
    await fetch(`${API_BASE}/ops/kill-switch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
      body: JSON.stringify({ active: killSwitchInput.checked }),
    });
  } catch (error) {
    showAlert(error.message);
  }
});

// Start
init();
