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

// Ops dashboard elements
const showOpsBtn = document.getElementById('show-ops-btn');
const opsBackBtn = document.getElementById('ops-back-btn');
const opsView = document.getElementById('ops-view');
const opsStatGrid = document.getElementById('ops-stat-grid');

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
const resultRisk = document.getElementById('result-risk');
const resultRiskFactors = document.getElementById('result-risk-factors');
const resultScheme = document.getElementById('result-scheme');
const resultClause = document.getElementById('result-clause');
const resultPolicyRules = document.getElementById('result-policy-rules');
const resultDocuments = document.getElementById('result-documents');
const resultRegulatory = document.getElementById('result-regulatory');
const resultFairness = document.getElementById('result-fairness');
const resultExplanation = document.getElementById('result-explanation');
const resultRecommendation = document.getElementById('result-recommendation');
const decisionControls = document.getElementById('decision-controls');
const acceptBtn = document.getElementById('accept-btn');
const showOverrideBtn = document.getElementById('show-override-btn');
const overrideForm = document.getElementById('override-form');
const overrideReasonCode = document.getElementById('override-reason-code');
const overrideReason = document.getElementById('override-reason');
const submitOverrideBtn = document.getElementById('submit-override-btn');
const decisionConfirmation = document.getElementById('decision-confirmation');

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
  opsView.classList.add('hidden');
}

function showDetail() {
  queueView.classList.add('hidden');
  detailView.classList.remove('hidden');
  opsView.classList.add('hidden');
}

function showOps() {
  queueView.classList.add('hidden');
  detailView.classList.add('hidden');
  opsView.classList.remove('hidden');
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

async function fetchMetrics() {
  try {
    const res = await fetch(`${API_BASE}/metrics`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    if (!res.ok) throw new Error('Failed to load ops metrics');
    const metrics = await res.json();
    renderMetrics(metrics);
  } catch (error) {
    showAlert(error.message);
  }
}

function formatPercent(value) {
  return value === null || value === undefined ? '-' : `${(value * 100).toFixed(1)}%`;
}

function renderMetrics(metrics) {
  const recCounts = metrics.recommendation_counts || {};
  const recSummary = Object.entries(recCounts).map(([k, v]) => `${k}: ${v}`).join(' &middot; ') || '-';

  const cards = [
    ['Throughput', metrics.throughput],
    ['Recommendation Mix', recSummary],
    ['Escalation Rate', formatPercent(metrics.escalation_rate)],
    ['Acceptance Rate', formatPercent(metrics.acceptance_rate)],
    ['Override Rate', formatPercent(metrics.override_rate)],
    ['Avg Cost / Assessment', metrics.avg_cost_usd !== null ? `$${metrics.avg_cost_usd.toFixed(6)}` : '-'],
    ['Avg Latency', metrics.avg_latency_ms !== null ? `${metrics.avg_latency_ms.toFixed(1)} ms` : '-'],
    ['P95 Latency', metrics.p95_latency_ms !== null ? `${metrics.p95_latency_ms.toFixed(1)} ms` : '-'],
    ['Cost Guardrail', `$${metrics.cost_guardrail_usd.toFixed(2)} / assessment`],
    ['Fairness Hard-Block', `${metrics.fairness_hard_block_pp} pp`],
    [
      'Retrieval Failure Rate',
      metrics.retrieval_failure_alert
        ? `<span style="color:var(--error);font-weight:600">${formatPercent(metrics.retrieval_failure_rate)} (ALERT)</span>`
        : formatPercent(metrics.retrieval_failure_rate),
    ],
  ];

  opsStatGrid.innerHTML = cards
    .map(([label, value]) => `
      <div class="stat-card">
        <div class="stat-label">${label}</div>
        <div class="stat-value">${value}</div>
      </div>
    `)
    .join('');
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

  escalationBanner.classList.toggle('hidden', !decision.escalation_flag);

  const bandClass = decision.risk_band ? `badge-${decision.risk_band.toLowerCase()}` : '';
  resultRisk.innerHTML = decision.risk_score !== null && decision.risk_score !== undefined
    ? `<span class="badge ${bandClass}">${decision.risk_band}</span>&nbsp; ${(decision.risk_score * 100).toFixed(1)}% probability of default`
    : 'No risk score available';

  const riskFactors = decision.evidence_chain?.risk_factors || [];
  resultRiskFactors.innerHTML = riskFactors.length
    ? riskFactors.map((factor) => {
        const direction = factor.impact >= 0 ? 'factor-up' : 'factor-down';
        const arrow = factor.impact >= 0 ? '&uarr;' : '&darr;';
        return `
          <div class="risk-factor ${direction}">
            <span class="risk-factor-arrow">${arrow}</span>
            <span class="risk-factor-label">${factor.label}</span>
            <span class="risk-factor-value">${typeof factor.value === 'number' ? factor.value.toFixed(2) : factor.value ?? '-'}</span>
          </div>
        `;
      }).join('')
    : '<span>No risk factors available</span>';

  const ev = decision.evidence_chain || {};

  resultScheme.textContent = ev.loan_scheme || '-';

  const allClauses = ev.retrieved_clauses && ev.retrieved_clauses.length
    ? ev.retrieved_clauses
    : (decision.retrieved_clause_id
        ? [{ clause_id: decision.retrieved_clause_id, title: '', text: decision.retrieved_clause_text, score: decision.retrieval_confidence }]
        : []);
  resultClause.innerHTML = allClauses.length
    ? allClauses.map((c) => `
        <div class="clause-item">
          <div class="clause-title">${c.clause_id}${c.title ? ' &mdash; ' + c.title : ''}</div>
          <div class="clause-text">${c.text || ''}</div>
          <div class="clause-meta">Confidence: ${((c.score || 0) * 100).toFixed(1)}%</div>
        </div>
      `).join('')
    : '<span>No policy clause retrieved (retrieval failed)</span>';

  const passedRules = ev.policy_passed_rules || [];
  const failedRules = ev.policy_failed_rules || [];
  resultPolicyRules.innerHTML = (passedRules.length + failedRules.length)
    ? [
        ...passedRules.map((r) => `<div class="rule-item rule-pass">&check; ${r.rule}: ${r.detail}</div>`),
        ...failedRules.map((r) => `<div class="rule-item rule-fail">&cross; ${r.rule}: ${r.detail}</div>`),
      ].join('')
    : '<span>No scheme-specific numeric rules evaluated</span>';

  const docVerification = ev.document_verification;
  resultDocuments.innerHTML = docVerification
    ? `
        <div class="doc-status ${docVerification.complete ? 'doc-ok' : 'doc-missing'}">
          ${docVerification.complete ? 'All required documents present' : 'Missing: ' + docVerification.missing_documents.join(', ')}
        </div>
        <div class="doc-status ${docVerification.consistent ? 'doc-ok' : 'doc-missing'}">
          ${docVerification.consistent ? 'No consistency issues found' : 'Consistency findings: ' + docVerification.consistency_findings.join(', ')}
        </div>
      `
    : '<span>-</span>';

  const subChecks = ev.regulatory_sub_checks || {};
  const subCheckKeys = Object.keys(subChecks);
  resultRegulatory.innerHTML = `
    <span class="badge badge-${(decision.regulatory_status || '').toLowerCase() === 'pass' ? 'approve' : 'decline'}">${decision.regulatory_status || '-'}</span>
    ${subCheckKeys.length ? '<div class="sub-check-list">' + subCheckKeys.map((k) => `<span class="sub-check-item">${k}: ${subChecks[k]}</span>`).join('') + '</div>' : ''}
  `;

  const triggeredSegments = ev.fairness_triggered_segments || [];
  resultFairness.innerHTML = ev.fairness_alert
    ? `<div class="fairness-alert">Fairness alert: ${triggeredSegments.map((s) => `${s.attribute}=${s.subgroup} (${s.delta_pp > 0 ? '+' : ''}${s.delta_pp}pp)`).join(', ')}</div>`
    : '<span>No fairness gap exceeding the 5pp guardrail</span>';

  resultExplanation.innerHTML = ev.narrative_explanation
    ? `<p>${ev.narrative_explanation}</p><div class="clause-meta">Source: ${ev.explanation_source}${ev.explanation_cost_usd ? ' &middot; $' + ev.explanation_cost_usd.toFixed(6) : ''}</div>`
    : '<span>-</span>';

  const recommendationClass = `badge-${decision.recommendation.toLowerCase()}`;
  resultRecommendation.innerHTML = `<span class="badge ${recommendationClass}">${decision.recommendation}</span>`;
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
showOpsBtn.addEventListener('click', () => { showOps(); fetchMetrics(); });
opsBackBtn.addEventListener('click', () => { showQueue(); fetchQueue(); });
assessBtn.addEventListener('click', handleAssess);
acceptBtn.addEventListener('click', () => handleDecision('accept'));
showOverrideBtn.addEventListener('click', () => {
  overrideForm.classList.remove('hidden');
  decisionControls.classList.add('hidden');
});
submitOverrideBtn.addEventListener('click', () => {
  const reason = overrideReason.value.trim();
  const reasonCode = overrideReasonCode.value;
  if (!reason) {
    showAlert('A reason is required to override a recommendation');
    return;
  }
  handleDecision('override', reason, reasonCode);
});

// Start
init();
