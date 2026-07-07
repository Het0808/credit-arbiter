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
const resultRisk = document.getElementById('result-risk');
const resultClause = document.getElementById('result-clause');
const resultRegulatory = document.getElementById('result-regulatory');
const resultRecommendation = document.getElementById('result-recommendation');
const decisionControls = document.getElementById('decision-controls');
const acceptBtn = document.getElementById('accept-btn');
const showOverrideBtn = document.getElementById('show-override-btn');
const overrideForm = document.getElementById('override-form');
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

  escalationBanner.classList.toggle('hidden', !decision.escalation_flag);

  const bandClass = decision.risk_band ? `badge-${decision.risk_band.toLowerCase()}` : '';
  resultRisk.innerHTML = decision.risk_score !== null && decision.risk_score !== undefined
    ? `<span class="badge ${bandClass}">${decision.risk_band}</span>&nbsp; ${(decision.risk_score * 100).toFixed(1)}% probability of default`
    : 'No risk score available';

  if (decision.retrieved_clause_id) {
    resultClause.innerHTML = `
      <div class="clause-title">${decision.retrieved_clause_id}</div>
      <div class="clause-text">${decision.retrieved_clause_text}</div>
      <div class="clause-meta">Confidence: ${(decision.retrieval_confidence * 100).toFixed(1)}%</div>
    `;
  } else {
    resultClause.innerHTML = '<span>No policy clause retrieved (retrieval failed)</span>';
  }

  resultRegulatory.textContent = decision.regulatory_status || '-';

  const recommendationClass = `badge-${decision.recommendation.toLowerCase()}`;
  resultRecommendation.innerHTML = `<span class="badge ${recommendationClass}">${decision.recommendation}</span>`;
}

async function handleDecision(action, reason) {
  try {
    const res = await fetch(`${API_BASE}/assessments/${currentAssessmentId}/decision`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({ action, reason: reason || null })
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
  if (!reason) {
    showAlert('A reason is required to override a recommendation');
    return;
  }
  handleDecision('override', reason);
});

// Start
init();
