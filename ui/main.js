const API_BASE = 'http://localhost:8000/api';

// DOM Elements
const loginForm = document.getElementById('login-form');
const registerForm = document.getElementById('register-form');
const dashboardView = document.getElementById('dashboard-view');
const showRegisterLink = document.getElementById('show-register');
const showLoginLink = document.getElementById('show-login');
const alertBox = document.getElementById('alert-box');
const logoutBtn = document.getElementById('logout-btn');
const userInfo = document.getElementById('user-info');

// State
let token = localStorage.getItem('halcyon_token');

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
  loginForm.classList.remove('hidden');
  registerForm.classList.add('hidden');
  dashboardView.classList.add('hidden');
  hideAlert();
}

function showRegister() {
  loginForm.classList.add('hidden');
  registerForm.classList.remove('hidden');
  dashboardView.classList.add('hidden');
  hideAlert();
}

function showDashboard(email) {
  loginForm.classList.add('hidden');
  registerForm.classList.add('hidden');
  dashboardView.classList.remove('hidden');
  userInfo.textContent = `Logged in as: ${email}`;
  hideAlert();
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

// Start
init();
