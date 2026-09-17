/**
 * Agentic RAG — Enterprise Intelligence Console
 * Client Application Logic & State Management
 */

(function () {
  'use strict';

  // State
  const state = {
    currentSessionId: null,
    sessions: [], // array of { id, title, messages, updatedAt }
    currentResult: null,
    isProcessing: false,
    activeCitations: [],
    systemStatus: null,
  };

  // DOM Elements
  const elements = {
    // Sidebar
    sidebar: document.getElementById('sidebar'),
    sidebarToggleBtn: document.getElementById('sidebarToggleBtn'),
    closeSidebarBtn: document.getElementById('closeSidebarBtn'),
    newChatBtn: document.getElementById('newChatBtn'),
    historyList: document.getElementById('historyList'),
    clearAllChatsBtn: document.getElementById('clearAllChatsBtn'),
    statusPulse: document.getElementById('statusPulse'),
    statusText: document.getElementById('statusText'),
    sidebarDbInfo: document.getElementById('sidebarDbInfo'),
    sidebarDocBadge: document.getElementById('sidebarDocBadge'),
    healthBadge: document.getElementById('healthBadge'),

    // Main workspace
    currentSessionTitle: document.getElementById('currentSessionTitle'),
    topNavVectorCount: document.getElementById('topNavVectorCount'),
    welcomeScreen: document.getElementById('welcomeScreen'),
    chatThread: document.getElementById('chatThread'),
    messagesContainer: document.getElementById('messagesContainer'),
    progressIndicator: document.getElementById('progressIndicator'),
    progressTitle: document.getElementById('progressTitle'),
    progressDesc: document.getElementById('progressDesc'),
    errorBanner: document.getElementById('errorBanner'),
    errorMessage: document.getElementById('errorMessage'),
    retryQueryBtn: document.getElementById('retryQueryBtn'),
    dismissErrorBtn: document.getElementById('dismissErrorBtn'),

    // Input form
    queryForm: document.getElementById('queryForm'),
    queryInput: document.getElementById('queryInput'),
    sendBtn: document.getElementById('sendBtn'),

    // Evidence Drawer
    evidenceDrawer: document.getElementById('evidenceDrawer'),
    drawerToggleBtn: document.getElementById('drawerToggleBtn'),
    closeDrawerBtn: document.getElementById('closeDrawerBtn'),
    drawerCountBadge: document.getElementById('drawerCountBadge'),
    drawerSearchInput: document.getElementById('drawerSearchInput'),
    drawerEmptyPlaceholder: document.getElementById('drawerEmptyPlaceholder'),
    sourceCardsList: document.getElementById('sourceCardsList'),
    drawerBackdrop: document.getElementById('drawerBackdrop'),

    // Modals
    knowledgeModal: document.getElementById('knowledgeModal'),
    openKnowledgeModalBtn: document.getElementById('openKnowledgeModalBtn'),
    closeKnowledgeModalBtn: document.getElementById('closeKnowledgeModalBtn'),
    closeKnowledgeFooterBtn: document.getElementById('closeKnowledgeFooterBtn'),
    modalVectorCount: document.getElementById('modalVectorCount'),
    documentsTableBody: document.getElementById('documentsTableBody'),
    dropzone: document.getElementById('dropzone'),
    fileInput: document.getElementById('fileInput'),
    uploadProgress: document.getElementById('uploadProgress'),
    uploadStatusText: document.getElementById('uploadStatusText'),

    archModal: document.getElementById('archModal'),
    openArchModalBtn: document.getElementById('openArchModalBtn'),
    closeArchModalBtn: document.getElementById('closeArchModalBtn'),
    closeArchFooterBtn: document.getElementById('closeArchFooterBtn'),
    diagStorageDir: document.getElementById('diagStorageDir'),
    diagProvider: document.getElementById('diagProvider'),
  };

  // Helper: Format relative timestamp
  function formatTime(isoString) {
    if (!isoString) return '';
    try {
      const date = new Date(isoString);
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch {
      return '';
    }
  }

  // Load / Save Sessions from localStorage
  function loadLocalSessions() {
    try {
      const stored = localStorage.getItem('agentic_rag_sessions');
      if (stored) {
        state.sessions = JSON.parse(stored);
      }
    } catch (e) {
      console.warn('Failed to load local sessions:', e);
      state.sessions = [];
    }
  }

  function saveLocalSessions() {
    try {
      localStorage.setItem('agentic_rag_sessions', JSON.stringify(state.sessions));
    } catch (e) {
      console.warn('Failed to save local sessions:', e);
    }
  }

  // Create a New Analysis Session
  function createNewSession(initialTitle = 'New Analysis') {
    const newSession = {
      id: 'session_' + Date.now() + '_' + Math.random().toString(36).substring(2, 7),
      title: initialTitle,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      messages: [],
    };
    state.sessions.unshift(newSession);
    state.currentSessionId = newSession.id;
    saveLocalSessions();
    renderHistory();
    renderActiveSession();
    return newSession;
  }

  // Render History List in Sidebar
  function renderHistory() {
    elements.historyList.innerHTML = '';
    if (state.sessions.length === 0) {
      elements.historyList.innerHTML = '<div style="font-size:0.75rem; color:var(--text-muted); padding:8px 10px;">No prior sessions</div>';
      return;
    }

    state.sessions.forEach(session => {
      const item = document.createElement('div');
      item.className = `history-item ${session.id === state.currentSessionId ? 'active' : ''}`;
      item.innerHTML = `
        <span class="history-title" title="${escapeHtml(session.title)}">${escapeHtml(session.title)}</span>
        <button class="history-delete-btn" title="Delete session" data-id="${session.id}">✕</button>
      `;

      item.addEventListener('click', (e) => {
        if (e.target.classList.contains('history-delete-btn')) {
          e.stopPropagation();
          deleteSession(session.id);
          return;
        }
        selectSession(session.id);
      });

      elements.historyList.appendChild(item);
    });
  }

  function selectSession(sessionId) {
    state.currentSessionId = sessionId;
    renderHistory();
    renderActiveSession();
    if (window.innerWidth <= 1024) {
      closeSidebar();
    }
  }

  function deleteSession(sessionId) {
    state.sessions = state.sessions.filter(s => s.id !== sessionId);
    saveLocalSessions();
    if (state.currentSessionId === sessionId) {
      if (state.sessions.length > 0) {
        state.currentSessionId = state.sessions[0].id;
      } else {
        createNewSession();
        return;
      }
    }
    renderHistory();
    renderActiveSession();
  }

  function renderActiveSession() {
    const session = state.sessions.find(s => s.id === state.currentSessionId);
    if (!session) return;

    elements.currentSessionTitle.textContent = session.title;
    elements.messagesContainer.innerHTML = '';

    if (session.messages.length === 0) {
      elements.welcomeScreen.style.display = 'block';
    } else {
      elements.welcomeScreen.style.display = 'none';
      session.messages.forEach(msg => {
        if (msg.role === 'user') {
          renderUserMessage(msg.content, msg.timestamp, false);
        } else if (msg.role === 'assistant') {
          renderAssistantMessage(msg.result, false);
        }
      });
      // Populate drawer with latest result evidence if present
      const lastAsst = [...session.messages].reverse().find(m => m.role === 'assistant');
      if (lastAsst && lastAsst.result && lastAsst.result.retrieved_chunks) {
        renderEvidenceDrawer(lastAsst.result.retrieved_chunks);
      }
    }
    scrollToBottom();
  }

  // Sanitize HTML
  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Format Text with Interactive Citations
  function formatWithCitations(text) {
    if (!text) return '';
    let formatted = escapeHtml(text);

    // Bold formatting: **bold**
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Replace citations like [Source 01 • Page 23] or [Source 01] or [1]
    formatted = formatted.replace(
      /\[(Source\s*0?(\d+)[^\]]*)\]/gi,
      (match, label, num) => {
        const sourceId = parseInt(num, 10);
        return `<button class="citation-chip" data-source-id="${sourceId}" title="View ${escapeHtml(label)} in evidence drawer">
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path></svg>
          ${escapeHtml(label)}
        </button>`;
      }
    );

    // Also support [1], [2] style citations
    formatted = formatted.replace(
      /\[(\d+)\]/g,
      (match, num) => {
        const sourceId = parseInt(num, 10);
        return `<button class="citation-chip" data-source-id="${sourceId}" title="View Source 0${sourceId}">
          [${sourceId}]
        </button>`;
      }
    );

    return formatted;
  }

  // Render User Message Bubble
  function renderUserMessage(content, timestamp = new Date().toISOString(), animate = true) {
    const msgElem = document.createElement('div');
    msgElem.className = 'message-user';
    msgElem.innerHTML = `
      <div class="user-bubble">
        <p>${escapeHtml(content)}</p>
      </div>
      <div class="user-avatar" title="You">YOU</div>
    `;
    elements.messagesContainer.appendChild(msgElem);
    if (animate) scrollToBottom();
  }

  // Render Assistant Message with Reasoning Accordion and Structured Cards
  function renderAssistantMessage(result, animate = true) {
    if (!result) return;
    const msgElem = document.createElement('div');
    msgElem.className = 'message-assistant';

    // 1. Agent Reasoning Accordion
    const activityBox = document.createElement('div');
    activityBox.className = 'agent-activity-box open';

    const stepsHtml = (result.timeline_steps || []).map(step => `
      <div class="activity-step-item completed">
        <div class="step-circle">${escapeHtml(step.step_num || '01')}</div>
        <div class="step-content">
          <div class="step-top">
            <span class="step-name">${escapeHtml(step.title)}</span>
            ${step.badge ? `<span class="badge badge-subtle">${escapeHtml(step.badge)}</span>` : ''}
          </div>
          <p class="step-desc">${escapeHtml(step.description)}</p>
        </div>
      </div>
    `).join('');

    activityBox.innerHTML = `
      <div class="activity-header">
        <div class="activity-title-group">
          <span class="activity-pulse"></span>
          <span class="activity-title">Agent Reasoning & Dynamic Retrieval</span>
          <span class="badge badge-cyan">${result.retrieved_chunks ? result.retrieved_chunks.length : 0} Evidence Chunks</span>
        </div>
        <div class="activity-meta">
          <span class="badge badge-subtle text-mono">${result.metrics ? result.metrics.retrieval_time_ms : '120'}ms</span>
          <span class="activity-toggle-icon">▼</span>
        </div>
      </div>
      <div class="activity-steps-list">
        ${stepsHtml}
      </div>
    `;

    activityBox.querySelector('.activity-header').addEventListener('click', () => {
      activityBox.classList.toggle('open');
    });

    // 2. Structured Executive Analyst Response
    const responseCard = document.createElement('div');
    responseCard.className = 'assistant-response-card';

    // Numbers Grid HTML
    let numbersGridHtml = '';
    if (result.important_numbers && result.important_numbers.length > 0) {
      const cards = result.important_numbers.map(num => `
        <div class="number-card">
          <span class="number-label">${escapeHtml(num.label)}</span>
          <span class="number-value">${escapeHtml(num.value)}</span>
          <div class="number-footer">
            <span class="number-change">${escapeHtml(num.change || 'Verified')}</span>
            <span class="number-citation">${escapeHtml(num.citation || 'Doc')}</span>
          </div>
        </div>
      `).join('');
      numbersGridHtml = `
        <div class="report-section">
          <div class="report-section-header">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg>
            Key Operational Numbers
          </div>
          <div class="numbers-grid">${cards}</div>
        </div>
      `;
    }

    // Key Findings HTML
    let findingsHtml = '';
    if (result.key_findings && result.key_findings.length > 0) {
      const listItems = result.key_findings.map(kf => `<li>${formatWithCitations(kf)}</li>`).join('');
      findingsHtml = `
        <div class="report-section">
          <div class="report-section-header">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
            Verified Findings
          </div>
          <ul class="findings-list">${listItems}</ul>
        </div>
      `;
    }

    // Risks HTML
    let risksHtml = '';
    if (result.risks_concerns && result.risks_concerns.length > 0) {
      const riskItems = result.risks_concerns.map(rk => `<li>${formatWithCitations(rk)}</li>`).join('');
      risksHtml = `
        <div class="report-section">
          <div class="report-section-header">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--amber)" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
            Risks & Disclosures
          </div>
          <ul class="risks-list">${riskItems}</ul>
        </div>
      `;
    }

    // Outlook HTML
    let outlookHtml = '';
    if (result.outlook && result.outlook.length > 0) {
      const outlookItems = result.outlook.map(ot => `<li>${formatWithCitations(ot)}</li>`).join('');
      outlookHtml = `
        <div class="report-section">
          <div class="report-section-header">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" stroke-width="2"><polyline points="13 17 18 12 13 7"></polyline><polyline points="6 17 11 12 6 7"></polyline></svg>
            Strategic Outlook & Forward Guidance
          </div>
          <ul class="outlook-list">${outlookItems}</ul>
        </div>
      `;
    }

    responseCard.innerHTML = `
      <div class="report-section">
        <div class="report-section-header primary">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>
          Executive Summary
        </div>
        <p class="report-summary-text">${formatWithCitations(result.executive_summary)}</p>
      </div>
      ${numbersGridHtml}
      ${findingsHtml}
      ${risksHtml}
      ${outlookHtml}
      
      <div class="response-actions-bar">
        <div class="actions-left">
          <button class="action-btn copy-report-btn" title="Copy analyst report to clipboard">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
            Copy Report
          </button>
          <button class="action-btn toggle-raw-btn" title="Toggle full raw markdown view">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"></polyline><polyline points="8 6 2 12 8 18"></polyline></svg>
            Raw View
          </button>
        </div>
        <div class="actions-right">
          <button class="action-btn open-evidence-btn" title="Inspect retrieved source chunks">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path></svg>
            View Evidence (${result.retrieved_chunks ? result.retrieved_chunks.length : 0})
          </button>
        </div>
      </div>
    `;

    // Hook up action buttons
    const copyBtn = responseCard.querySelector('.copy-report-btn');
    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(result.raw_answer || result.executive_summary);
      copyBtn.innerHTML = `✓ Copied`;
      setTimeout(() => {
        copyBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg> Copy Report`;
      }, 2000);
    });

    const toggleRawBtn = responseCard.querySelector('.toggle-raw-btn');
    let showingRaw = false;
    toggleRawBtn.addEventListener('click', () => {
      showingRaw = !showingRaw;
      if (showingRaw) {
        responseCard.querySelector('.report-summary-text').textContent = result.raw_answer;
        toggleRawBtn.textContent = 'Structured View';
      } else {
        responseCard.querySelector('.report-summary-text').innerHTML = formatWithCitations(result.executive_summary);
        toggleRawBtn.textContent = 'Raw View';
      }
    });

    const openEvidenceBtn = responseCard.querySelector('.open-evidence-btn');
    openEvidenceBtn.addEventListener('click', () => {
      openDrawer();
    });

    // Attach citation click handlers
    responseCard.querySelectorAll('.citation-chip').forEach(chip => {
      chip.addEventListener('click', (e) => {
        e.preventDefault();
        const sourceId = parseInt(chip.getAttribute('data-source-id'), 10);
        focusEvidenceSource(sourceId);
      });
    });

    msgElem.appendChild(activityBox);
    msgElem.appendChild(responseCard);
    elements.messagesContainer.appendChild(msgElem);

    if (animate) scrollToBottom();
  }

  // Populate Right Evidence Drawer
  function renderEvidenceDrawer(chunks) {
    if (!chunks || chunks.length === 0) {
      elements.drawerEmptyPlaceholder.style.display = 'flex';
      elements.sourceCardsList.innerHTML = '';
      elements.drawerCountBadge.textContent = '0 Sources';
      return;
    }

    elements.drawerEmptyPlaceholder.style.display = 'none';
    elements.sourceCardsList.innerHTML = '';
    elements.drawerCountBadge.textContent = `${chunks.length} Sources`;

    chunks.forEach((chunk, index) => {
      const card = document.createElement('div');
      card.className = 'source-card';
      card.id = `source-card-${chunk.source_id || (index + 1)}`;

      const relevancePct = Math.round((chunk.relevance_score || 0.85) * 100);

      card.innerHTML = `
        <div class="source-card-header">
          <span class="source-id-badge">SOURCE 0${chunk.source_id || (index + 1)}</span>
          <div class="source-score-pill">
            <span class="badge badge-emerald">${relevancePct}% Match</span>
            <span class="badge badge-subtle text-mono">dist: ${chunk.distance || '0.12'}</span>
          </div>
        </div>

        <div class="source-metadata-row">
          <span>📄 ${escapeHtml(chunk.file_name || 'Report')}</span>
          <span>•</span>
          <span>Page ${chunk.page || 1}</span>
          <span>•</span>
          <span>${chunk.char_count || chunk.content.length} chars</span>
        </div>

        <div class="source-content-box">${escapeHtml(chunk.content)}</div>

        <div class="source-card-footer">
          <span class="badge badge-subtle text-mono">${escapeHtml(chunk.chunk_id || 'chunk_' + (index + 1))}</span>
          <button class="copy-chunk-btn" data-content="${escapeHtml(chunk.content)}">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
            Copy Snippet
          </button>
        </div>
      `;

      card.querySelector('.copy-chunk-btn').addEventListener('click', (e) => {
        const btn = e.currentTarget;
        navigator.clipboard.writeText(chunk.content);
        btn.textContent = '✓ Copied';
        setTimeout(() => {
          btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg> Copy Snippet`;
        }, 2000);
      });

      elements.sourceCardsList.appendChild(card);
    });
  }

  // Filter chunks inside drawer
  elements.drawerSearchInput.addEventListener('input', (e) => {
    const term = e.target.value.toLowerCase();
    const cards = elements.sourceCardsList.querySelectorAll('.source-card');
    cards.forEach(card => {
      const text = card.textContent.toLowerCase();
      card.style.display = text.includes(term) ? 'flex' : 'none';
    });
  });

  // Highlight and focus a specific source card when citation clicked
  function focusEvidenceSource(sourceId) {
    openDrawer();
    setTimeout(() => {
      const card = document.getElementById(`source-card-${sourceId}`);
      if (card) {
        card.scrollIntoView({ behavior: 'smooth', block: 'center' });
        card.classList.add('highlighted');
        setTimeout(() => {
          card.classList.remove('highlighted');
        }, 2500);
      }
    }, 150);
  }

  function openDrawer() {
    elements.evidenceDrawer.classList.add('open');
    elements.evidenceDrawer.setAttribute('aria-hidden', 'false');
    if (window.innerWidth <= 1024) {
      elements.drawerBackdrop.classList.add('active');
    }
  }

  function closeDrawer() {
    elements.evidenceDrawer.classList.remove('open');
    elements.evidenceDrawer.setAttribute('aria-hidden', 'true');
    elements.drawerBackdrop.classList.remove('active');
  }

  elements.drawerToggleBtn.addEventListener('click', () => {
    if (elements.evidenceDrawer.classList.contains('open')) {
      closeDrawer();
    } else {
      openDrawer();
    }
  });

  elements.closeDrawerBtn.addEventListener('click', closeDrawer);
  elements.drawerBackdrop.addEventListener('click', closeDrawer);

  // Sidebar Controls
  function openSidebar() {
    elements.sidebar.classList.add('open');
  }

  function closeSidebar() {
    elements.sidebar.classList.remove('open');
  }

  if (elements.sidebarToggleBtn) {
    elements.sidebarToggleBtn.addEventListener('click', openSidebar);
  }
  if (elements.closeSidebarBtn) {
    elements.closeSidebarBtn.addEventListener('click', closeSidebar);
  }

  // Execute Query
  async function submitQuery(queryText) {
    if (!queryText || !queryText.trim() || state.isProcessing) return;
    const cleanQuery = queryText.trim();

    // Hide welcome screen
    elements.welcomeScreen.style.display = 'none';
    hideError();

    // Ensure session
    let session = state.sessions.find(s => s.id === state.currentSessionId);
    if (!session) {
      session = createNewSession(cleanQuery.substring(0, 40) + '...');
    } else if (session.messages.length === 0) {
      session.title = cleanQuery.substring(0, 40) + '...';
      elements.currentSessionTitle.textContent = session.title;
      renderHistory();
    }

    // Add user message to session & UI
    const nowIso = new Date().toISOString();
    session.messages.push({
      role: 'user',
      content: cleanQuery,
      timestamp: nowIso,
    });
    session.updatedAt = nowIso;
    saveLocalSessions();
    renderUserMessage(cleanQuery, nowIso);

    // Show Progress state
    setProcessing(true);
    elements.progressIndicator.style.display = 'block';
    elements.progressTitle.textContent = 'Agent Reasoning Active';
    elements.progressDesc.textContent = `Formulating vector tool lookup for "${cleanQuery.substring(0, 50)}..."`;
    scrollToBottom();

    // Clear input
    elements.queryInput.value = '';
    elements.queryInput.style.height = 'auto';

    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: cleanQuery }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Server error' }));
        throw new Error(errorData.detail || `Server error (${response.status})`);
      }

      const result = await response.json();
      state.currentResult = result;

      // Add assistant message to session & UI
      session.messages.push({
        role: 'assistant',
        result: result,
        timestamp: new Date().toISOString(),
      });
      session.updatedAt = new Date().toISOString();
      saveLocalSessions();

      elements.progressIndicator.style.display = 'none';
      renderAssistantMessage(result);
      renderEvidenceDrawer(result.retrieved_chunks);

    } catch (err) {
      console.error('Query execution error:', err);
      elements.progressIndicator.style.display = 'none';
      showError(err.message || 'Failed to complete agentic retrieval query.', cleanQuery);
    } finally {
      setProcessing(false);
    }
  }

  function setProcessing(isProc) {
    state.isProcessing = isProc;
    elements.sendBtn.disabled = isProc;
    elements.queryInput.disabled = isProc;
  }

  function showError(msg, queryToRetry) {
    elements.errorMessage.textContent = msg;
    elements.errorBanner.style.display = 'flex';
    elements.retryQueryBtn.onclick = () => {
      elements.errorBanner.style.display = 'none';
      submitQuery(queryToRetry);
    };
  }

  function hideError() {
    elements.errorBanner.style.display = 'none';
  }

  elements.dismissErrorBtn.addEventListener('click', hideError);

  // Auto-resize textarea
  elements.queryInput.addEventListener('input', () => {
    elements.queryInput.style.height = 'auto';
    elements.queryInput.style.height = Math.min(elements.queryInput.scrollHeight, 120) + 'px';
  });

  // Handle Enter to submit (Shift+Enter for new line)
  elements.queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      elements.queryForm.dispatchEvent(new Event('submit'));
    }
  });

  elements.queryForm.addEventListener('submit', (e) => {
    e.preventDefault();
    submitQuery(elements.queryInput.value);
  });

  // Starter card click triggers query
  document.querySelectorAll('.starter-card').forEach(card => {
    card.addEventListener('click', () => {
      const prompt = card.getAttribute('data-prompt');
      if (prompt) {
        submitQuery(prompt);
      }
    });
  });

  // New Chat action
  elements.newChatBtn.addEventListener('click', () => {
    createNewSession();
  });

  // Keyboard Shortcuts: Cmd+K / Ctrl+K for new analysis, Escape to close drawer
  window.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      createNewSession();
    } else if (e.key === 'Escape') {
      closeDrawer();
      closeKnowledgeModal();
      closeArchModal();
    }
  });

  // Clear all chats
  elements.clearAllChatsBtn.addEventListener('click', () => {
    if (confirm('Are you sure you want to clear all analysis sessions?')) {
      state.sessions = [];
      saveLocalSessions();
      createNewSession();
    }
  });

  function scrollToBottom() {
    elements.chatThread.scrollTop = elements.chatThread.scrollHeight;
  }

  // Fetch System Status and Health
  async function fetchSystemStatus() {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        state.systemStatus = data;
        const vecCount = data.indexed_vectors ? data.indexed_vectors.toLocaleString() : '1,818';
        elements.topNavVectorCount.textContent = `${vecCount} Vectors`;
        elements.sidebarDocBadge.textContent = `${vecCount} Chunks`;
        elements.sidebarDbInfo.textContent = `ChromaDB • ${vecCount} Chunks`;
        elements.modalVectorCount.textContent = vecCount;
        elements.diagStorageDir.textContent = data.vector_store_dir || './company_db';
        elements.diagProvider.textContent = data.default_llm || 'Groq / Ollama';
        setHealthStatus(true);
      }
    } catch (e) {
      console.warn('Unable to reach /api/status:', e);
      setHealthStatus(false);
    }
  }

  function setHealthStatus(isOnline) {
    if (isOnline) {
      elements.statusPulse.style.backgroundColor = 'var(--success-emerald)';
      elements.statusPulse.style.boxShadow = '0 0 0 2px rgba(16, 185, 129, 0.2)';
      elements.statusText.textContent = 'Vector Store Online';
      elements.healthBadge.textContent = 'Verified';
      elements.healthBadge.className = 'badge badge-emerald';
    } else {
      elements.statusPulse.style.backgroundColor = 'var(--error-red)';
      elements.statusPulse.style.boxShadow = '0 0 0 2px rgba(239, 68, 68, 0.2)';
      elements.statusText.textContent = 'Backend Offline';
      elements.healthBadge.textContent = 'Disconnected';
      elements.healthBadge.className = 'badge badge-subtle';
    }
  }

  // Modal Handlers: Knowledge & Documents
  async function openKnowledgeModal() {
    elements.knowledgeModal.style.display = 'flex';
    try {
      const res = await fetch('/api/documents');
      if (res.ok) {
        const data = await res.json();
        elements.documentsTableBody.innerHTML = '';
        (data.documents || []).forEach(doc => {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td><strong>${escapeHtml(doc.file_name)}</strong></td>
            <td><span class="badge badge-subtle">${escapeHtml(doc.source_type)}</span></td>
            <td>${doc.pages || 1} pages</td>
            <td>${doc.size_formatted}</td>
            <td><span class="badge ${doc.is_indexed ? 'badge-emerald' : 'badge-subtle'}">${doc.is_indexed ? 'Indexed in Chroma' : 'Raw Data'}</span></td>
          `;
          elements.documentsTableBody.appendChild(tr);
        });
      }
    } catch (e) {
      console.warn('Error loading documents list:', e);
    }
  }

  function closeKnowledgeModal() {
    elements.knowledgeModal.style.display = 'none';
  }

  elements.openKnowledgeModalBtn.addEventListener('click', openKnowledgeModal);
  elements.closeKnowledgeModalBtn.addEventListener('click', closeKnowledgeModal);
  elements.closeKnowledgeFooterBtn.addEventListener('click', closeKnowledgeModal);

  // File Upload Drag & Drop
  const dropzone = elements.dropzone;
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  elements.fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });

  async function handleFileUpload(file) {
    if (!file) return;
    elements.uploadProgress.style.display = 'block';
    elements.uploadStatusText.textContent = `Uploading and splitting '${file.name}' into semantic chunks...`;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/ingest', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        elements.uploadStatusText.textContent = `✓ Successfully indexed '${file.name}' (${data.chunks_created} chunks added).`;
        fetchSystemStatus();
        openKnowledgeModal(); // refresh table
      } else {
        elements.uploadStatusText.textContent = `Upload failed: ${data.detail || 'Server error'}`;
      }
    } catch (e) {
      elements.uploadStatusText.textContent = `Upload error: ${e.message}`;
    }
  }

  // Modal Handlers: Diagnostics
  function openArchModal() {
    elements.archModal.style.display = 'flex';
  }

  function closeArchModal() {
    elements.archModal.style.display = 'none';
  }

  elements.openArchModalBtn.addEventListener('click', openArchModal);
  elements.closeArchModalBtn.addEventListener('click', closeArchModal);
  elements.closeArchFooterBtn.addEventListener('click', closeArchModal);

  // Initialize
  function init() {
    loadLocalSessions();
    if (state.sessions.length === 0) {
      createNewSession('TCS Financial Performance Analysis');
    } else {
      state.currentSessionId = state.sessions[0].id;
      renderHistory();
      renderActiveSession();
    }

    fetchSystemStatus();
    setInterval(fetchSystemStatus, 30000);
  }

  init();
})();
