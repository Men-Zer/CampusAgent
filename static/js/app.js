/* ============================================================
   AgenticRAG — Application Logic
   Zero-dependency | SSE streaming | Theme toggle | Tool calls
   ============================================================ */

// ---------- Config ----------
const API_BASE = 'http://127.0.0.1:8000';
const THEME_KEY = 'it-desk-theme';
const SESSION_KEY = 'it-desk-sessions';
const ACTIVE_SESSION_KEY = 'it-desk-active-session';
const STREAM_KEY = 'it-desk-stream';
const FONT_SIZE_KEY = 'hc-font-size';
const USER_ID_KEY = 'hb-user-id';  // 持久化 UUID

// 用户 UUID：首次自动生成，存 localStorage，永不改变
function getUserId() {
  let uid = localStorage.getItem(USER_ID_KEY);
  if (!uid) {
    uid = crypto.randomUUID ? crypto.randomUUID() : 'u_' + Date.now().toString(36);
    localStorage.setItem(USER_ID_KEY, uid);
  }
  return uid;
}

// ---------- File Upload ----------
let _uploading = false;
async function handleFileUpload() {
  if (_uploading) return;
  const files = dom.fileInput.files;
  if (!files.length) return;

  _uploading = true;
  const uid = getUserId();
  for (const file of files) {
    const form = new FormData();
    form.append('file', file);
    form.append('user_id', uid);

    try {
      const res = await fetch('/upload', { method: 'POST', body: form });
      const data = await res.json();
      const text = data.ok
        ? `已上传「${file.name}」到你的知识库`
        : `上传失败：${data.error}`;
      addUploadMessage(text);
    } catch (e) {
      addUploadMessage(`上传「${file.name}」出错：${e.message}`);
    }
  }
  dom.fileInput.value = '';
  _uploading = false;
}

function addUploadMessage(text) {
  // 加入当前会话的消息数组，切换历史不会丢也不会串
  const sid = state.currentSessionId;
  if (!sid || !state.conversations.has(sid)) return;
  const conv = state.conversations.get(sid);
  conv.messages.push({ role: 'system', content: text, timestamp: Date.now() });
  saveSessions();
  // 渲染
  const el = document.createElement('div');
  el.className = 'message system';
  el.innerHTML = `<div class="message-body">...${escapeHtml(text)}...</div>`;
  dom.thread.appendChild(el);
  el.scrollIntoView({ behavior: 'smooth' });
}

// ---------- State ----------
const state = {
  sidebarOpen: false,
  currentSessionId: null,         // restored from localStorage in init()
  conversations: new Map(),       // sessionId → { title, messages[], createdAt }
  isStreaming: false,
  streamEnabled: localStorage.getItem(STREAM_KEY) !== 'false',  // 默认开启，仅当明确存了 'false' 才关闭
  abortController: null,
  currentStreamingMsg: null,  // Bug #5: 当前正在流式的 assistantMsg 引用，loadSessionIntoThread 用它判断要不要强制 streaming=false
  theme: localStorage.getItem(THEME_KEY) || 'dark',
  panelOpen: true,
};

// ---------- DOM refs ----------
const $ = (sel) => document.querySelector(sel);
const dom = {
  app:             $('#app'),
  thread:          $('#thread'),
  emptyState:      $('#emptyState'),
  quickPrompts:    $('#quickPrompts'),
  conversationList:$('#conversationList'),
  composerInput:   $('#composerInput'),
  btnSend:         $('#btnSend'),
  btnNewChat:      $('#btnNewChat'),
  btnTheme:        $('#btnTheme'),
  btnTogglePanel:  $('#btnTogglePanel'),
  btnStreamToggle: $('#btnStreamToggle'),
  btnClosePanel:   $('#btnClosePanel'),
  panel:           $('#panel'),
  panelContent:    $('#panelContent'),
  panelEmpty:      $('#panelEmpty'),
  connectionStatus:$('#connectionStatus'),
  connectionLabel: $('#connectionLabel'),
  modelBadge:      $('#modelBadge'),
  composerHint:    $('#composerHint'),
  btnUpload:       $('#btnUpload'),
  fileInput:       $('#fileInput'),
};

// ---------- Font Size ----------
function setFontSize(size) {
  document.body.classList.remove('font-small', 'font-medium', 'font-large');
  document.body.classList.add(`font-${size}`);
  document.querySelectorAll('.font-size-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.size === size);
  });
  localStorage.setItem(FONT_SIZE_KEY, size);
}

// ---------- Init ----------
function init() {
  applyTheme(state.theme);
  // Restore font size preference
  const savedSize = localStorage.getItem(FONT_SIZE_KEY) || 'medium';
  setFontSize(savedSize);
  loadSessions();
  restoreActiveSession();
  renderConversationList();
  bindEvents();
  checkConnection();
  setInterval(checkConnection, 30000);
  // 每次 currentSessionId 变化时立刻存到 localStorage
  // 防止刷新页面前还没发消息导致 session 丢失
}

// ---------- Theme ----------
function applyTheme(theme) {
  state.theme = theme;
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem(THEME_KEY, theme);
  // 主题切换后重新初始化 + 重新渲染所有 mermaid 图（handDrawn 主题需要重置）
  if (window.reinitMermaid) {
    window.reinitMermaid(theme);
    // 清除已渲染标记，恢复原始代码，触发重新渲染
    document.querySelectorAll('.mermaid-rendered').forEach(el => {
      el.classList.remove('mermaid-rendered');
      el.removeAttribute('data-processed');
      const src = el.getAttribute('data-source');
      if (src) { try { el.textContent = decodeURIComponent(src); } catch (_) { el.textContent = src; } }
    });
    renderMermaidBlocks(document.getElementById('thread'));
  }

  const darkIcon = dom.btnTheme.querySelector('.theme-icon-dark');
  const lightIcon = dom.btnTheme.querySelector('.theme-icon-light');
  if (theme === 'light') {
    darkIcon.style.display = 'none';
    lightIcon.style.display = '';
  } else {
    darkIcon.style.display = '';
    lightIcon.style.display = 'none';
  }
}

function toggleTheme() {
  applyTheme(state.theme === 'dark' ? 'light' : 'dark');
}

// ---------- Sessions ----------
function loadSessions() {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    if (raw) {
      const data = JSON.parse(raw);
      state.conversations = new Map(Object.entries(data));
    }
  } catch (_) {}
}

function saveSessions() {
  try {
    const obj = Object.fromEntries(state.conversations);
    localStorage.setItem(SESSION_KEY, JSON.stringify(obj));
  } catch (e) {
    console.warn('[storage] 保存会话失败（可能超出 localStorage 限制）:', e);
  }
}

function saveActiveSession() {
  localStorage.setItem(ACTIVE_SESSION_KEY, state.currentSessionId);
}

function restoreActiveSession() {
  const saved = localStorage.getItem(ACTIVE_SESSION_KEY);
  if (saved && state.conversations.has(saved)) {
    state.currentSessionId = saved;
    loadSessionIntoThread();
  } else {
    state.currentSessionId = crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(36);
    state.conversations.set(state.currentSessionId, {
      title: '新对话',
      messages: [],
      createdAt: Date.now(),
    });
    saveSessions();
    saveActiveSession();  // P1-12: 持久化活跃会话 ID
  }
}

function getOrCreateSession() {
  if (!state.conversations.has(state.currentSessionId)) {
    state.conversations.set(state.currentSessionId, {
      title: '新对话',
      messages: [],
      createdAt: Date.now(),
    });
  }
  return state.conversations.get(state.currentSessionId);
}

function newSession() {
  if (state.isStreaming) return;
  state.currentSessionId = crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(36);
  state.conversations.set(state.currentSessionId, {
    title: '新对话',
    messages: [],
    createdAt: Date.now(),
  });
  saveSessions();
  saveActiveSession();
  renderConversationList();
  clearThread();
  dom.emptyState.style.display = '';
  dom.panelContent.innerHTML = '';
  dom.panelEmpty.style.display = '';
}

// ---------- Conversation List ----------
function renderConversationList() {
  const list = dom.conversationList;
  list.innerHTML = '';

  const sorted = [...state.conversations.entries()]
    .sort(([, a], [, b]) => b.createdAt - a.createdAt)
    .slice(0, 50);

  for (const [sid, conv] of sorted) {
    const item = document.createElement('div');
    item.className = 'conversation-item' + (sid === state.currentSessionId ? ' active' : '');
    item.setAttribute('role', 'option');
    item.setAttribute('aria-selected', sid === state.currentSessionId ? 'true' : 'false');
    item.dataset.sessionId = sid;

    const icon = document.createElement('span');
    icon.className = 'conv-icon';
    icon.textContent = '💬';

    const title = document.createElement('span');
    title.className = 'conv-title';
    title.textContent = conv.title || '新对话';

    const time = document.createElement('span');
    time.className = 'conv-time';
    time.textContent = formatTime(conv.createdAt);

    item.append(icon, title, time);
    item.addEventListener('click', () => switchSession(sid));
    list.appendChild(item);
  }

  // 同步到移动端 overlay 列表
  const mobileList = $('#conversationMobileList');
  if (mobileList) {
    mobileList.innerHTML = list.innerHTML;
    mobileList.querySelectorAll('.conversation-item').forEach(el => {
      el.addEventListener('click', () => {
        switchSession(el.dataset.sessionId);
        // 关闭 overlay
        const ov = $('#sidebarMobileOverlay');
        if (ov) { ov.classList.remove('open'); setTimeout(() => { if (!ov.classList.contains('open')) ov.style.display = 'none'; }, 300); }
      });
    });
  }
}

function switchSession(sid) {
  if (state.isStreaming) { stopGeneration(); }
  if (sid === state.currentSessionId) return;
  state.currentSessionId = sid;
  saveActiveSession();
  renderConversationList();
  loadSessionIntoThread();
}

function loadSessionIntoThread() {
  clearThread();
  const conv = getOrCreateSession();

  if (conv.messages.length === 0) {
    dom.emptyState.style.display = '';
    dom.panelContent.innerHTML = '';
    dom.panelEmpty.style.display = '';
    return;
  }

  dom.emptyState.style.display = 'none';
  for (const msg of conv.messages) {
    // Bug #5 修复：只有"不是当前正在流式的消息"才强制设为 completed。
    if (msg.streaming && msg !== state.currentStreamingMsg) {
      msg.streaming = false;
    }
    dom.thread.appendChild(renderMessage(msg));
  }
  // 恢复历史会话后，渲染所有 assistant 消息里的 mermaid 块
  renderMermaidBlocks(dom.thread);
  scrollToBottom(true);
}

function clearThread() {
  // 关键：innerHTML='' 会移除 emptyState 和它的子元素 quickPrompts
  // 然后 appendChild(emptyState) 会把整个子树挂回去
  // 不要单独再 appendChild(quickPrompts)，否则会把 quickPrompts 从 emptyState 内部移走
  // 变成兄弟元素，empty-state 的 flex 布局就失效了
  dom.thread.innerHTML = '';
  dom.thread.appendChild(dom.emptyState);
}

// ---------- Connection Check ----------
async function checkConnection() {
  if (document.visibilityState === 'hidden') return;  // P2-48: 页面不可见时跳过
  try {
    const resp = await fetch(API_BASE + '/api/health', { signal: AbortSignal.timeout(3000) });
    if (resp.ok) {
      dom.connectionStatus.style.background = 'var(--success)';
      dom.connectionLabel.textContent = '已连接';
    } else {
      throw new Error();
    }
  } catch {
    dom.connectionStatus.style.background = 'var(--error)';
    dom.connectionLabel.textContent = '未连接';
  }
}

// ---------- Event Binding ----------
function bindEvents() {
  // Theme toggle
  dom.btnTheme.addEventListener('click', toggleTheme);

  // Font size control
  document.querySelectorAll('.font-size-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const size = btn.dataset.size;
      setFontSize(size);
    });
  });

  // New chat
  dom.btnNewChat.addEventListener('click', newSession);

  // Stream toggle
  dom.btnStreamToggle.addEventListener('click', () => {
    if (state.isStreaming) return;  // 流式进行中不允许切换
    state.streamEnabled = !state.streamEnabled;
    localStorage.setItem(STREAM_KEY, state.streamEnabled);
    updateStreamToggleUI();
  });
  updateStreamToggleUI();

  // Composer
  dom.composerInput.addEventListener('input', onComposerInput);
  dom.composerInput.addEventListener('keydown', onComposerKeydown);
  // 发送按钮在 streaming 时变为停止按钮
  dom.btnSend.addEventListener('click', () => {
    if (state.isStreaming) {
      stopGeneration();
    } else {
      sendMessage();
    }
  });

  // 页面刷新/关闭前中断进行中的请求，避免 TypeError: Failed to fetch
  window.addEventListener('beforeunload', () => {
    if (state.abortController) {
      state.abortController.abort();
    }
  });

  // File upload
  dom.btnUpload.addEventListener('click', () => dom.fileInput.click());
  dom.fileInput.addEventListener('change', handleFileUpload);

  // Panel toggle
  dom.btnTogglePanel.addEventListener('click', () => togglePanel());
  dom.btnClosePanel.addEventListener('click', () => togglePanel(false));

  // Mobile new chat button
  const btnNewChatMobile = $('#btnNewChatMobile');
  if (btnNewChatMobile) {
    btnNewChatMobile.addEventListener('click', () => {
      newSession();
      const ov = $('#sidebarMobileOverlay');
      if (ov) { ov.classList.remove('open'); setTimeout(() => { if (!ov.classList.contains('open')) ov.style.display = 'none'; }, 300); }
    });
  }

  // Mobile close button
  const btnCloseMobile = document.querySelector('.btn-close-mobile');
  if (btnCloseMobile) {
    btnCloseMobile.addEventListener('click', () => {
      const ov = $('#sidebarMobileOverlay');
      if (ov) { ov.classList.remove('open'); setTimeout(() => { if (!ov.classList.contains('open')) ov.style.display = 'none'; }, 300); }
    });
  }

  // Quick prompts
  dom.quickPrompts.addEventListener('click', (e) => {
    const btn = e.target.closest('.quick-prompt');
    if (btn) {
      dom.composerInput.value = btn.dataset.prompt;
      sendMessage();
    }
  });

  // Click handler for thread (event delegation)
  dom.thread.addEventListener('click', (e) => {
    const header = e.target.closest('.tool-call-header');
    if (header) {
      const block = header.parentElement;
      const body = block.querySelector('.tool-call-body');
      header.classList.toggle('expanded');
      body.classList.toggle('open');
    }

    const copyBtn = e.target.closest('.btn-copy');
    if (copyBtn) {
      const codeEl = copyBtn.closest('.code-block-wrapper').querySelector('.code-block');
      navigator.clipboard.writeText(codeEl.textContent).then(() => {
        copyBtn.classList.add('copied');
        copyBtn.textContent = '✓ Copied';
        setTimeout(() => {
          copyBtn.classList.remove('copied');
          copyBtn.textContent = 'Copy';
        }, 2000);
      });
    }

    // Copy message
    const copyMsg = e.target.closest('.btn-copy-msg');
    if (copyMsg) {
      const msgEl = copyMsg.closest('.message');
      const text = msgEl.querySelector('.message-content').textContent;
      navigator.clipboard.writeText(text);
      copyMsg.textContent = '✓';
      setTimeout(() => { copyMsg.textContent = 'Copy'; }, 1500);
    }

    const citation = e.target.closest('.citation-tag');
    if (citation) {
      const idx = parseInt(citation.dataset.index);
      highlightSourceInPanel(idx);
    }
  });
}

// ---------- Composer ----------
function onComposerInput() {
  const val = dom.composerInput.value.trim();
  // streaming 状态下按钮由 setStreamingUI 控制，不禁用
  if (!state.isStreaming) {
    dom.btnSend.disabled = !val;
  }
  // Auto-resize
  dom.composerInput.style.height = 'auto';
  dom.composerInput.style.height = Math.min(dom.composerInput.scrollHeight, 160) + 'px';
}

function onComposerKeydown(e) {
  // Enter 直接发送，Shift+Enter 或 Ctrl+Enter 换行
  if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.metaKey && !e.isComposing) {
    e.preventDefault();
    sendMessage();
  }
}

// ---------- Send Message ----------

// P1-22: 重试发送上一条消息（而非读输入框）
function retryLastMessage() {
  const last = state.lastUserMessage || '';
  if (last) {
    composerInput.value = last;
    sendMessage();
  }
}
// P1-11: 复制错误信息（安全方式）
function copyErrorInfo() {
  const el = document.querySelector('.error-banner .error-message');
  if (!el) return;
  const text = el.textContent || '';
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).catch(() => {});
  } else {
    // P2-43: fallback for non-HTTPS
    const ta = document.createElement('textarea');
    ta.value = text; document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); } catch (_) {}
    document.body.removeChild(ta);
  }
}

async function sendMessage() {
  const content = dom.composerInput.value.trim();
  if (state.isStreaming) {
    // P1-25: 流式期间给视觉反馈而非静默
    composerInput.placeholder = '正在生成中，请稍候...';
    setTimeout(() => { composerInput.placeholder = '输入消息，Enter 发送'; }, 1500);
    return;
  }
  if (!content) return;

  dom.composerInput.value = '';
  dom.composerInput.style.height = 'auto';
  setStreamingUI(true);

  dom.emptyState.style.display = 'none';

  const conv = getOrCreateSession();

  // Add user message
  const userMsg = { role: 'user', content, timestamp: Date.now() };
  conv.messages.push(userMsg);
  dom.thread.appendChild(renderMessage(userMsg));

  // Auto-title
  if (conv.title === '新对话' && conv.messages.length <= 2) {
    conv.title = content.slice(0, 30) + (content.length > 30 ? '...' : '');
    renderConversationList();
  }

  // Add placeholder for assistant
  const assistantMsg = { role: 'assistant', content: '', toolCalls: [], sources: [], timestamp: Date.now(), streaming: true };
  conv.messages.push(assistantMsg);
  const assistantEl = renderMessage(assistantMsg);
  dom.thread.appendChild(assistantEl);

  scrollToBottom(true);
  saveSessions();

  try {
    state.isStreaming = true;
    state.abortController = new AbortController();
    state.currentStreamingMsg = assistantMsg;  // Bug #5: 标记当前流式消息

    if (state.streamEnabled) {
      await streamResponse(content, assistantMsg, assistantEl);
    } else {
      await nonStreamResponse(content, assistantMsg, assistantEl);
    }
  } catch (err) {
    // 用户主动停止、刷新页面、关闭页面都属于正常中断，不弹错误
    const isUserInitiated = err.name === 'AbortError'
      || (err.name === 'TypeError' && err.message === 'Failed to fetch');
    if (isUserInitiated) {
      // Bug #9 修复：abort 后必须复位 streaming 状态 + 标记 interrupted，
      // 否则刷新页面后 renderMessage 看到 msg.streaming=true 会显示永不消失的光标
      assistantMsg.streaming = false;
      assistantMsg.interrupted = true;
      const el = getMessageEl(assistantMsg);
      if (el) {
        const cursor = el.querySelector('.streaming-cursor');
        if (cursor) cursor.remove();
        // 触发 interrupted badge 渲染（renderMessage 里检测 msg.interrupted）
        // 这里不整体重渲，只追加 badge 避免内容闪烁
        const body = el.querySelector('.message-body');
        if (body && !body.querySelector('.interrupted-badge')) {
          const badge = document.createElement('div');
          badge.className = 'interrupted-badge';
          badge.innerHTML = '⚠️ 回复已中断';
          body.appendChild(badge);
        }
      }
    } else {
      handleError(err, assistantMsg, assistantEl);
    }
  } finally {
    state.isStreaming = false;
    state.abortController = null;
    state.currentStreamingMsg = null;  // Bug #5: 清理引用
    setStreamingUI(false);
    saveSessions();
  }
}

function setStreamingUI(active) {
  if (active) {
    dom.btnSend.innerHTML = '⏹';
    dom.btnSend.classList.add('streaming');
    dom.btnSend.disabled = false;  // 停止按钮必须可点击
    dom.btnSend.title = '停止生成';
    dom.btnSend.setAttribute('aria-label', '停止生成');
  } else {
    dom.btnSend.innerHTML = '↑';
    dom.btnSend.classList.remove('streaming');
    dom.btnSend.title = '发送消息';
    dom.btnSend.setAttribute('aria-label', '发送消息');
    // disabled 状态由 composerInput 的 input 事件控制
    dom.btnSend.disabled = !dom.composerInput.value.trim();
  }
}

function updateStreamToggleUI() {
  if (!dom.btnStreamToggle) return;
  if (state.streamEnabled) {
    dom.btnStreamToggle.classList.add('on');
    dom.btnStreamToggle.setAttribute('aria-checked', 'true');
    dom.btnStreamToggle.title = '流式输出已开启 · 点击关闭（改为等全部生成后一次显示）';
  } else {
    dom.btnStreamToggle.classList.remove('on');
    dom.btnStreamToggle.setAttribute('aria-checked', 'false');
    dom.btnStreamToggle.title = '流式输出已关闭 · 点击开启（改为逐字显示回答）';
  }
}

function stopGeneration() {
  if (state.abortController) {
    state.abortController.abort();
    setStreamingUI(false);
  }
}

// regenerate() 已移除（死代码）

// ---------- Non-streaming ----------
async function nonStreamResponse(userMessage, assistantMsg, assistantEl) {
  const contentEl = assistantEl.querySelector('.message-content');
  const roleEl = assistantEl.querySelector('.message-role');

  const response = await fetch(API_BASE + '/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: getUserId(),
      session_id: state.currentSessionId,
      message: userMessage,
    }),
    signal: state.abortController.signal,
  });

  if (!response.ok) throw new Error(`服务器错误 (${response.status})`);

  const data = await response.json();
  assistantMsg.content = data.answer || '抱歉，没有得到回答。';
  assistantMsg.sources = (data.sources || []).map(s => ({ title: s, source: 'knowledge_base' }));
  assistantMsg.context = data.context || '';
  assistantMsg.toolCalls = data.tool_calls || [];
  assistantMsg.streaming = false;

  roleEl.textContent = '校园助手';
  renderMarkdown(contentEl, assistantMsg.content);
  renderMermaidBlocks(contentEl);  // 渲染 mermaid 块为 SVG
  showContextInPanel(assistantMsg);
  refreshToolCallBlocks(assistantEl, assistantMsg.toolCalls || []);
  // 非流式下补充追问（消息先渲染时内容为空没触发，现在内容到了要重算）
  addFollowUps(assistantEl, assistantMsg);

  const cursor = assistantEl.querySelector('.streaming-cursor');
  if (cursor) cursor.remove();
}

// ---------- SSE Streaming ----------
// Helper: find the current DOM element for a message, or return null if
// the user has navigated away and the element was detached.
function getMessageEl(msg) {
  if (!msg.id) return null;
  return dom.thread.querySelector(`[data-msg-id="${msg.id}"]`);
}

async function streamResponse(userMessage, assistantMsg, assistantEl) {
  // No closure-captured contentEl/roleEl — we re-query each tick so that
  // a session-switch → switch-back picks up the current DOM element.

  const response = await fetch(API_BASE + '/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: getUserId(),
      session_id: state.currentSessionId,
      message: userMessage,
    }),
    signal: state.abortController.signal,
  });

  if (!response.ok) {
    throw new Error(`服务器错误 (${response.status})`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    // Re-resolve the current DOM element for this message every tick.
    // The user may have switched sessions, in which case the old element
    // is detached and we just update the message data without touching DOM.
    const currentEl = getMessageEl(assistantMsg);
    const currentBody = currentEl ? currentEl.querySelector('.message-body') : null;
    const currentContent = currentEl ? currentEl.querySelector('.message-content') : null;
    const currentRole = currentEl ? currentEl.querySelector('.message-role') : null;
    const isVisible = !!currentEl;

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const payload = line.slice(6);

      // --- Tool call started ---
      if (payload.startsWith('[TOOL_START]')) {
        const tc = JSON.parse(payload.slice(13).trim());
        assistantMsg.toolCalls = assistantMsg.toolCalls || [];

        // 去重：loadSessionIntoThread 切换会话后重渲染时可能已恢复 toolCalls，
        // 但 stream 仍在运行，会重新发送 [TOOL_START] 事件，导致重复添加。
        const tcKey = tc.name + '::' + JSON.stringify(tc.input || {});
        const isDuplicate = assistantMsg.toolCalls.some(
          existing => existing.name + '::' + JSON.stringify(existing.input || {}) === tcKey
        );

        if (!isDuplicate) {
          assistantMsg.toolCalls.push(tc);
          if (isVisible) {
            const tcBlock = renderToolCallBlock(tc);
            // DOM 级去重：检查是否已有相同 key 的 DOM 元素
            if (currentBody && !currentBody.querySelector(`[data-tc-key="${CSS.escape(tcKey)}"]`)) {
              currentBody.appendChild(tcBlock);
            }
          }
        }
        showContextInPanel(assistantMsg);
        scrollToBottom();
        continue;
      }

      // --- Tool call ended ---
      if (payload.startsWith('[TOOL_END]')) {
        const update = JSON.parse(payload.slice(11).trim());
        const tcs = assistantMsg.toolCalls || [];
        const tc = tcs.find(t => t.name === update.name && t.status === 'running');
        if (tc) {
          tc.output = update.output || tc.output;
          tc.status = update.status || 'done';
        }
        if (isVisible) refreshToolCallBlocks(currentEl, tcs);
        showContextInPanel(assistantMsg);
        scrollToBottom();
        continue;
      }

      // --- Sources ---
      if (payload.startsWith('[SOURCES]')) {
        const srcList = JSON.parse(payload.slice(10).trim());
        assistantMsg.sources = srcList.map(s => ({ title: s, source: 'knowledge_base' }));
        showContextInPanel(assistantMsg);
        continue;
      }

      // --- Errors in stream ---
      if (payload.startsWith('[ERROR]')) {
        const errMsg = payload.slice(8).trim();
        assistantMsg.error = errMsg;
        assistantMsg.streaming = false;
        if (isVisible) {
          const cursor = currentContent ? currentContent.querySelector('.streaming-cursor') : null;
          if (cursor) cursor.remove();
          currentEl.appendChild(renderErrorBanner(errMsg));
        }
        continue;
      }

      // --- Context (must arrive before [DONE]) ---
      // 后端现在保证 [CONTEXT] 在 [DONE] 之前发，这样 [DONE] 分支调
      // showContextInPanel(msg) 时 msg.context 已经赋值
      if (payload.startsWith('[CONTEXT]')) {
        try {
          assistantMsg.context = JSON.parse(payload.slice(10).trim());
        } catch {
          assistantMsg.context = payload.slice(10).trim();
        }
        continue;
      }

      // --- Done ---
      if (payload === '[DONE]') {
        assistantMsg.streaming = false;
        // 后处理：模型输出的脏符号归一化
        // 注意：后端流式已用 JSON 包装 + 字面 \n 转真换行，这里不再重复转义
        let cleaned = assistantMsg.content
          // ### 或 ## 开头的行 → 真正的 markdown 标题（前后空行）
          .replace(/^#{2,3}\s+(.+)$/gm, '\n\n## $1\n\n')
          // 连续多个换行压成两个
          .replace(/\n{3,}/g, '\n\n')
          // 去掉开头多余的空白
          .trim();
        assistantMsg.content = cleaned;
        if (isVisible && currentContent) {
          renderMarkdown(currentContent, assistantMsg.content);
          const cursor = currentContent.querySelector('.streaming-cursor');
          if (cursor) cursor.remove();
          if (currentRole) currentRole.textContent = '校园助手';
          addMessageActions(currentEl, assistantMsg);
          // 渲染 mermaid 块为 SVG（流式完成后代码才完整，此时才能安全渲染）
          // P1-9: 清除流式过程中的失败标记，强制重新渲染
          requestAnimationFrame(() => {
            currentContent.querySelectorAll('.mermaid').forEach(b => {
              b.classList.remove('mermaid-rendered');               b.removeAttribute('data-processed');               b.innerHTML = '';               const src = b.getAttribute('data-source');               if (src) { try { b.textContent = decodeURIComponent(src); } catch (_) {} }             });             renderMermaidBlocks(currentContent);           });
        }
        extractSourcesFromContent(assistantMsg);
        showContextInPanel(assistantMsg);
        continue;
      }

      // --- Plain text chunk ---
      // 后端用 {'t': text} JSON 包装，避免 text 里的换行/方括号破坏 SSE 协议
      let textChunk = null;
      try {
        const data = JSON.parse(payload);
        if (data && typeof data.t === 'string') {
          // 新格式：JSON 包装的文本块
          textChunk = data.t;
        } else if (data && data.role === 'assistant' && currentRole) {
          // 兼容旧格式：role 事件
          currentRole.textContent = '校园助手';
        }
      } catch {
        // 向后兼容：纯文本 chunk（旧后端或未包装的文本）
        // 只有形如 [UPPER_CASE_IDENTIFIER] 的才是未识别控制事件，
        // [1] [2] [a] 等数字/小写方括号都当普通文本（引用编号等）
        if (/^\[[A-Z][A-Z_]*\]/.test(payload)) {
          console.warn('[stream] 未识别的控制事件，已忽略:', payload.slice(0, 40));
          continue;
        }
        textChunk = payload;
      }
      if (textChunk !== null) {
        assistantMsg.content += textChunk;
        if (isVisible && currentContent) {
          renderContentWithStreaming(currentContent, assistantMsg.content);
        }
      }
    }

    scrollToBottom();
  }

  // Fallback: if streaming ends without [DONE], clean up
  if (assistantMsg.streaming) {
    assistantMsg.streaming = false;
    const finalEl = getMessageEl(assistantMsg);
    if (finalEl) {
      const finalContent = finalEl.querySelector('.message-content');
      const finalRole = finalEl.querySelector('.message-role');
      const cursor = finalContent ? finalContent.querySelector('.streaming-cursor') : null;
      if (cursor) cursor.remove();
      if (finalRole) finalRole.textContent = finalRole.textContent || '校园助手';
      addMessageActions(finalEl, assistantMsg);
    }
    extractSourcesFromContent(assistantMsg);
    showContextInPanel(assistantMsg);
  }
}

// ---------- Rendering ----------
function addMessageActions(assistantEl, msg) {
  // Remove existing actions first
  const existing = assistantEl.querySelector('.message-actions');
  if (existing) existing.remove();

  if (msg.role !== 'assistant' || msg.streaming || msg.error || !msg.content) return;

  const body = assistantEl.querySelector('.message-body');
  if (!body) return;

  const actions = document.createElement('div');
  actions.className = 'message-actions';
  actions.innerHTML = `
    <button class="btn-msg-action btn-copy-msg">📋 复制</button>
  `;
  body.appendChild(actions);
}

function renderMessage(msg) {
  if (!msg.id) msg.id = crypto.randomUUID ? crypto.randomUUID() : ('m-' + Date.now() + '-' + Math.random().toString(36).slice(2));

  // 系统消息：上传通知等，走简单渲染
  if (msg.role === 'system') {
    const el = document.createElement('div');
    el.className = 'message system';
    el.setAttribute('data-msg-id', msg.id);
    el.innerHTML = `<div class="message-body"><div class="message-content" style="background:var(--bg-elevated);border:1px dashed var(--border);padding:8px 12px;border-radius:8px;font-size:0.9em;color:var(--text-secondary);">${escapeHtml(msg.content)}</div></div>`;
    return el;
  }

  const wrapper = document.createElement('div');
  wrapper.className = `message ${msg.role}`;
  wrapper.setAttribute('role', 'article');
  wrapper.setAttribute('data-msg-id', msg.id);

  // Avatar
  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  avatar.textContent = msg.role === 'user' ? 'U' : 'AI';
  avatar.setAttribute('aria-hidden', 'true');

  // Body
  const body = document.createElement('div');
  body.className = 'message-body';

  // Role label
  const role = document.createElement('span');
  role.className = 'message-role';
  role.textContent = msg.role === 'user' ? '你' : (msg.streaming ? '思考中...' : '校园助手');

  // Content
  const content = document.createElement('div');
  content.className = 'message-content';

  if (msg.role === 'assistant') {
    renderAssistantContent(content, msg);
  } else {
    // 用户消息：\n 转 <br>，确保换行保留
    content.innerHTML = msg.content.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br>');
  }

  body.append(role, content);

  // Tool call blocks
  if (msg.toolCalls && msg.toolCalls.length > 0) {
    for (const tc of msg.toolCalls) {
      body.appendChild(renderToolCallBlock(tc));
    }
  }

  // Error banner
  if (msg.error) {
    body.appendChild(renderErrorBanner(msg.error));
  }

  // Interrupted badge
  if (msg.interrupted) {
    const badge = document.createElement('div');
    badge.className = 'interrupted-badge';
    badge.innerHTML = '⚠️ 回复已中断';
    body.appendChild(badge);
  }

  // Source badge — distinguish knowledge base vs web search
  if (msg.role === 'assistant' && !msg.streaming && !msg.error && msg.content) {
    const sourceBadge = document.createElement('div');
    sourceBadge.className = 'source-badge';
    if (msg.sources && msg.sources.length > 0) {
      sourceBadge.innerHTML = '📄 来自知识库';
    } else if (msg.toolCalls && msg.toolCalls.some(tc => tc.name === 'web_search')) {
      sourceBadge.innerHTML = '🌐 来自互联网';
    }
    if (sourceBadge.innerHTML) body.appendChild(sourceBadge);
  }

  // Message action buttons (assistant only, after streaming done)
  if (msg.role === 'assistant' && !msg.streaming && !msg.error && msg.content) {
    const actions = document.createElement('div');
    actions.className = 'message-actions';
    actions.innerHTML = `<button class="btn-msg-action btn-copy-msg">📋 复制</button>`;
    body.appendChild(actions);

    // Quick follow-up suggestions — only on the LAST assistant message
    const assistantMessages = dom.thread.querySelectorAll('.message.assistant');
    const isLastMessage = assistantMessages.length === 0 ||
      (wrapper === assistantMessages[assistantMessages.length - 1]);
    if (isLastMessage) {
      addFollowUps(wrapper, msg);
    }
  }

  wrapper.append(avatar, body);
  return wrapper;
}

function renderAssistantContent(el, msg) {
  if (msg.streaming) {
    renderContentWithStreaming(el, msg.content);
  } else {
    renderMarkdown(el, msg.content);
    renderCitations(el, msg.sources);
  }
}

function renderContentWithStreaming(el, text) {
  // Bug #6 修复：流式过程中模型可能只吐了半个标题（如 "## 标" 还没换行），
  // 此时 renderMarkdown 的 ^## (.+)$ 正则虽然能匹配，但渲染出的 <h2> 后面
  // 紧跟着未换行的正文，视觉上标题和内容挤在一起。
  // 解决：流式态临时把行首的 # 前缀去掉（改成普通文本），避免半截标题被渲染成标签。
  // [DONE] 后由 finalize 统一调 renderMarkdown 做完整渲染，那时所有换行都到齐了。
  const streamSafeText = text.replace(/^#{1,6}\s+/gm, '');
  renderMarkdown(el, streamSafeText);
  // Ensure cursor
  let cursor = el.querySelector('.streaming-cursor');
  if (!cursor) {
    cursor = document.createElement('span');
    cursor.className = 'streaming-cursor';
    cursor.setAttribute('aria-hidden', 'true');
    el.appendChild(cursor);
  }
}

// ---------- Markdown Parser (lightweight) ----------
function renderMarkdown(el, text) {
  if (!text) { el.innerHTML = ''; return; }

  let html = text;

  // Escape HTML but allow our own tags
  html = html.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  // Code blocks (fenced)
  html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    return renderCodeBlock(lang, code.trim());
  });

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code style="background:var(--bg-code);border:1px solid var(--border-light);border-radius:4px;padding:1px 5px;font-family:var(--font-mono);font-size:0.85em;">$1</code>');

  // Bold
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

  // Italic
  html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

  // Headers
  html = html.replace(/^###### (.+)$/gm, '<h6 style="font-size:0.95em;font-weight:600;margin:8px 0 4px;">$1</h6>');
  html = html.replace(/^##### (.+)$/gm, '<h5 style="font-size:1.0em;font-weight:600;margin:9px 0 5px;">$1</h5>');
  html = html.replace(/^#### (.+)$/gm, '<h4 style="font-size:1.05em;font-weight:600;margin:10px 0 5px;">$1</h4>');
  html = html.replace(/^### (.+)$/gm, '<h3 style="font-size:1.1em;font-weight:600;margin:12px 0 6px;">$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2 style="font-size:1.2em;font-weight:700;margin:14px 0 6px;">$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1 style="font-size:1.3em;font-weight:700;margin:16px 0 8px;">$1</h1>');

  // Numbered lists
  html = html.replace(/^(\d+)\. (.+)$/gm, '<li style="margin-left:20px;list-style-type:decimal;">$2</li>');
  html = html.replace(/(<li[^>]*>.*<\/li>\n?)+/g, '<ol style="margin:6px 0;">$&</ol>');

  // Unordered lists
  html = html.replace(/^- (.+)$/gm, '<li style="margin-left:20px;list-style-type:disc;">$1</li>');

  // Paragraphs (double newlines)
  html = html.replace(/\n\n/g, '</p><p style="margin:8px 0;">');
  html = '<p style="margin:8px 0;">' + html + '</p>';

  // Cleanup empty paragraphs
  html = html.replace(/<p[^>]*>\s*<\/p>/g, '');

  // P0-4 方案B: 先保护代码块，再做 \n → <br>，最后还原
  var _codePlaceholders = [];
  html = html.replace(/<(div class="code-block-wrapper"|details class="mermaid-code-details"|pre class="mermaid-source"|div class="mermaid" data-source)[\s\S]*?<\/div>/g, function(m) {
    _codePlaceholders.push(m);
    return '\u0000CODE' + (_codePlaceholders.length - 1) + '\u0000';
  });
  // Single newlines to <br>（不影响代码块）
  html = html.replace(/\n/g, '<br>');
  // 还原代码块
  html = html.replace(/\u0000CODE(\d+)\u0000/g, function(_, i) {
    return _codePlaceholders[parseInt(i, 10)];
  });

  el.innerHTML = html;
}

function renderCodeBlock(lang, code) {
  // mermaid 块：折叠源码 + 渲染 SVG 容器
  // 折叠解决"代码占很大空间"问题，<div class="mermaid"> 由 mermaid.run() 渲染成 SVG
  // mermaid 支持的图表类型：mermaid/graph/flowchart/mindmap/sequenceDiagram 等
  var _mermaidLangs = ['mermaid', 'mindmap', 'graph', 'flowchart', 'sequenceDiagram', 'classDiagram', 'stateDiagram', 'erDiagram', 'journey', 'gantt', 'pie', 'timeline', 'quadrantChart', 'gitGraph', 'C4Context'];
  if (_mermaidLangs.indexOf(lang) !== -1) {
    const escaped = escapeHtml(code);
    // 图默认展开显示，点击 summary 可折叠/展开图（源码不显示，用户不需要看）
    return `<div class="mermaid-wrapper">
      <details class="mermaid-chart-details" open>
        <summary class="mermaid-summary">🎨 图表（点击收起/展开）</summary>
        <div class="mermaid" data-source="${encodeURIComponent(code)}">${escaped}</div>
      </details>
    </div>`;
  }
  // 普通代码块（保持原逻辑）
  return `<div class="code-block-wrapper">
    <div class="code-block-header">
      <span class="code-block-lang">${lang || 'code'}</span>
      <button class="btn-copy">Copy</button>
    </div>
    <pre class="code-block">${escapeHtml(code)}</pre>
  </div>`;
}

// 渲染容器内所有未渲染的 mermaid 块（流式 [DONE] 后 / 非流式 / 切换会话恢复后调用）
// mermaid.run 失败时降级显示原始代码，不阻塞页面
// Check if mermaid is loaded and ready
// Official mermaid.run API implementation
async function renderMermaidBlocks(container) {
  if (!window.mermaid) {
    console.warn('[mermaid] mermaid not loaded, retrying in 200ms');
    setTimeout(() => renderMermaidBlocks(container), 200);
    return;
  }

  // Ensure mermaid is initialized
  if (!window.__mermaidInitialized) {
    try {
      await window.mermaid.initialize(Object.assign({ 
        startOnLoad: false,
        securityLevel: 'strict',
        theme: 'base',
        look: 'handDrawn'
      }, window.__mermaidThemes[document.documentElement.getAttribute('data-theme') || 'dark']));
      window.__mermaidInitialized = true;
    } catch (e) {
      console.warn('[mermaid] init failed:', e);
      return;
    }
  }

  // Find all unprocessed mermaid elements
  const blocks = container.querySelectorAll('div.mermaid:not(.mermaid-rendered)');
  if (!blocks.length) return;

  // Clear previous state and restore source code
  blocks.forEach(b => {
    b.classList.remove('mermaid-rendered');
    b.removeAttribute('data-processed');
    b.innerHTML = '';
    const src = b.getAttribute('data-source');
    if (src) {
      try { b.textContent = decodeURIComponent(src); } catch (_) {}
    }
  });

  try {
    // Use official mermaid.run API
    await window.mermaid.run({
      nodes: blocks,
      suppressErrors: true
    });
    
    // Mark as rendered
    blocks.forEach(b => b.classList.add('mermaid-rendered'));
    
  } catch (e) {
    console.warn('[mermaid] render failed:', e);
    blocks.forEach(b => {
      b.classList.add('mermaid-rendered');
      if (!b.querySelector('.mermaid-error-tip')) {
        b.innerHTML = '<div class="mermaid-error-tip" style="color: #ff6b6b; padding: 10px; background: #ffe0e0; border-radius: 4px; margin: 10px 0;">⚠️ 图表代码不完整或语法错误，请检查代码格式</div>';
      }
    });
  }
}


// ---------- Missing Helpers (补全) ----------

// 1. escapeHtml(text)
// 被 addUploadMessage / renderCodeBlock / renderErrorBanner 调用
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// 2. formatTime(timestamp)
// 被 renderConversationList 调用，区分今天/非今天
function formatTime(ts) {
  const d = new Date(ts);
  const now = new Date();
  const isToday = d.toDateString() === now.toDateString();
  if (isToday) {
    return d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', hour12: false });
  }
  return (d.getMonth() + 1) + '/' + d.getDate();
}

// 3. scrollToBottom(force)
// 被 loadSessionIntoThread / sendMessage / streamResponse 调用
// 智能滚动：用户主动往上翻时不强制拉回底部（除非 force=true）
function scrollToBottom(force) {
  // force=true 时无条件滚动（用于发送消息、切换会话等场景）
  if (force) {
    requestAnimationFrame(() => {
      dom.thread.scrollTop = dom.thread.scrollHeight;
    });
    return;
  }
  // 判断用户是否在底部附近（80px 容差）
  const distFromBottom = dom.thread.scrollHeight - dom.thread.scrollTop - dom.thread.clientHeight;
  if (distFromBottom <= 80) {
    requestAnimationFrame(() => {
      dom.thread.scrollTop = dom.thread.scrollHeight;
    });
  }
}

// 4. togglePanel(forceOpen)
// 被 bindEvents 调用，使用 CSS class "collapsed" 控制面板隐藏
function togglePanel(forceOpen) {
  if (typeof forceOpen === 'boolean') {
    state.panelOpen = forceOpen;
  } else {
    state.panelOpen = !state.panelOpen;
  }
  dom.panel.classList.toggle('collapsed', !state.panelOpen);
}

// 5. showContextInPanel(msg)
// 被 nonStreamResponse / streamResponse 调用，展示发送给 LLM 的完整提示词
function showContextInPanel(msg) {
  if (!msg || !msg.context) {
    dom.panelEmpty.style.display = '';
    dom.panelContent.innerHTML = '';
    return;
  }
  dom.panelEmpty.style.display = 'none';
  const text = typeof msg.context === 'string' ? msg.context : JSON.stringify(msg.context, null, 2);
  dom.panelContent.innerHTML = '<pre style="white-space:pre-wrap;word-break:break-word;font-size:12px;line-height:1.6;color:var(--text-secondary);font-family:var(--font-mono);">' + escapeHtml(text) + '</pre>';
}

// 6. extractSourcesFromContent(msg)
// 被 streamResponse [DONE] 分支和结尾 fallback 调用
function extractSourcesFromContent(msg) {
  if (!msg || !msg.content) return;
  const matches = [...msg.content.matchAll(/来自《(.+?)》/g)];
  if (matches.length > 0) {
    msg.sources = matches.map(m => ({ title: m[1], source: 'knowledge_base' }));
  }
}

// 7. renderErrorBanner(errorMsg)
// 被 streamResponse [ERROR] 分支和 renderMessage 调用
function renderErrorBanner(errorMsg) {
  const banner = document.createElement('div');
  banner.className = 'error-banner';
  banner.innerHTML = `
    <span class="error-icon">⚠️</span>
    <div class="error-content">
      <div class="error-message">${escapeHtml(errorMsg)}</div>
      <div class="error-actions">
        <button class="btn-retry" onclick="retryLastMessage()">重试</button>
        <button class="btn-copy-error" onclick="copyErrorInfo()">复制错误</button>
      </div>
    </div>
  `;
  return banner;
}

// 8. renderToolCallBlock(tc)
// 被 streamResponse [TOOL_START] 和 renderMessage 调用
function renderToolCallBlock(tc) {
  const block = document.createElement('div');
  block.className = 'tool-call-block';
  const tcKey = tc.name + '::' + JSON.stringify(tc.input || {});
  block.setAttribute('data-tc-key', tcKey);

  const toolIcons = {
    'search_knowledge_base': '📚',
    'web_search': '🌐',
    'get_current_time': '🕐',
    'fetch_webpage': '📄',
  };
  const icon = toolIcons[tc.name] || '🔧';
  const toolLabels = {
    'search_knowledge_base': '知识库检索',
    'web_search': '联网搜索',
    'get_current_time': '获取时间',
    'fetch_webpage': '读取网页',
  };
  const label = toolLabels[tc.name] || tc.name;

  const statusText = tc.status === 'running' ? '运行中' : (tc.status === 'error' ? '错误' : '完成');

  const header = document.createElement('div');
  header.className = 'tool-call-header';
  header.innerHTML = `
    <span class="tool-call-icon">${icon}</span>
    <span class="tool-call-name">${escapeHtml(label)}</span>
    <span class="tool-call-status ${tc.status || 'done'}">${statusText}</span>
    <span class="tool-call-chevron">▼</span>
  `;

  const body = document.createElement('div');
  body.className = 'tool-call-body';
  const inner = document.createElement('div');
  inner.className = 'tool-call-body-inner';

  // Input section
  if (tc.input && Object.keys(tc.input).length > 0) {
    const inputSection = document.createElement('div');
    inputSection.className = 'tool-call-section';
    inputSection.innerHTML = `
      <div class="tool-call-section-label">输入</div>
      <div class="tool-call-json">${escapeHtml(JSON.stringify(tc.input, null, 2))}</div>
    `;
    inner.appendChild(inputSection);
  }

  // Output section
  if (tc.output) {
    const outputSection = document.createElement('div');
    outputSection.className = 'tool-call-section';
    outputSection.innerHTML = `
      <div class="tool-call-section-label">输出</div>
      <div class="tool-call-json">${escapeHtml(tc.output)}</div>
    `;
    inner.appendChild(outputSection);
  }

  body.appendChild(inner);
  block.append(header, body);
  return block;
}

// 9. refreshToolCallBlocks(el, toolCalls)
// 被 nonStreamResponse 和 streamResponse [TOOL_END] 调用
function refreshToolCallBlocks(el, toolCalls) {
  if (!el) return;
  // 移除旧的工具调用块
  el.querySelectorAll('.tool-call-block').forEach(b => b.remove());
  // 重新渲染
  const body = el.querySelector('.message-body');
  if (!body) return;
  // 在 error-banner 之前插入，如果没有 error-banner 就追加到最后
  const errorBanner = body.querySelector('.error-banner');
  for (const tc of toolCalls) {
    const block = renderToolCallBlock(tc);
    if (errorBanner) {
      body.insertBefore(block, errorBanner);
    } else {
      body.appendChild(block);
    }
  }
}

// 10. addFollowUps(el, msg)
// 被 nonStreamResponse 和 renderMessage 调用，在助手消息末尾追加追问建议
function addFollowUps(el, msg) {
  // 移除已有的追问
  const existing = el.querySelector('.follow-up-suggestions');
  if (existing) existing.remove();

  if (!msg.content || msg.content.length < 10) return;

  // 根据回答内容生成简单的追问建议
  const suggestions = [];
  if (msg.sources && msg.sources.length > 0) {
    suggestions.push('还有其他相关资料吗？');
  }
  if (msg.toolCalls && msg.toolCalls.some(tc => tc.name === 'web_search')) {
    suggestions.push('能搜索更多最新信息吗？');
  }
  suggestions.push('能详细解释一下吗？');
  suggestions.push('有没有具体的操作步骤？');

  const container = document.createElement('div');
  container.className = 'follow-up-suggestions';
  container.innerHTML = '<span class="follow-up-label">追问：</span>';
  for (const s of suggestions.slice(0, 3)) {
    const btn = document.createElement('button');
    btn.className = 'btn-follow-up';
    btn.textContent = s;
    btn.addEventListener('click', () => {
      dom.composerInput.value = s;
      sendMessage();
    });
    container.appendChild(btn);
  }

  const body = el.querySelector('.message-body');
  if (body) body.appendChild(container);
}

// 11. renderCitations(el, sources)
// 被 renderAssistantContent 调用，在内容末尾追加来源引用标签
function renderCitations(el, sources) {
  if (!sources || sources.length === 0) return;
  const container = document.createElement('div');
  container.style.cssText = 'margin-top:8px;display:flex;flex-wrap:wrap;gap:4px;';
  for (let i = 0; i < sources.length; i++) {
    const tag = document.createElement('span');
    tag.className = 'citation-tag';
    tag.setAttribute('data-index', i);
    tag.textContent = '📎 ' + sources[i].title;
    container.appendChild(tag);
  }
  el.appendChild(container);
}

// 12. handleError(err, assistantMsg, assistantEl)
// 被 sendMessage catch 块调用，处理流式/非流式请求的错误回退
function handleError(err, assistantMsg, assistantEl) {
  assistantMsg.streaming = false;
  assistantMsg.error = err.message || '未知错误';
  const contentEl = assistantEl.querySelector('.message-content');
  if (contentEl) {
    const cursor = contentEl.querySelector('.streaming-cursor');
    if (cursor) cursor.remove();
  }
  const body = assistantEl.querySelector('.message-body');
  if (body) {
    body.appendChild(renderErrorBanner(assistantMsg.error));
  }
}

// 13. highlightSourceInPanel(idx)
// 被 bindEvents 中的 citation-tag 点击事件调用，高亮右栏对应来源
function highlightSourceInPanel(idx) {
  const tags = dom.panelContent.querySelectorAll('.citation-tag, .source-item');
  if (tags[idx]) {
    tags.forEach(t => t.style.background = '');
    tags[idx].style.background = 'var(--accent-muted)';
    tags[idx].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

// ---------- Boot ----------
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
