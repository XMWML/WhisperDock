/* WhisperDock static client. Persistent workspace settings stay in the project API. */
(function () {
  'use strict';

  const API = '/api';
  const ALL_OUTPUT_FORMATS = ['txt', 'json', 'srt', 'vtt', 'tsv'];
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const state = {
    models: [],
    history: [],
    files: [],
    currentJob: null,
    jobTimer: null,
    currentResult: null,
    historyPage: 1,
    recording: null,
    recordMode: 'file',
    realtime: null,
    realtimeSequence: 0,
    liveTranscript: '',
    recorderTimer: null,
    recorderStartedAt: null,
    health: null,
    preferences: loadPreferences(),
  };

  const el = {
    sidebar: $('#sidebar'),
    audioInput: $('#audio-input'),
    dropZone: $('#drop-zone'),
    queue: $('#file-queue'),
    queueItems: $('#queue-items'),
    transcribe: $('#transcribe-btn'),
    modelSelect: $('#model-select'),
    modelHint: $('#model-hint'),
    activeModel: $('#active-model-name'),
    todayCount: $('#today-count'),
    modelCount: $('#model-count'),
    modelGrid: $('#model-grid'),
    historyBody: $('#history-body'),
    historySummary: $('#history-summary'),
    historyPagination: $('#history-pagination'),
    resultPanel: $('#result-panel'),
    resultText: $('#result-text'),
    resultSegments: $('#result-segments'),
    resultJson: $('#result-json'),
    resultMeta: $('#result-meta'),
    jobProgress: $('#job-progress'),
    progressFill: $('#progress-fill'),
    progressPercent: $('#progress-percent'),
    progressStatus: $('#progress-status'),
    recordBtn: $('#record-btn'),
    recordLabel: $('#record-label'),
    recordTimer: $('#record-timer'),
    modelDialog: $('#model-dialog'),
    modelForm: $('#model-form'),
    detailsDialog: $('#model-details-dialog'),
    detailBody: $('#model-detail-body'),
    detailTitle: $('#detail-model-name'),
    workspacePath: $('#workspace-path'),
    livePanel: $('#live-panel'),
    liveStatus: $('#live-status'),
    livePending: $('#live-pending'),
    liveTranscript: $('#live-transcript'),
  };

  function loadPreferences() {
    return {
      keepTemp: true,
      checkModels: true,
      defaultLanguage: 'auto',
      defaultFormat: 'all',
      defaultDevice: 'auto',
      defaultPrompt: '',
    };
  }

  async function api(path, options) {
    const response = await fetch(`${API}${path}`, Object.assign({ headers: { Accept: 'application/json' } }, options || {}));
    const type = response.headers.get('content-type') || '';
    const data = type.includes('application/json') ? await response.json().catch(() => ({})) : await response.text();
    if (!response.ok) {
      const message = typeof data === 'object' ? (data.detail || data.message || data.error) : data;
      throw new Error(typeof message === 'string' && /<\/?(?:html|body|!doctype)/i.test(message) ? `请求失败 (${response.status})` : (message || `请求失败 (${response.status})`));
    }
    return data;
  }

  let settingsSaveTimer = null;

  function outputFormatChoice(formats) {
    const values = Array.isArray(formats) ? formats.filter((item) => typeof item === 'string') : [];
    if (ALL_OUTPUT_FORMATS.every((format) => values.includes(format))) return 'all';
    return values.find((format) => ALL_OUTPUT_FORMATS.includes(format)) || 'txt';
  }

  function outputFormatList(value) {
    return value === 'all' ? ALL_OUTPUT_FORMATS.slice() : [value || 'txt'];
  }

  function applyProjectSettings(settings) {
    if (!settings || typeof settings !== 'object') return;
    state.preferences.keepTemp = typeof settings.keep_uploads === 'boolean' ? settings.keep_uploads : state.preferences.keepTemp;
    state.preferences.defaultLanguage = settings.default_language || 'auto';
    state.preferences.defaultFormat = outputFormatChoice(settings.default_output_formats);
    state.preferences.defaultDevice = ['auto', 'cpu', 'mps', 'cuda'].includes(settings.default_device) ? settings.default_device : 'auto';
    state.preferences.defaultPrompt = typeof settings.default_prompt === 'string' ? settings.default_prompt : state.preferences.defaultPrompt;
    applyPreferencesToForm();
  }

  async function loadProjectSettings() {
    try {
      applyProjectSettings(await api('/settings'));
    } catch (error) {
      // Keep the migrated local values when an older or offline backend has no settings endpoint.
      console.warn('Project settings failed:', error);
    }
  }

  function projectSettingsPayload() {
    return {
      default_device: state.preferences.defaultDevice || 'auto',
      default_output_formats: outputFormatList(state.preferences.defaultFormat),
      keep_uploads: Boolean(state.preferences.keepTemp),
      default_language: state.preferences.defaultLanguage === 'auto' ? null : state.preferences.defaultLanguage,
      default_prompt: state.preferences.defaultPrompt || '',
    };
  }

  function scheduleProjectSettingsSave() {
    window.clearTimeout(settingsSaveTimer);
    settingsSaveTimer = window.setTimeout(async () => {
      try {
        applyProjectSettings(await api('/settings', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
          body: JSON.stringify(projectSettingsPayload()),
        }));
      } catch (error) {
        toast('项目设置未保存', error.message, 'error');
      }
    }, 350);
  }

  function toast(title, message, kind) {
    const item = document.createElement('div');
    item.className = `toast ${kind || 'info'}`;
    item.innerHTML = `<span class="toast-icon">${kind === 'error' ? '!' : kind === 'success' ? '✓' : 'i'}</span><div><strong>${escapeHTML(title)}</strong>${message ? `<p>${escapeHTML(message)}</p>` : ''}</div><button class="toast-close" type="button" aria-label="关闭">×</button>`;
    $('.toast-region').appendChild(item);
    const remove = () => item.remove();
    $('button', item).addEventListener('click', remove);
    window.setTimeout(remove, 5200);
  }

  function escapeHTML(value) {
    return String(value == null ? '' : value).replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[char]);
  }

  function formatBytes(bytes) {
    const value = Number(bytes);
    if (!Number.isFinite(value) || value < 0) return '--';
    if (value < 1024) return `${value} B`;
    const units = ['KB', 'MB', 'GB', 'TB'];
    let size = value / 1024;
    let index = 0;
    while (size >= 1024 && index < units.length - 1) { size /= 1024; index += 1; }
    return `${size >= 10 || index === 0 ? size.toFixed(0) : size.toFixed(1)} ${units[index]}`;
  }

  function formatDuration(seconds) {
    const value = Number(seconds);
    if (!Number.isFinite(value) || value < 0) return '--';
    const h = Math.floor(value / 3600);
    const m = Math.floor((value % 3600) / 60);
    const s = Math.floor(value % 60);
    return h ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}` : `${m}:${String(s).padStart(2, '0')}`;
  }

  function formatDate(value) {
    if (!value) return '--';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return new Intl.DateTimeFormat('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }).format(date);
  }

  function slugFromFile(file) {
    return file.name.replace(/\.[^/.]+$/, '').replace(/[^\w\u4e00-\u9fff]+/g, '-').slice(0, 48) || 'recording';
  }

  function sourceTypeFor(source) {
    const value = String(source || '').trim();
    if (/^https?:\/\//i.test(value)) return 'url';
    if (/^(\.|\/|~[\/]|models[\\/])/.test(value) || /^[A-Za-z]:[\\/]/.test(value)) return 'local';
    return 'huggingface';
  }

  function safeModelId(source) {
    const value = String(source || '').trim().toLowerCase();
    return value.replace(/[^a-z0-9._-]+/g, '-').replace(/^[._-]+|[._-]+$/g, '').slice(0, 64) || 'custom-model';
  }

  function normaliseSource(source, fallback) {
    if (typeof source === 'string') return source;
    if (source && typeof source === 'object') {
      return source.hf_repo || source.repo_id || source.url || source.local_path || source.model_name || source.path || source.kind || fallback || '';
    }
    return fallback || '';
  }

  function normaliseModel(raw, index) {
    raw = raw || {};
    const id = raw.id || raw.model_id || raw.key || raw.name || `model-${index}`;
    const loaded = raw.loaded === true || raw.in_memory === true || raw.status === 'loaded';
    const downloaded = raw.downloaded === true || raw.installed === true || raw.local === true || loaded || ['downloaded', 'ready', 'available'].includes(String(raw.status || '').toLowerCase());
    return {
      id: String(id),
      name: raw.name || raw.display_name || String(id),
      source: normaliseSource(raw.source, raw.repo_id || raw.url || raw.model_path || String(id)),
      size: raw.size || raw.model_size || raw.variant || raw.precision || raw.parameters || '--',
      bytes: raw.bytes || raw.size_bytes || raw.disk_size || raw.download_size,
      backend: raw.engine || raw.backend || 'openai-whisper',
      status: raw.error ? 'error' : loaded ? 'loaded' : downloaded ? 'downloaded' : 'remote',
      loaded,
      downloaded,
      path: raw.path || raw.local_path || raw.directory || '',
      notes: raw.notes || raw.description || '',
      updatedAt: raw.updated_at || raw.created_at || '',
      raw,
    };
  }

  function unwrapList(data, keys) {
    if (Array.isArray(data)) return data;
    for (const key of keys) if (data && Array.isArray(data[key])) return data[key];
    return [];
  }

  async function refreshHealth() {
    try {
      const data = await api('/health');
      state.health = data || {};
      const online = !data || !data.status || ['ok', 'healthy', 'online', 'ready'].includes(String(data.status).toLowerCase());
      setServiceState(online, online ? '本地服务已连接' : '服务状态异常');
      const workspace = data.workspace_path || data.project_path || data.root_path;
      if (workspace) el.workspacePath.textContent = workspace;
      const version = data.version || data.app_version;
      if (version) {
        $('#app-version').textContent = `v${String(version).replace(/^v/, '')}`;
        $('#about-version').textContent = `版本 v${String(version).replace(/^v/, '')}`;
      }
      updateStorage(data);
    } catch (error) {
      setServiceState(false, '本地服务未连接');
      $('#health-label').textContent = '离线';
      console.warn('Health check failed:', error);
    }
  }

  function setServiceState(online, message) {
    const dots = $$('.status-dot');
    dots.forEach((dot) => {
      if (dot.id === 'health-dot' || dot.closest('.service-state')) {
        dot.classList.toggle('online', online);
        dot.classList.toggle('offline', !online);
      }
    });
    $('#health-label').textContent = online ? '已连接' : '离线';
    $('#service-state span:last-child').textContent = message;
  }

  function updateStorage(data) {
    const used = data.storage_used_bytes || data.used_bytes || data.storage?.used_bytes;
    const total = data.storage_total_bytes || data.total_bytes || data.storage?.total_bytes;
    const modelBytes = data.model_storage_bytes || data.storage?.models_bytes;
    if (used != null) $('#storage-used').textContent = `已用 ${formatBytes(used)}`;
    else if (modelBytes != null) $('#storage-used').textContent = `模型 ${formatBytes(modelBytes)}`;
    if (total != null) {
      $('#storage-total').textContent = `共 ${formatBytes(total)}`;
      $('#storage-meter-fill').style.width = `${Math.min(100, Math.max(0, (Number(used || 0) / Number(total || 1)) * 100))}%`;
    } else {
      $('#storage-total').textContent = '可迁移';
      $('#storage-meter-fill').style.width = '31%';
    }
  }

  async function refreshModels() {
    try {
      const data = await api('/models');
      state.models = unwrapList(data, ['models', 'items', 'data']).map(normaliseModel);
      renderModels();
      renderModelSelect();
    } catch (error) {
      state.models = [];
      renderModels(error.message);
      renderModelSelect();
      console.warn('Model list failed:', error);
    }
  }

  function renderModels(errorMessage) {
    const query = ($('#model-search').value || '').trim().toLowerCase();
    const status = $('#model-filter').value;
    const filtered = state.models.filter((model) => {
      const text = `${model.name} ${model.id} ${model.source}`.toLowerCase();
      return (!query || text.includes(query)) && (status === 'all' || model.status === status);
    });
    el.modelCount.textContent = state.models.length;
    if (errorMessage) {
      el.modelGrid.innerHTML = `<div class="empty-state"><span class="empty-icon">!</span><h3>无法读取模型目录</h3><p>${escapeHTML(errorMessage)}</p></div>`;
      return;
    }
    if (!filtered.length) {
      const hasModels = state.models.length > 0;
      el.modelGrid.innerHTML = `<div class="empty-state"><span class="empty-icon">${hasModels ? '⌕' : '▣'}</span><h3>${hasModels ? '没有符合条件的模型' : '还没有本地模型'}</h3><p>${hasModels ? '调整搜索或状态筛选后重试。' : '添加官方 Whisper 模型或自定义微调模型。'}</p>${hasModels ? '' : '<button class="primary-btn" data-add-model type="button">添加模型</button>'}</div>`;
      $('[data-add-model]', el.modelGrid)?.addEventListener('click', openModelDialog);
      return;
    }
    el.modelGrid.innerHTML = filtered.map((model) => {
      const statusText = { loaded: '已加载到内存', downloaded: '已下载', remote: '待下载', error: '需要处理' }[model.status] || model.status;
      const action = model.loaded ? '<button class="secondary-btn" data-model-action="unload" type="button">卸载内存</button>' : model.downloaded ? '<button class="primary-btn" data-model-action="load" type="button">加载模型</button>' : '<button class="primary-btn" data-model-action="download" type="button">下载模型</button>';
      return `<article class="model-card ${model.loaded ? 'loaded' : ''}" data-model-id="${escapeHTML(model.id)}">
        <div class="model-card-header"><span class="model-symbol">W</span><div class="model-title"><strong title="${escapeHTML(model.name)}">${escapeHTML(model.name)}</strong><span title="${escapeHTML(model.source)}">${escapeHTML(model.source)}</span></div><button class="model-menu" data-model-action="details" type="button" title="模型详情">⋯</button></div>
        <span class="model-status ${model.status}">${statusText}</span>
        <div class="model-facts"><div class="model-fact"><span>模型大小</span><strong>${escapeHTML(String(model.size))}${model.bytes ? ` · ${formatBytes(model.bytes)}` : ''}</strong></div><div class="model-fact"><span>推理后端</span><strong>${escapeHTML(model.backend)}</strong></div></div>
        <div class="model-actions">${action}<button class="icon-btn" data-model-action="delete" type="button" title="删除模型">⌫</button></div>
      </article>`;
    }).join('');
    $$('[data-model-action]', el.modelGrid).forEach((button) => button.addEventListener('click', () => modelAction(button)));
  }

  function renderModelSelect() {
    const loaded = state.models.filter((model) => model.loaded);
    const previous = el.modelSelect.value;
    if (!loaded.length) {
      el.modelSelect.innerHTML = '<option value="">没有已加载的模型</option>';
      el.modelHint.textContent = state.models.some((model) => model.downloaded) ? '请在模型管理中加载一个已下载的模型' : '请先在模型管理中下载并加载模型';
      el.activeModel.textContent = '未加载';
      updateTranscribeButton();
      return;
    }
    el.modelSelect.innerHTML = loaded.map((model) => `<option value="${escapeHTML(model.id)}">${escapeHTML(model.name)}</option>`).join('');
    el.modelSelect.value = loaded.some((model) => model.id === previous) ? previous : loaded[0].id;
    const current = loaded.find((model) => model.id === el.modelSelect.value);
    el.activeModel.textContent = current?.name || '已加载';
    el.modelHint.textContent = `${loaded.length} 个模型已加载到内存`;
    updateTranscribeButton();
  }

  async function modelAction(button) {
    const card = button.closest('[data-model-id]');
    const id = card?.dataset.modelId;
    const model = state.models.find((candidate) => candidate.id === id);
    const action = button.dataset.modelAction;
    if (!id || !action) return;
    if (action === 'details') { openModelDetails(model); return; }
    if (action === 'download') {
      button.disabled = true;
      button.textContent = '下载中...';
      try {
        await api(`/models/${encodeURIComponent(id)}/download`, { method: 'POST' });
        await refreshModels();
        toast('模型下载完成', '现在可以将模型加载到内存。', 'success');
      } catch (error) {
        toast('模型下载失败', error.message, 'error');
        button.disabled = false;
      }
      return;
    }
    if (action === 'delete') {
      if (!window.confirm(`确定要删除本地模型“${model?.name || id}”吗？模型文件会从项目的 models/ 目录移除。`)) return;
      button.disabled = true;
      try {
        await api(`/models/${encodeURIComponent(id)}`, { method: 'DELETE' });
        toast('模型已删除', '本地模型文件已从项目目录移除。', 'success');
        await refreshModels();
      } catch (error) { toast('删除模型失败', error.message, 'error'); button.disabled = false; }
      return;
    }
    button.disabled = true;
    button.textContent = action === 'load' ? '加载中...' : '卸载中...';
    try {
      await api(`/models/${encodeURIComponent(id)}/${action}`, action === 'load' ? { method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/json' }, body: JSON.stringify({ device: state.preferences.defaultDevice || 'auto' }) } : { method: 'POST' });
      toast(action === 'load' ? '模型已加载' : '模型已卸载', action === 'load' ? '现在可以用于新的识别任务。' : '显存和内存已经释放。', 'success');
      await refreshModels();
    } catch (error) {
      toast(action === 'load' ? '加载模型失败' : '卸载模型失败', error.message, 'error');
      button.disabled = false;
    }
  }

  function openModelDialog(model) {
    if (model) {
      $('#model-name-input').value = model.name || '';
      $('#model-id-input').value = model.id || model.source || '';
      $('#model-size-input').value = ['tiny', 'base', 'small', 'medium', 'large-v3'].includes(String(model.size)) ? model.size : 'custom';
      $('#model-backend-input').value = ['openai-whisper', 'transformers'].includes(model.backend) ? model.backend : 'openai-whisper';
      $('#model-source-type-input').value = sourceTypeFor(model.source);
      $('#model-notes-input').value = model.notes || '';
    } else {
      el.modelForm.reset();
      $('#model-size-input').value = 'large-v3';
      $('#model-backend-input').value = 'openai-whisper';
      $('#model-source-type-input').value = 'auto';
    }
    el.modelDialog.showModal();
    window.setTimeout(() => $('#model-name-input').focus(), 30);
  }

  function openModelDetails(model) {
    if (!model) return;
    el.detailTitle.textContent = model.name;
    const statusText = { loaded: '已加载到内存', downloaded: '已下载', remote: '待下载', error: '需要处理' }[model.status] || model.status;
    el.detailBody.innerHTML = `<div class="detail-list">
      <div class="detail-row"><span>模型 ID</span><code>${escapeHTML(model.id)}</code></div>
      <div class="detail-row"><span>来源</span><code>${escapeHTML(model.source)}</code></div>
      <div class="detail-row"><span>状态</span><strong>${escapeHTML(statusText)}</strong></div>
      <div class="detail-row"><span>模型大小</span><strong>${escapeHTML(String(model.size))}${model.bytes ? ` (${formatBytes(model.bytes)})` : ''}</strong></div>
      <div class="detail-row"><span>推理后端</span><strong>${escapeHTML(model.backend)}</strong></div>
      <div class="detail-row"><span>本地路径</span><code>${escapeHTML(model.path || '下载后显示')}</code></div>
      ${model.notes ? `<div class="detail-row"><span>备注</span><strong>${escapeHTML(model.notes)}</strong></div>` : ''}
    </div><div class="detail-actions"><button class="secondary-btn" id="detail-close" type="button">关闭</button></div>`;
    $('#detail-close').addEventListener('click', () => el.detailsDialog.close());
    el.detailsDialog.showModal();
  }

  function syncModelEngineForSource() {
    const selectedType = $('#model-source-type-input').value;
    const sourceType = selectedType === 'auto' ? sourceTypeFor($('#model-id-input').value) : selectedType;
    if (sourceType === 'huggingface') $('#model-backend-input').value = 'transformers';
    if (sourceType === 'url') $('#model-backend-input').value = 'openai-whisper';
  }

  async function submitModel(event) {
    const submitter = event.submitter;
    if (submitter?.value === 'cancel') return;
    event.preventDefault();
    const source = $('#model-id-input').value.trim();
    const sourceType = $('#model-source-type-input').value === 'auto' ? sourceTypeFor(source) : $('#model-source-type-input').value;
    const selectedEngine = $('#model-backend-input').value;
    // Hugging Face repositories are Transformers checkpoints; direct checkpoint URLs are OpenAI Whisper files.
    const engine = sourceType === 'huggingface' ? 'transformers' : sourceType === 'url' ? 'openai-whisper' : selectedEngine;
    const payload = {
      name: $('#model-name-input').value.trim(),
      id: safeModelId(source),
      engine,
      source_type: sourceType,
      size: $('#model-size-input').value,
      notes: $('#model-notes-input').value.trim(),
    };
    if (sourceType === 'huggingface') payload.hf_repo = source;
    if (sourceType === 'url') payload.url = source;
    if (sourceType === 'local') payload.local_path = source;
    if (!payload.name || !source) return;
    const button = $('#submit-model-btn');
    button.disabled = true;
    button.innerHTML = '<span class="button-icon">◌</span>正在提交...';
    try {
      const created = await api('/models', { method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/json' }, body: JSON.stringify(payload) });
      el.modelDialog.close();
      const createdId = created.id || created.model_id || created.model?.id || payload.id;
      await api(`/models/${encodeURIComponent(createdId)}/download`, { method: 'POST' });
      toast('模型下载完成', '模型已保存到项目目录，现在可以加载使用。', 'success');
      await refreshModels();
    } catch (error) {
      toast('添加模型失败', error.message, 'error');
    } finally {
      button.disabled = false;
      button.innerHTML = '<span class="button-icon">↓</span>开始下载';
    }
  }

  function addFiles(fileList) {
    const added = Array.from(fileList || []).filter((file) => {
      if (!file || !file.size) return false;
      const duplicate = state.files.some((existing) => existing.file.name === file.name && existing.file.size === file.size && existing.file.lastModified === file.lastModified);
      return !duplicate;
    }).map((file) => ({ key: `${file.name}-${file.size}-${file.lastModified}-${Math.random().toString(36).slice(2)}`, file }));
    if (!added.length) return;
    state.files.push(...added);
    renderQueue();
    toast('已添加音频', `队列中共有 ${state.files.length} 个文件。`, 'success');
  }

  function renderQueue() {
    el.queue.hidden = state.files.length === 0;
    el.queueItems.innerHTML = state.files.map(({ key, file }) => {
      const type = (file.name.split('.').pop() || 'audio').slice(0, 4).toUpperCase();
      return `<div class="queue-item" data-file-key="${escapeHTML(key)}"><span class="file-type">${escapeHTML(type)}</span><div class="queue-file-meta"><div class="queue-file-name" title="${escapeHTML(file.name)}">${escapeHTML(file.name)}</div><span class="queue-file-size">${formatBytes(file.size)}</span></div><button class="queue-remove" type="button" title="移除文件">×</button></div>`;
    }).join('');
    $$('.queue-remove', el.queueItems).forEach((button) => button.addEventListener('click', () => {
      const key = button.closest('[data-file-key]').dataset.fileKey;
      state.files = state.files.filter((item) => item.key !== key);
      renderQueue();
    }));
    updateTranscribeButton();
  }

  function updateTranscribeButton() {
    el.transcribe.disabled = !state.files.length || !el.modelSelect.value || Boolean(state.currentJob);
    const label = el.transcribe.querySelector('span:nth-child(2)');
    if (label && !state.currentJob) label.textContent = state.files.length > 1 ? `批量识别 ${state.files.length} 个文件` : '开始识别';
  }

  function optionalNumber(id) {
    const value = $(`#${id}`).value.trim();
    return value === '' ? undefined : Number(value);
  }

  function temperatureValue() {
    const value = $('#temperature').value.trim();
    if (!value) return undefined;
    return value.includes(',') ? value : Number(value);
  }

  function selectedOutputFormats() {
    return outputFormatList($('#output-format').value).join(',');
  }

  function collectOptions() {
    const language = $('#language-select').value;
    const verbose = $('#verbose-select').value;
    const initialPrompt = $('#initial-prompt').value.trim();
    const hotwords = $('#hotwords').value.trim();
    return compactObject({
      model_id: el.modelSelect.value,
      language: language === 'auto' ? undefined : language,
      task: $('#task-select').value,
      temperature: temperatureValue(),
      beam_size: optionalNumber('beam-size'),
      best_of: optionalNumber('best-of'),
      patience: optionalNumber('patience'),
      length_penalty: optionalNumber('length-penalty'),
      sample_len: optionalNumber('sample-len'),
      compression_ratio_threshold: optionalNumber('compression-ratio'),
      logprob_threshold: optionalNumber('logprob-threshold'),
      no_speech_threshold: optionalNumber('no-speech-threshold'),
      max_initial_timestamp: optionalNumber('max-initial-timestamp'),
      hallucination_silence_threshold: optionalNumber('hallucination-silence'),
      clip_timestamps: $('#clip-timestamps').value.trim() || undefined,
      initial_prompt: [initialPrompt, hotwords].filter(Boolean).join(initialPrompt && hotwords ? '\n术语：' : '') || undefined,
      prompt: $('#prompt').value.trim() || undefined,
      prefix: $('#prefix').value.trim() || undefined,
      suppress_tokens: $('#suppress-tokens').value.trim() || undefined,
      prepend_punctuations: $('#prepend-punctuations').value.trim() || undefined,
      append_punctuations: $('#append-punctuations').value.trim() || undefined,
      verbose: verbose === 'live' ? 'live' : verbose === 'true',
      threads: optionalNumber('threads'),
      word_timestamps: $('#word-timestamps').checked,
      condition_on_previous_text: $('#condition-previous').checked,
      fp16: $('#fp16').checked,
      suppress_blank: $('#suppress-blank').checked,
      without_timestamps: $('#without-timestamps').checked,
      carry_initial_prompt: $('#carry-initial-prompt').checked,
    });
  }

  function compactObject(object) {
    return Object.fromEntries(Object.entries(object).filter(([, value]) => value !== undefined && value !== null && value !== ''));
  }

  async function startTranscription() {
    if (!state.files.length || !el.modelSelect.value || state.currentJob) return;
    const isBatch = state.files.length > 1;
    const form = new FormData();
    if (isBatch) state.files.forEach(({ file }) => form.append('files', file, file.name));
    else form.append('file', state.files[0].file, state.files[0].file.name);
    const options = collectOptions();
    form.append('model_id', el.modelSelect.value);
    delete options.model_id;
    form.append('options', JSON.stringify(options));
    form.append('output_formats', selectedOutputFormats());
    setJobProgress(true, 4, isBatch ? `正在提交 ${state.files.length} 个文件...` : '正在上传音频...');
    state.currentJob = { id: null, type: 'transcription', files: state.files.slice() };
    updateTranscribeButton();
    try {
      const result = await api(isBatch ? '/transcriptions/batch' : '/transcriptions', { method: 'POST', body: form });
      const jobId = getJobId(result);
      if (jobId) {
        state.currentJob.id = jobId;
        setJobProgress(true, getProgress(result, 8), getJobMessage(result) || '任务已进入队列...');
        watchJob(jobId, { type: 'transcription' });
      } else {
        finishTranscription(result);
      }
    } catch (error) {
      state.currentJob = null;
      setJobProgress(false);
      updateTranscribeButton();
      toast('识别任务未能启动', error.message, 'error');
    }
  }

  function getJobId(data) {
    if (!data || typeof data !== 'object') return null;
    return data.job_id || data.id || data.jobId || data.task_id || data.taskId || (data.job && (data.job.id || data.job.job_id)) || null;
  }

  function getProgress(data) {
    const raw = data?.progress ?? data?.percentage ?? data?.percent ?? data?.job?.progress;
    const numeric = Number(raw);
    if (!Number.isFinite(numeric)) return 0;
    return numeric <= 1 ? Math.round(numeric * 100) : Math.round(numeric);
  }

  function getJobMessage(data) {
    return data?.message || data?.status_message || data?.detail || data?.job?.message || '';
  }

  function jobIsDone(data) {
    const status = String(data?.status || data?.state || data?.job?.status || '').toLowerCase();
    return ['completed', 'completed_with_errors', 'complete', 'done', 'succeeded', 'success', 'finished'].includes(status) || Boolean(data?.result && !['queued', 'running', 'processing'].includes(status));
  }

  function jobFailed(data) {
    const status = String(data?.status || data?.state || data?.job?.status || '').toLowerCase();
    return ['failed', 'error', 'cancelled', 'canceled'].includes(status);
  }

  function watchJob(jobId, context) {
    window.clearTimeout(state.jobTimer);
    const poll = async () => {
      try {
        const data = await api(`/jobs/${encodeURIComponent(jobId)}`);
        const percent = getProgress(data);
        if (context.type === 'transcription') setJobProgress(true, percent || 10, getJobMessage(data) || '正在识别语音...');
        if (jobFailed(data)) {
          state.currentJob = null;
          setJobProgress(false);
          updateTranscribeButton();
          toast('识别任务失败', data.error || data.message || '任务未能完成。', 'error');
          return;
        }
        if (jobIsDone(data)) {
          finishTranscription(data.result || data);
          return;
        }
        state.jobTimer = window.setTimeout(poll, 1300);
      } catch (error) {
        if (context.type === 'transcription') {
          state.currentJob = null;
          setJobProgress(false);
          updateTranscribeButton();
        }
        toast('无法读取任务进度', error.message, 'error');
      }
    };
    state.jobTimer = window.setTimeout(poll, 700);
  }

  async function finishTranscription(result) {
    window.clearTimeout(state.jobTimer);
    state.currentJob = null;
    setJobProgress(true, 100, '识别完成');
    window.setTimeout(() => setJobProgress(false), 800);
    state.currentResult = normaliseResult(result);
    renderResult();
    state.files = [];
    renderQueue();
    updateTranscribeButton();
    toast('识别完成', '结果已保存到项目工作区。', 'success');
    await refreshHistory();
  }

  function normaliseResult(raw) {
    raw = raw || {};
    const result = raw.result && typeof raw.result === 'object' ? raw.result : raw;
    const records = Array.isArray(result.results) ? result.results : Array.isArray(raw.results) ? raw.results : [];
    const text = result.text || result.transcript || result.transcription || result.output_text || records.map((item) => item.text || item.result?.text || '').filter(Boolean).join('\n\n') || '';
    const segments = Array.isArray(result.segments) ? result.segments : Array.isArray(result.chunks) ? result.chunks : [];
    const outputs = result.outputs || result.output_urls || raw.output_urls || result.output_files || result.files || result.downloads || raw.downloads || {};
    return {
      text: typeof text === 'string' ? text : String(text || ''),
      segments,
      raw: result,
      outputs,
      records,
      language: result.language || result.detected_language,
      duration: result.duration || result.duration_seconds,
      model: result.model || result.model_id,
      filename: result.filename || result.file_name || raw.filename || state.files[0]?.file?.name || 'transcript',
    };
  }

  function renderResult() {
    const result = state.currentResult;
    if (!result) return;
    el.resultPanel.hidden = false;
    const tags = [result.filename, result.language, result.duration ? formatDuration(result.duration) : '', result.model].filter(Boolean);
    el.resultMeta.textContent = tags.join(' · ') || '任务完成';
    el.resultText.textContent = result.text || '后端未返回纯文本结果。';
    el.resultJson.textContent = JSON.stringify(result.raw, null, 2);
    if (result.segments.length) {
      el.resultSegments.innerHTML = result.segments.map((segment, index) => {
        const start = formatDuration(segment.start ?? segment.start_time ?? 0);
        const end = formatDuration(segment.end ?? segment.end_time ?? 0);
        const value = segment.text || segment.transcript || segment.content || '';
        return `<div class="segment-item"><span class="segment-time">${start} - ${end}</span><span class="segment-text">${escapeHTML(value)}</span></div>`;
      }).join('');
    } else {
      el.resultSegments.innerHTML = '<div class="empty-state" style="min-height:120px"><span class="empty-icon">⌁</span><h3>没有时间轴数据</h3><p>请在高级参数中开启词级时间戳，或使用支持 segments 的后端。</p></div>';
    }
    activateResultTab('text');
    el.resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function activateResultTab(tab) {
    $$('.result-tab').forEach((button) => button.classList.toggle('active', button.dataset.resultTab === tab));
    el.resultText.hidden = tab !== 'text';
    el.resultSegments.hidden = tab !== 'segments';
    el.resultJson.hidden = tab !== 'json';
  }

  function setJobProgress(show, percent, message) {
    el.jobProgress.hidden = !show;
    if (show) {
      const value = Math.max(0, Math.min(100, Number(percent) || 0));
      el.progressFill.style.width = `${value}%`;
      el.progressPercent.textContent = `${value}%`;
      el.progressStatus.textContent = message || '正在处理...';
    }
  }

  async function refreshHistory() {
    try {
      const data = await api('/history');
      state.history = unwrapList(data, ['history', 'items', 'results', 'data']);
      renderHistory();
      const today = new Date().toDateString();
      $('#today-count').textContent = state.history.filter((item) => new Date(item.completed_at || item.created_at || item.timestamp).toDateString() === today).length;
    } catch (error) {
      state.history = [];
      renderHistory(error.message);
      console.warn('History list failed:', error);
    }
  }

  function normaliseHistory(raw, index) {
    raw = raw || {};
    const id = raw.id || raw.history_id || raw.job_id || `history-${index}`;
    const status = String(raw.status || raw.state || 'completed').toLowerCase();
    return {
      id: String(id),
      filename: raw.filename || raw.file_name || raw.name || raw.files?.[0]?.name || (raw.kind === 'batch' ? '批量识别任务' : `未命名任务 ${index + 1}`),
      status: ['done', 'success', 'succeeded'].includes(status) ? 'completed' : status,
      model: raw.model || raw.model_name || raw.model_id || '--',
      language: raw.language || raw.detected_language || '--',
      duration: raw.duration || raw.duration_seconds,
      date: raw.completed_at || raw.finished_at || raw.created_at || raw.timestamp,
      text: raw.text || raw.transcript || '',
      raw,
    };
  }

  function renderHistory(errorMessage) {
    const search = ($('#history-search').value || '').trim().toLowerCase();
    const dateFilter = $('#history-date-filter').value;
    const now = Date.now();
    let items = state.history.map(normaliseHistory).filter((item) => {
      const text = `${item.filename} ${item.text} ${item.model}`.toLowerCase();
      if (search && !text.includes(search)) return false;
      if (dateFilter === 'all' || !item.date) return true;
      const age = now - new Date(item.date).getTime();
      if (dateFilter === 'today') return new Date(item.date).toDateString() === new Date().toDateString();
      if (dateFilter === 'week') return age <= 7 * 86400000;
      if (dateFilter === 'month') return age <= 30 * 86400000;
      return true;
    });
    items.sort((a, b) => new Date(b.date || 0) - new Date(a.date || 0));
    const pageSize = 9;
    const pages = Math.max(1, Math.ceil(items.length / pageSize));
    state.historyPage = Math.min(state.historyPage, pages);
    const start = (state.historyPage - 1) * pageSize;
    const pageItems = items.slice(start, start + pageSize);
    el.historySummary.textContent = errorMessage ? '无法读取记录' : `共 ${items.length} 条记录`;
    if (errorMessage) {
      el.historyBody.innerHTML = `<tr class="table-empty"><td colspan="7"><span class="empty-icon">!</span><strong>无法读取识别历史</strong><small>${escapeHTML(errorMessage)}</small></td></tr>`;
      el.historyPagination.innerHTML = '';
      return;
    }
    if (!pageItems.length) {
      el.historyBody.innerHTML = `<tr class="table-empty"><td colspan="7"><span class="empty-icon">◷</span><strong>${items.length ? '没有符合条件的记录' : '还没有识别记录'}</strong><small>${items.length ? '调整搜索条件后重试' : '完成一项识别任务后，结果会出现在这里'}</small></td></tr>`;
    } else {
      el.historyBody.innerHTML = pageItems.map((item) => {
        const extension = (item.filename.split('.').pop() || 'AUD').slice(0, 4).toUpperCase();
        const statusText = { completed: '已完成', processing: '处理中', queued: '队列中', failed: '失败', error: '失败' }[item.status] || item.status;
        return `<tr data-history-id="${escapeHTML(item.id)}"><td><div class="history-file"><span class="history-file-icon">${escapeHTML(extension)}</span><div><strong title="${escapeHTML(item.filename)}">${escapeHTML(item.filename)}</strong><span>${item.text ? escapeHTML(item.text.slice(0, 48)) : '无摘要'}</span></div></div></td><td><span class="table-status ${escapeHTML(item.status)}">${escapeHTML(statusText)}</span></td><td>${escapeHTML(item.model)}</td><td>${escapeHTML(item.language)}</td><td>${formatDuration(item.duration)}</td><td>${escapeHTML(formatDate(item.date))}</td><td><button class="row-action" data-history-download title="下载结果" type="button">↓</button></td></tr>`;
      }).join('');
      $$('[data-history-download]').forEach((button) => button.addEventListener('click', () => downloadHistory(button.closest('tr').dataset.historyId)));
    }
    el.historyPagination.innerHTML = pages > 1 ? Array.from({ length: pages }, (_, index) => `<button class="page-btn ${index + 1 === state.historyPage ? 'active' : ''}" data-page="${index + 1}" type="button">${index + 1}</button>`).join('') : '';
    $$('.page-btn', el.historyPagination).forEach((button) => button.addEventListener('click', () => { state.historyPage = Number(button.dataset.page); renderHistory(); }));
  }

  async function downloadHistory(id) {
    try {
      await downloadFromEndpoint(`/history/export?id=${encodeURIComponent(id)}`, `whisperdock-${id}.zip`);
    } catch (error) {
      toast('下载历史结果失败', error.message, 'error');
    }
  }

  async function downloadCurrentResult() {
    const result = state.currentResult;
    if (!result) return;
    const output = pickOutput(result.outputs, $('#output-format').value);
    if (typeof output === 'string' && /^https?:|^\//.test(output)) {
      window.open(output, '_blank', 'noopener');
      return;
    }
    if (output && typeof output === 'object' && output.url) {
      window.open(output.url, '_blank', 'noopener');
      return;
    }
    const filename = `${slugFromFile({ name: result.filename || 'transcript' })}.txt`;
    downloadBlob(new Blob([result.text || ''], { type: 'text/plain;charset=utf-8' }), filename);
    toast('已准备下载', '浏览器已下载纯文本结果。', 'success');
  }

  function pickOutput(outputs, format) {
    if (!outputs) return null;
    if (Array.isArray(outputs)) return outputs.find((item) => String(item.format || item.name || '').includes(format)) || outputs[0];
    return outputs[format] || outputs.txt || outputs.url || null;
  }

  async function downloadFromEndpoint(path, fallbackName) {
    const response = await fetch(`${API}${path}`);
    if (!response.ok) {
      const data = await response.json().catch(() => null);
      throw new Error(data?.detail || data?.message || `下载失败 (${response.status})`);
    }
    const disposition = response.headers.get('content-disposition') || '';
    const match = disposition.match(/filename\*?=(?:UTF-8''|\")?([^;\"]+)/i);
    const filename = match ? decodeURIComponent(match[1].replace(/^"|"$/g, '')) : fallbackName;
    downloadBlob(await response.blob(), filename);
    toast('下载已开始', filename, 'success');
  }

  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function copyResult() {
    if (!state.currentResult?.text) return;
    try {
      await navigator.clipboard.writeText(state.currentResult.text);
      toast('已复制', '识别文本已复制到剪贴板。', 'success');
    } catch (_) {
      const area = document.createElement('textarea');
      area.value = state.currentResult.text;
      document.body.appendChild(area);
      area.select();
      document.execCommand('copy');
      area.remove();
      toast('已复制', '识别文本已复制到剪贴板。', 'success');
    }
  }

  function startRecording() {
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      toast('浏览器不支持录音', '请使用最新版 Chrome、Edge、Firefox 或 Safari，或直接上传音频文件。', 'error');
      return;
    }
    navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
      const options = chooseRecorderOptions();
      const recorder = new MediaRecorder(stream, options);
      const chunks = [];
      recorder.addEventListener('dataavailable', (event) => { if (event.data.size) chunks.push(event.data); });
      recorder.addEventListener('stop', () => {
        const type = recorder.mimeType || 'audio/webm';
        const extension = recorderExtension(type);
        const file = new File([new Blob(chunks, { type })], `recording-${new Date().toISOString().replace(/[:.]/g, '-')}.${extension}`, { type });
        stream.getTracks().forEach((track) => track.stop());
        stopRecorderClock();
        state.recording = null;
        el.recordBtn.classList.remove('recording');
        el.recordLabel.textContent = '开始录音';
        addFiles([file]);
        toast('录音已加入队列', '可以调整参数后开始识别。', 'success');
      });
      recorder.start(350);
      state.recording = { recorder, stream };
      state.recorderStartedAt = Date.now();
      el.recordBtn.classList.add('recording');
      el.recordLabel.textContent = '停止录音';
      startRecorderClock();
      toast('正在录音', '再次点击“停止录音”即可将音频加入队列。', 'info');
    }).catch((error) => toast('无法使用麦克风', error.message || '请在浏览器中允许麦克风权限。', 'error'));
  }

  function chooseRecorderOptions() {
    const types = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/mp4'];
    const mimeType = types.find((type) => MediaRecorder.isTypeSupported(type));
    return mimeType ? { mimeType } : undefined;
  }

  function recorderExtension(type) {
    return String(type || '').includes('mp4') ? 'm4a' : String(type || '').includes('ogg') ? 'ogg' : 'webm';
  }

  function stopRecording() {
    if (state.recording?.recorder?.state !== 'inactive') state.recording.recorder.stop();
  }

  function setRecordMode(mode) {
    if (!['file', 'realtime'].includes(mode) || state.recordMode === mode) return;
    if (state.recording || state.realtime) {
      toast('请先结束当前录音', '录音结束后可以切换模式。', 'info');
      return;
    }
    state.recordMode = mode;
    $$('[data-record-mode]').forEach((button) => button.classList.toggle('active', button.dataset.recordMode === mode));
    el.recordLabel.textContent = mode === 'realtime' ? '开始实时识别' : '开始录音';
    el.livePanel.hidden = mode !== 'realtime';
    if (mode === 'realtime') renderLiveTranscript('准备分段识别');
  }

  async function startRealtimeRecording() {
    if (!el.modelSelect.value) {
      toast('需要先加载模型', '请先在模型管理中加载一个模型到内存。', 'error');
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      toast('浏览器不支持录音', '请使用最新版 Chrome、Edge、Firefox 或 Safari，或直接上传音频文件。', 'error');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const session = {
        id: ++state.realtimeSequence,
        stream,
        active: true,
        recorder: null,
        timer: null,
        pending: 0,
        finalised: false,
      };
      state.realtime = session;
      state.liveTranscript = '';
      el.livePanel.hidden = false;
      renderLiveTranscript('正在监听，第一段结果很快出现');
      state.recorderStartedAt = Date.now();
      el.recordBtn.classList.add('recording');
      el.recordLabel.textContent = '停止实时识别';
      startRecorderClock();
      startRealtimeSegment(session);
      toast('已开始分段实时识别', '音频会以独立短片段在本地顺序处理。', 'success');
    } catch (error) {
      toast('无法使用麦克风', error.message || '请在浏览器中允许麦克风权限。', 'error');
    }
  }

  function startRealtimeSegment(session) {
    if (!session.active || state.realtime !== session) return;
    const chunks = [];
    let recorder;
    try {
      recorder = new MediaRecorder(session.stream, chooseRecorderOptions());
    } catch (error) {
      finishRealtimeSession(session, error);
      return;
    }
    session.recorder = recorder;
    recorder.addEventListener('dataavailable', (event) => { if (event.data.size) chunks.push(event.data); });
    recorder.addEventListener('stop', () => {
      window.clearTimeout(session.timer);
      session.timer = null;
      const type = recorder.mimeType || 'audio/webm';
      const blob = new Blob(chunks, { type });
      if (blob.size > 0) sendRealtimeChunk(blob, type, session);
      if (session.active) {
        session.timer = window.setTimeout(() => startRealtimeSegment(session), 80);
      } else {
        session.stream.getTracks().forEach((track) => track.stop());
        session.recorder = null;
        renderLiveTranscript(session.pending ? '正在完成最后的片段' : '实时识别已停止');
        maybeFinishRealtimeSession(session);
      }
    });
    recorder.start();
    session.timer = window.setTimeout(() => {
      if (recorder.state !== 'inactive') recorder.stop();
    }, 3800);
  }

  function stopRealtimeRecording() {
    const session = state.realtime;
    if (!session || !session.active) return;
    session.active = false;
    window.clearTimeout(session.timer);
    if (session.recorder?.state !== 'inactive') session.recorder.stop();
    else {
      session.stream.getTracks().forEach((track) => track.stop());
      maybeFinishRealtimeSession(session);
    }
    el.recordBtn.classList.remove('recording');
    el.recordBtn.disabled = true;
    el.recordLabel.textContent = '正在整理...';
    stopRecorderClock();
  }

  async function sendRealtimeChunk(blob, type, session) {
    if (!blob.size) return;
    session.pending += 1;
    renderLiveTranscript(`正在识别第 ${session.pending} 个片段`);
    const form = new FormData();
    const extension = recorderExtension(type);
    form.append('file', new File([blob], `realtime-${session.id}-${Date.now()}.${extension}`, { type }), `realtime-${session.id}.${extension}`);
    form.append('model_id', el.modelSelect.value);
    const options = collectOptions();
    delete options.model_id;
    form.append('options', JSON.stringify(options));
    form.append('output_formats', 'txt,json');
    try {
      const submitted = await api('/realtime/chunks', { method: 'POST', body: form });
      const jobId = getJobId(submitted);
      if (jobId) pollRealtimeJob(jobId, session);
      else {
        appendRealtimeResult(normaliseResult(submitted), session);
        session.pending -= 1;
        maybeFinishRealtimeSession(session);
      }
    } catch (error) {
      session.pending -= 1;
      toast('实时片段识别失败', error.message, 'error');
      maybeFinishRealtimeSession(session);
    }
  }

  function pollRealtimeJob(jobId, session) {
    const poll = async () => {
      try {
        const data = await api(`/jobs/${encodeURIComponent(jobId)}`);
        if (jobFailed(data)) {
          session.pending -= 1;
          toast('实时片段识别失败', data.error?.message || data.message || '该片段未能完成。', 'error');
          maybeFinishRealtimeSession(session);
          return;
        }
        if (jobIsDone(data)) {
          session.pending -= 1;
          appendRealtimeResult(normaliseResult(data.result || data), session);
          maybeFinishRealtimeSession(session);
          return;
        }
        window.setTimeout(poll, 750);
      } catch (error) {
        session.pending -= 1;
        toast('无法读取实时结果', error.message, 'error');
        maybeFinishRealtimeSession(session);
      }
    };
    window.setTimeout(poll, 400);
  }

  function appendRealtimeResult(result, session) {
    if (state.realtime !== session || !result.text) return;
    state.liveTranscript = `${state.liveTranscript}${state.liveTranscript ? '\n' : ''}${result.text}`;
    renderLiveTranscript(session.active ? '正在监听下一段音频' : '正在完成最后的片段');
  }

  function renderLiveTranscript(status) {
    el.livePanel.hidden = false;
    el.liveStatus.textContent = status || '分段实时识别';
    const session = state.realtime;
    el.livePending.textContent = session?.pending ? `${session.pending} 个片段处理中` : '';
    el.liveTranscript.textContent = state.liveTranscript || '实时结果会显示在这里。';
  }

  function maybeFinishRealtimeSession(session) {
    if (session.active || session.pending > 0 || session.finalised || state.realtime !== session) return;
    session.finalised = true;
    state.realtime = null;
    el.recordBtn.disabled = false;
    el.recordLabel.textContent = '开始实时识别';
    el.recordBtn.classList.remove('recording');
    renderLiveTranscript(state.liveTranscript ? '实时识别已完成' : '实时识别已停止');
    if (state.liveTranscript) {
      state.currentResult = {
        text: state.liveTranscript,
        segments: [],
        raw: { text: state.liveTranscript, mode: 'segmented-realtime' },
        outputs: {},
        filename: 'realtime-transcript',
      };
      renderResult();
      refreshHistory();
    }
  }

  function finishRealtimeSession(session, error) {
    session.active = false;
    session.stream?.getTracks().forEach((track) => track.stop());
    state.realtime = null;
    el.recordBtn.disabled = false;
    el.recordBtn.classList.remove('recording');
    el.recordLabel.textContent = '开始实时识别';
    stopRecorderClock();
    if (error) toast('无法开始实时识别', error.message || '浏览器无法创建音频片段。', 'error');
  }

  function startRecorderClock() {
    stopRecorderClock();
    const tick = () => { el.recordTimer.textContent = formatDuration((Date.now() - state.recorderStartedAt) / 1000); };
    tick();
    state.recorderTimer = window.setInterval(tick, 500);
  }

  function stopRecorderClock() {
    window.clearInterval(state.recorderTimer);
    state.recorderTimer = null;
    el.recordTimer.textContent = '00:00';
  }

  function switchView(name) {
    const panel = $(`[data-view-panel="${name}"]`);
    if (!panel) return;
    $$('.nav-item').forEach((button) => button.classList.toggle('active', button.dataset.view === name));
    $$('.view').forEach((view) => view.classList.toggle('active', view === panel));
    const names = { studio: '识别工作台', models: '模型管理', history: '识别历史', settings: '设置与帮助' };
    $('#breadcrumb-section').textContent = names[name];
    $('#breadcrumb-current').textContent = name === 'studio' ? '新建任务' : names[name];
    el.sidebar.classList.remove('open');
    if (name === 'models') refreshModels();
    if (name === 'history') refreshHistory();
  }

  function applyPreferencesToForm() {
    $('#keep-temp').checked = state.preferences.keepTemp;
    $('#check-models').checked = state.preferences.checkModels;
    $('#default-language').value = state.preferences.defaultLanguage;
    $('#default-format').value = state.preferences.defaultFormat;
    $('#default-device').value = state.preferences.defaultDevice || 'auto';
    $('#default-prompt').value = state.preferences.defaultPrompt;
    $('#language-select').value = state.preferences.defaultLanguage;
    $('#output-format').value = state.preferences.defaultFormat;
    $('#initial-prompt').value = state.preferences.defaultPrompt;
  }

  function bindEvents() {
    $$('.nav-item').forEach((button) => button.addEventListener('click', () => switchView(button.dataset.view)));
    $$('[data-view-link]').forEach((button) => button.addEventListener('click', () => switchView(button.dataset.viewLink)));
    $('#mobile-menu').addEventListener('click', () => el.sidebar.classList.toggle('open'));
    $('#refresh-btn').addEventListener('click', async () => { await Promise.all([refreshHealth(), refreshModels(), refreshHistory()]); toast('数据已刷新', '本地服务状态、模型和历史记录已更新。', 'success'); });
    $('#model-refresh-btn').addEventListener('click', refreshModels);
    $('#manage-models-btn').addEventListener('click', () => switchView('models'));
    $('#add-model-btn').addEventListener('click', () => openModelDialog());
    el.modelForm.addEventListener('submit', submitModel);
    $('#model-source-type-input').addEventListener('change', syncModelEngineForSource);
    $('#model-id-input').addEventListener('input', () => {
      if ($('#model-source-type-input').value === 'auto') syncModelEngineForSource();
    });
    $('#close-details-btn').addEventListener('click', () => el.detailsDialog.close());
    $('#browse-btn').addEventListener('click', (event) => { event.stopPropagation(); el.audioInput.click(); });
    el.dropZone.addEventListener('click', () => el.audioInput.click());
    el.dropZone.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); el.audioInput.click(); } });
    el.audioInput.addEventListener('change', () => { addFiles(el.audioInput.files); el.audioInput.value = ''; });
    ['dragenter', 'dragover'].forEach((eventName) => el.dropZone.addEventListener(eventName, (event) => { event.preventDefault(); el.dropZone.classList.add('dragging'); }));
    ['dragleave', 'drop'].forEach((eventName) => el.dropZone.addEventListener(eventName, (event) => { event.preventDefault(); el.dropZone.classList.remove('dragging'); }));
    el.dropZone.addEventListener('drop', (event) => addFiles(event.dataTransfer.files));
    $('#clear-queue-btn').addEventListener('click', () => { state.files = []; renderQueue(); });
    $$('[data-record-mode]').forEach((button) => button.addEventListener('click', () => setRecordMode(button.dataset.recordMode)));
    el.recordBtn.addEventListener('click', () => {
      if (state.realtime) { stopRealtimeRecording(); return; }
      if (state.recording) { stopRecording(); return; }
      if (state.recordMode === 'realtime') startRealtimeRecording();
      else startRecording();
    });
    $('#advanced-toggle').addEventListener('click', () => {
      const button = $('#advanced-toggle');
      const expanded = button.getAttribute('aria-expanded') === 'true';
      button.setAttribute('aria-expanded', String(!expanded));
      $('#advanced-fields').hidden = expanded;
      $('span', button).textContent = expanded ? '展开高级参数' : '收起高级参数';
    });
    el.transcribe.addEventListener('click', startTranscription);
    $('#cancel-job-btn').addEventListener('click', () => {
      window.clearTimeout(state.jobTimer);
      state.currentJob = null;
      setJobProgress(false);
      updateTranscribeButton();
      toast('已停止监听页面进度', '服务器中的任务可能仍在继续，可稍后在历史记录中查看。', 'info');
    });
    $$('.result-tab').forEach((button) => button.addEventListener('click', () => activateResultTab(button.dataset.resultTab)));
    $('#copy-result-btn').addEventListener('click', copyResult);
    $('#download-result-btn').addEventListener('click', downloadCurrentResult);
    $('#model-search').addEventListener('input', renderModels);
    $('#model-filter').addEventListener('change', renderModels);
    $('#history-search').addEventListener('input', () => { state.historyPage = 1; renderHistory(); });
    $('#history-date-filter').addEventListener('change', () => { state.historyPage = 1; renderHistory(); });
    $('#export-history-btn').addEventListener('click', async () => {
      try { await downloadFromEndpoint('/history/export', 'whisperdock-history.zip'); } catch (error) { toast('导出历史失败', error.message, 'error'); }
    });
    $('#keep-temp').addEventListener('change', (event) => { state.preferences.keepTemp = event.target.checked; scheduleProjectSettingsSave(); });
    $('#check-models').addEventListener('change', (event) => { state.preferences.checkModels = event.target.checked; });
    $('#default-language').addEventListener('change', (event) => { state.preferences.defaultLanguage = event.target.value; $('#language-select').value = event.target.value; scheduleProjectSettingsSave(); });
    $('#default-format').addEventListener('change', (event) => { state.preferences.defaultFormat = event.target.value; $('#output-format').value = event.target.value; scheduleProjectSettingsSave(); });
    $('#default-device').addEventListener('change', (event) => { state.preferences.defaultDevice = event.target.value; scheduleProjectSettingsSave(); });
    $('#default-prompt').addEventListener('input', (event) => { state.preferences.defaultPrompt = event.target.value; $('#initial-prompt').value = event.target.value; scheduleProjectSettingsSave(); });
    $('#reset-preferences').addEventListener('click', () => { state.preferences = { keepTemp: true, checkModels: true, defaultLanguage: 'auto', defaultFormat: 'all', defaultDevice: 'auto', defaultPrompt: '' }; applyPreferencesToForm(); scheduleProjectSettingsSave(); toast('已恢复默认设置', '新的识别任务会使用默认参数。', 'success'); });
    $('#open-workspace-btn').addEventListener('click', () => toast('项目工作区', el.workspacePath.textContent || '当前项目目录', 'info'));
    $('#check-update-btn').addEventListener('click', () => toast('当前为本地模式', '请从项目的 GitHub Releases 页面检查新版本。', 'info'));
    document.addEventListener('keydown', (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') { event.preventDefault(); startTranscription(); }
      if (!event.metaKey && !event.ctrlKey && !event.altKey && !event.shiftKey && !['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) {
        const views = { '1': 'studio', '2': 'models', '3': 'history', '4': 'settings' };
        if (views[event.key]) switchView(views[event.key]);
      }
    });
  }

  async function init() {
    applyPreferencesToForm();
    bindEvents();
    renderQueue();
    await refreshHealth();
    await loadProjectSettings();
    if (state.preferences.checkModels) await refreshModels();
    else renderModelSelect();
    await refreshHistory();
    window.setInterval(refreshHealth, 30000);
  }

  init();
}());
