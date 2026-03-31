/* ==========================================
   XRay Diagnostics — Frontend Logic
   ========================================== */

// ── Tab navigation ──
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.add('hidden'));
    btn.classList.add('active');
    document.getElementById(`tab-${tab}`).classList.remove('hidden');
    if (tab === 'history') loadHistory();
    if (tab === 'stats')   loadStats();
  });
});

// ── Health check ──
async function checkHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    const dot   = document.getElementById('status-dot');
    const label = document.getElementById('status-label');

    if (!data.claude) {
      dot.className = 'status-dot error';
      label.textContent = 'Claude API não configurado';
    } else if (!data.elasticsearch) {
      dot.className = 'status-dot warn';
      label.textContent = 'Claude OK · ES: memória';
    } else {
      dot.className = 'status-dot ok';
      label.textContent = 'Claude + Elasticsearch OK';
    }
  } catch {
    document.getElementById('status-dot').className = 'status-dot error';
    document.getElementById('status-label').textContent = 'Servidor indisponível';
  }
}
checkHealth();

// ── File input & drag-drop ──
const dropZone    = document.getElementById('drop-zone');
const fileInput   = document.getElementById('file-input');
const previewSec  = document.getElementById('preview-section');
const previewImg  = document.getElementById('preview-img');
const previewName = document.getElementById('preview-name');
const previewSize = document.getElementById('preview-size');
let selectedFile  = null;

function fmtBytes(n) {
  if (n < 1024) return n + ' B';
  if (n < 1048576) return (n/1024).toFixed(1) + ' KB';
  return (n/1048576).toFixed(1) + ' MB';
}

function setFile(file) {
  if (!file) return;
  selectedFile = file;
  previewName.textContent = file.name;
  previewSize.textContent = fmtBytes(file.size);

  if (file.type.startsWith('image/')) {
    const reader = new FileReader();
    reader.onload = e => {
      previewImg.src = e.target.result;
      previewImg.style.display = 'block';
    };
    reader.readAsDataURL(file);
  } else {
    previewImg.style.display = 'none';
  }

  dropZone.classList.add('hidden');
  previewSec.classList.remove('hidden');
  document.getElementById('result-card').classList.add('hidden');
}

fileInput.addEventListener('change', e => setFile(e.target.files[0]));

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  setFile(e.dataTransfer.files[0]);
});
dropZone.addEventListener('click', () => fileInput.click());

document.getElementById('btn-clear').addEventListener('click', () => {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src = '';
  previewSec.classList.add('hidden');
  dropZone.classList.remove('hidden');
  document.getElementById('result-card').classList.add('hidden');
});

// ── Analyze ──
document.getElementById('btn-analyze').addEventListener('click', async () => {
  if (!selectedFile) return;

  const resultCard    = document.getElementById('result-card');
  const loadingOverlay = document.getElementById('loading-overlay');
  const resultBody    = document.getElementById('result-body');
  const analyzeBtn    = document.getElementById('btn-analyze');

  resultCard.classList.remove('hidden');
  loadingOverlay.classList.remove('hidden');
  resultBody.innerHTML = '';
  analyzeBtn.disabled = true;

  const removeErr = resultCard.querySelector('.error-msg');
  if (removeErr) removeErr.remove();

  try {
    const form = new FormData();
    form.append('file', selectedFile);

    const res = await fetch('/api/upload', { method: 'POST', body: form });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || 'Erro desconhecido');

    loadingOverlay.classList.add('hidden');
    renderDiagnosis(data.diagnosis, resultBody);
    setRiskBadge(data.diagnosis?.risk_level || 'Indeterminado');

  } catch (err) {
    loadingOverlay.classList.add('hidden');
    const div = document.createElement('div');
    div.className = 'error-msg';
    div.textContent = '⚠ ' + err.message;
    resultCard.appendChild(div);
  } finally {
    analyzeBtn.disabled = false;
  }
});

// ── Render diagnosis ──
function setRiskBadge(level) {
  const badge = document.getElementById('result-risk-badge');
  badge.textContent = level;
  badge.className = 'risk-badge risk-' + level;
}

function renderDiagnosis(d, container) {
  if (!d) { container.innerHTML = '<p class="empty-state">Sem dados de diagnóstico.</p>'; return; }

  let html = '';

  // Summary
  html += section('Resumo', `<div class="result-summary">${esc(d.summary || '—')}</div>`);

  // Meta
  const metaItems = [
    { label: 'Região', value: d.region },
    { label: 'Incidência', value: d.laterality },
    { label: 'Qualidade', value: d.image_quality?.score },
    { label: 'Modelo', value: d.model },
  ].filter(m => m.value);

  if (metaItems.length) {
    html += section('Metadados',
      `<div class="meta-row">` +
      metaItems.map(m => `<div class="meta-item"><span>${esc(m.label)}:</span><strong>${esc(m.value)}</strong></div>`).join('') +
      `</div>`
    );
  }

  // Confidence
  const conf = d.confidence ?? 0;
  html += section('Confiança da Análise',
    `<div class="confidence-bar-wrap">
       <span style="font-size:.85rem;color:var(--text-muted)">${conf}%</span>
       <div class="confidence-bar-track">
         <div class="confidence-bar-fill" style="width:${conf}%"></div>
       </div>
     </div>`
  );

  // Findings
  if (d.findings && d.findings.length) {
    const items = d.findings.map(f =>
      `<div class="finding-item">
         <span class="finding-severity sev-${esc(f.severity)}">${esc(f.severity)}</span>
         <div class="finding-content">
           <div class="finding-location">${esc(f.location)}</div>
           <div class="finding-desc">${esc(f.description)}</div>
         </div>
       </div>`
    ).join('');
    html += section('Achados Radiológicos', items);
  }

  // Areas of concern
  if (d.areas_of_concern && d.areas_of_concern.length) {
    const items = d.areas_of_concern.map(c =>
      `<div class="concern-item">
         <div class="concern-area">📍 ${esc(c.area)}</div>
         <div class="concern-obs">${esc(c.observation)}</div>
         ${c.significance ? `<div class="concern-sig">↳ ${esc(c.significance)}</div>` : ''}
       </div>`
    ).join('');
    html += section('Áreas de Atenção', items);
  }

  // Normal structures
  if (d.normal_structures && d.normal_structures.length) {
    html += section('Estruturas Normais',
      `<div style="font-size:.85rem;color:var(--text-muted);line-height:1.8">${d.normal_structures.map(esc).join(' · ')}</div>`
    );
  }

  // Recommendations
  if (d.recommendations && d.recommendations.length) {
    html += section('Recomendações',
      `<ul class="rec-list">${d.recommendations.map(r => `<li>${esc(r)}</li>`).join('')}</ul>`
    );
  }

  // Image quality notes
  if (d.image_quality?.notes) {
    html += section('Qualidade da Imagem',
      `<div style="font-size:.85rem;color:var(--text-muted)">${esc(d.image_quality.notes)}</div>`
    );
  }

  // Disclaimer
  html += `<div class="disclaimer-box">⚠ ${esc(d.disclaimer || 'Esta análise é gerada por IA e deve ser revisada por um radiologista.')}</div>`;

  container.innerHTML = html;
}

function section(title, content) {
  return `<div class="result-section">
    <div class="result-section-title">${title}</div>
    ${content}
  </div>`;
}

function esc(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── History ──
async function loadHistory() {
  const grid = document.getElementById('history-grid');
  grid.innerHTML = '<div class="empty-state">Carregando...</div>';
  try {
    const res  = await fetch('/api/images?size=50');
    const data = await res.json();
    renderGrid(data.images, grid);
  } catch {
    grid.innerHTML = '<div class="empty-state error-msg">Erro ao carregar histórico.</div>';
  }
}

function renderGrid(images, grid) {
  if (!images || !images.length) {
    grid.innerHTML = '<div class="empty-state">Nenhuma análise encontrada.</div>';
    return;
  }
  grid.innerHTML = images.map(img => historyCard(img)).join('');
  grid.querySelectorAll('.history-card').forEach((card, i) => {
    card.addEventListener('click', () => openModal(images[i]));
  });
}

function historyCard(img) {
  const d    = img.diagnosis || {};
  const risk = d.risk_level  || 'Indeterminado';
  const date = img.upload_date
    ? new Date(img.upload_date).toLocaleDateString('pt-BR', { day:'2-digit', month:'short', year:'numeric', hour:'2-digit', minute:'2-digit' })
    : '—';
  const imgSrc = img.stored_filename ? `/uploads/${img.stored_filename}` : null;

  return `<div class="history-card">
    <div class="history-img-wrap">
      ${imgSrc
        ? `<img class="history-img" src="${esc(imgSrc)}" alt="${esc(img.original_filename)}" loading="lazy"/>`
        : `<div class="history-img-placeholder">🩻</div>`}
    </div>
    <div class="history-info">
      <div class="history-name" title="${esc(img.original_filename)}">${esc(img.original_filename)}</div>
      <div class="history-date">${date}</div>
      <div class="history-footer">
        <span class="history-region">${esc(d.region || '—')}</span>
        <span class="risk-badge risk-${esc(risk)}">${esc(risk)}</span>
      </div>
    </div>
  </div>`;
}

document.getElementById('btn-refresh-history').addEventListener('click', loadHistory);

// ── Search ──
document.getElementById('btn-search').addEventListener('click', doSearch);
document.getElementById('search-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') doSearch();
});

async function doSearch() {
  const q = document.getElementById('search-input').value.trim();
  if (!q) return;
  const grid = document.getElementById('search-results');
  grid.innerHTML = '<div class="empty-state">Buscando...</div>';
  try {
    const res  = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q }),
    });
    const data = await res.json();
    renderGrid(data.results, grid);
  } catch {
    grid.innerHTML = '<div class="empty-state error-msg">Erro na busca.</div>';
  }
}

function quickSearch(term) {
  document.getElementById('search-input').value = term;
  doSearch();
  // switch to search tab
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.add('hidden'));
  document.querySelector('[data-tab="search"]').classList.add('active');
  document.getElementById('tab-search').classList.remove('hidden');
}

// ── Stats ──
async function loadStats() {
  try {
    const res  = await fetch('/api/stats');
    const data = await res.json();

    document.getElementById('stat-total').textContent = data.total ?? 0;

    const risk = data.by_risk || {};
    const high = (risk['Alto'] || 0) + (risk['Crítico'] || 0);
    const mod  = risk['Moderado'] || 0;
    const norm = (risk['Normal'] || 0) + (risk['Baixo'] || 0);

    document.getElementById('stat-high').textContent   = high;
    document.getElementById('stat-mod').textContent    = mod;
    document.getElementById('stat-normal').textContent = norm;

    renderBarChart('chart-risk', risk);
    renderBarChart('chart-region', data.by_region || {});

    const storageEl = document.getElementById('storage-info');
    storageEl.textContent = data.storage === 'elasticsearch'
      ? '✅ Elasticsearch ativo — dados persistentes'
      : '⚠ Modo memória — dados perdidos ao reiniciar. Configure ELASTICSEARCH_URL no .env para persistência.';

  } catch {
    document.getElementById('stat-total').textContent = '!';
  }
}

function renderBarChart(containerId, data) {
  const el = document.getElementById(containerId);
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1]);
  if (!entries.length) { el.innerHTML = '<div style="color:var(--text-muted);font-size:.85rem">Sem dados</div>'; return; }
  const max = Math.max(...entries.map(e => e[1]));
  el.innerHTML = entries.map(([label, count]) =>
    `<div class="bar-row">
       <div class="bar-label" title="${esc(label)}">${esc(label)}</div>
       <div class="bar-track"><div class="bar-fill" style="width:${(count/max*100).toFixed(1)}%"></div></div>
       <div class="bar-count">${count}</div>
     </div>`
  ).join('');
}

// ── Modal ──
function openModal(img) {
  const body = document.getElementById('modal-body');
  const d    = img.diagnosis || {};
  const risk = d.risk_level || 'Indeterminado';
  const date = img.upload_date
    ? new Date(img.upload_date).toLocaleDateString('pt-BR', { dateStyle:'long', timeStyle:'short' })
    : '—';
  const imgSrc = img.stored_filename ? `/uploads/${img.stored_filename}` : null;

  body.innerHTML = `
    <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:20px;padding-right:32px">
      <div>
        <h2 style="font-size:1.05rem;font-weight:600">${esc(img.original_filename)}</h2>
        <p style="color:var(--text-muted);font-size:.82rem;margin-top:4px">${date}</p>
      </div>
      <span class="risk-badge risk-${esc(risk)}">${esc(risk)}</span>
    </div>
    ${imgSrc ? `<img src="${esc(imgSrc)}" style="width:100%;max-height:300px;object-fit:contain;border-radius:8px;background:#000;margin-bottom:20px;border:1px solid var(--border)"/>` : ''}
    <div id="modal-diagnosis"></div>
    <div style="margin-top:20px;display:flex;gap:10px;justify-content:flex-end">
      <button class="btn btn-ghost btn-sm" onclick="deleteImage('${img.id}')">🗑 Excluir</button>
    </div>
  `;

  renderDiagnosis(d, document.getElementById('modal-diagnosis'));
  document.getElementById('modal-overlay').classList.remove('hidden');
}

document.getElementById('modal-close').addEventListener('click', () =>
  document.getElementById('modal-overlay').classList.add('hidden')
);
document.getElementById('modal-overlay').addEventListener('click', e => {
  if (e.target === document.getElementById('modal-overlay'))
    document.getElementById('modal-overlay').classList.add('hidden');
});

async function deleteImage(id) {
  if (!confirm('Excluir esta análise?')) return;
  try {
    await fetch(`/api/images/${id}`, { method: 'DELETE' });
    document.getElementById('modal-overlay').classList.add('hidden');
    loadHistory();
  } catch {
    alert('Erro ao excluir.');
  }
}
