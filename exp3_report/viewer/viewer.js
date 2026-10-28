// Report viewer: renders REPORT.md and turns links to files in this repository into citations.
// Hover a citation to see a snapshot of the cited lines, page or figure; click it to open the
// whole source in a panel beside the report. The citation formats are listed at the top of REPORT.md.
(() => {
  'use strict';

  const CONFIG = {
    report: 'REPORT.md',
    repo: 'Garetftn/imagism-exp3', // used for the "Open on GitHub" links
    branch: 'main',
    folder: 'exp3_report/', // where this page sits in the repository; citations are relative to it
    // PDFs are shown as page images, with word boxes for highlighting a quoted passage.
    // Made with: pdftoppm -r 120 -jpeg  and  pdftotext -bbox  (see paper/README.md)
    pdfs: {
      'paper/Encoding_Imagism.pdf': {
        page: n => `paper/pages/p-${String(n).padStart(2, '0')}.jpg`,
        words: 'paper/words.json',
      },
    },
    snapshotLines: 12, // most lines a hover snapshot shows
  };

  const KIND_BY_EXT = {
    py: 'code', r: 'code', js: 'code', sh: 'code',
    md: 'notes', txt: 'notes',
    csv: 'data', tsv: 'data', json: 'data',
    png: 'figure', jpg: 'figure', jpeg: 'figure', gif: 'figure', svg: 'figure', webp: 'figure',
    pdf: 'paper',
  };
  const LANGUAGE = { py: 'python', r: 'r', js: 'javascript', sh: 'bash', json: 'json', md: 'markdown' };
  const A4 = 841.89 / 595.276; // page height / width
  const NUMBER = /^-?\d+(\.\d+)?(e[-+]?\d+)?$/i;

  // ---- helpers --------------------------------------------------------------------------------
  const $ = s => document.querySelector(s);
  const pct = v => `${(v * 100).toFixed(3)}%`;

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function note(text, isError) {
    return el('div', isError ? 'note error' : 'note', text);
  }

  function errorNote(c, err) {
    const hint = c.path.startsWith('../') ? ' Is the paper’s code at the repository root?' : '';
    return note(`Could not load ${c.path} (${err.message}).${hint}`, true);
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
  }

  const cache = new Map();
  function load(path, type = 'text') {
    const key = `${type}:${path}`;
    if (!cache.has(key)) {
      const p = fetch(encodeURI(path)).then(r => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText || ''}`.trim());
        return type === 'json' ? r.json() : r.text();
      });
      p.catch(() => cache.delete(key)); // let a failed file be tried again
      cache.set(key, p);
    }
    return cache.get(key);
  }

  // ---- citations --------------------------------------------------------------------------------
  // A citation is a relative link to a file in the repository, optionally with a location:
  //   #L12 or #L12-L20 (lines; for CSV files line 1 is the header), #page=9 (PDF page).
  // For a PDF, the link title gives words to highlight on that page.
  function parseCite(href) {
    if (!href || href.startsWith('#') || href.startsWith('//') || /^[a-z][a-z0-9+.-]*:/i.test(href)) return null;
    const hash = href.indexOf('#');
    let path;
    try {
      path = decodeURIComponent((hash < 0 ? href : href.slice(0, hash)).replace(/^\.\//, ''));
    } catch (e) {
      return null;
    }
    const ext = path.includes('.') ? path.split('.').pop().toLowerCase() : '';
    const kind = KIND_BY_EXT[ext];
    if (!kind) return null;
    const frag = hash < 0 ? '' : href.slice(hash + 1);
    const c = { path, ext, kind };
    const lines = frag.match(/^L(\d+)(?:-L?(\d+))?$/i);
    if (lines) {
      c.from = +lines[1];
      c.to = Math.max(c.from, +(lines[2] || lines[1]));
    }
    const page = frag.match(/(?:^|&)page=(\d+)/);
    if (page) c.page = +page[1];
    return c;
  }

  // the tag shown after a citation; code is split into the paper's code and yours
  function kindOf(c) {
    if (c.kind === 'code') {
      if (c.path.startsWith('../')) return ['original code', 'k-orig'];
      return [c.path.startsWith('fixed/') ? 'fixed code' : 'code', 'k-fixed'];
    }
    return {
      notes: ['notes', 'k-notes'], data: ['data', 'k-data'], figure: ['figure', 'k-figure'], paper: ['paper', 'k-paper'],
    }[c.kind];
  }

  function where(c) {
    if (c.from) return `L${c.from}${c.to > c.from ? `–${c.to}` : ''}`;
    return c.page ? `p. ${c.page}` : '';
  }

  function githubUrl(c) {
    const path = new URL(c.path, `https://x/${CONFIG.folder}`).pathname.slice(1); // resolves ../
    const url = `https://github.com/${CONFIG.repo}/blob/${CONFIG.branch}/${path.split('/').map(encodeURIComponent).join('/')}`;
    if (!c.from) return url;
    const plain = ['md', 'csv', 'tsv'].includes(c.ext) ? '?plain=1' : ''; // line anchors need the plain view
    return `${url}${plain}#L${c.from}${c.to > c.from ? `-L${c.to}` : ''}`;
  }

  // ---- views: code and text --------------------------------------------------------------------------------
  function toLines(text) {
    const t = text.replace(/\r\n?/g, '\n');
    return (t.endsWith('\n') ? t.slice(0, -1) : t).split('\n');
  }

  // highlight.js output can open a <span> on one line and close it lines later (strings, comments);
  // close and reopen the open spans at every line break so each line stands on its own
  function splitHtmlLines(html) {
    const lines = [];
    const open = [];
    let cur = '';
    const token = /<span[^>]*>|<\/span>|\n|[^<\n]+|</g;
    let m;
    while ((m = token.exec(html))) {
      const t = m[0];
      if (t === '\n') {
        lines.push(cur + '</span>'.repeat(open.length));
        cur = open.join('');
      } else {
        if (t.startsWith('<span')) open.push(t);
        else if (t === '</span>') open.pop();
        cur += t;
      }
    }
    lines.push(cur + '</span>'.repeat(open.length));
    return lines;
  }

  function highlighted(text, ext) {
    const lang = LANGUAGE[ext];
    if (window.hljs && lang && hljs.getLanguage(lang) && text.length < 500000) {
      try {
        const t = text.replace(/\r\n?/g, '\n').replace(/\n$/, '');
        return splitHtmlLines(hljs.highlight(t, { language: lang, ignoreIllegals: true }).value);
      } catch (e) {
        // fall through to plain text
      }
    }
    return toLines(text).map(escapeHtml);
  }

  function codeView(lines, c, start = 1, end = lines.length) {
    let html = '';
    for (let i = start; i <= end; i++) {
      const hit = c.from && i >= c.from && i <= c.to;
      html += `<div class="ln${hit ? ' hit' : ''}"><span class="num">${i}</span><span class="src">${lines[i - 1] || ' '}</span></div>`;
    }
    const box = el('div', 'code');
    box.innerHTML = html;
    return box;
  }

  // ---- views: CSV tables --------------------------------------------------------------------------------
  function splitRow(line, sep) {
    const out = [];
    let cur = '';
    let quoted = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (quoted) {
        if (ch === '"' && line[i + 1] === '"') { cur += '"'; i++; }
        else if (ch === '"') quoted = false;
        else cur += ch;
      } else if (ch === '"') quoted = true;
      else if (ch === sep) { out.push(cur); cur = ''; }
      else cur += ch;
    }
    out.push(cur);
    return out;
  }

  function csvView(text, c, isSnapshot) {
    const lines = toLines(text);
    const sep = c.ext === 'tsv' ? '\t' : ',';
    let lo = 2;
    let hi = lines.length;
    if (isSnapshot && c.from) {
      lo = Math.max(2, c.from);
      hi = Math.min(lines.length, c.to, lo + CONFIG.snapshotLines - 1);
    } else if (isSnapshot) {
      hi = Math.min(lines.length, 1 + CONFIG.snapshotLines);
    }
    const cells = (line, tag) => splitRow(line, sep)
      .map(v => `<${tag}${tag === 'td' && NUMBER.test(v) ? ' class="n"' : ''}>${escapeHtml(v)}</${tag}>`).join('');
    let html = `<thead><tr><th class="num">1</th>${cells(lines[0] || '', 'th')}</tr></thead><tbody>`;
    for (let i = lo; i <= hi; i++) {
      const hit = c.from && i >= c.from && i <= c.to;
      html += `<tr${hit ? ' class="hit"' : ''}><td class="num">${i}</td>${cells(lines[i - 1], 'td')}</tr>`;
    }
    const table = el('table', 'csv');
    table.innerHTML = `${html}</tbody>`;
    return table;
  }

  // ---- views: paper pages --------------------------------------------------------------------------------
  // Word boxes of one page: [word, x0, y0, x1, y1] with coordinates as fractions of the page size.
  async function pageWords(conf, n) {
    return (await load(conf.words, 'json'))[n - 1] || [];
  }

  // Boxes around the quoted words: [] when there is no quote, null when it is not found.
  // Letters and digits only are compared, so line-break hyphens, quotes and ligatures don't matter.
  function findQuote(words, quote) {
    if (!quote) return [];
    const norm = s => s.normalize('NFKC').toLowerCase().replace(/[^\p{L}\p{N}]/gu, '');
    let all = '';
    const owner = [];
    words.forEach((w, i) => {
      const t = norm(w[0]);
      all += t;
      for (let k = 0; k < t.length; k++) owner.push(i);
    });
    const q = norm(quote);
    const at = q ? all.indexOf(q) : -1;
    if (at < 0) return null;
    const boxes = []; // one box per printed line
    for (const [, x0, y0, x1, y1] of words.slice(owner[at], owner[at + q.length - 1] + 1)) {
      const line = boxes.find(b => Math.abs(b.y0 - y0) < 0.006);
      if (line) Object.assign(line, { x0: Math.min(line.x0, x0), x1: Math.max(line.x1, x1), y1: Math.max(line.y1, y1) });
      else boxes.push({ x0, y0, x1, y1 });
    }
    return boxes;
  }

  function pageView(conf, n, boxes, eager) {
    const page = el('div', 'page');
    const img = new Image();
    img.loading = eager ? 'eager' : 'lazy';
    img.decoding = 'async';
    img.alt = `Page ${n}`;
    img.src = conf.page(n);
    page.append(img);
    for (const b of boxes || []) {
      const m = el('div', 'mark');
      Object.assign(m.style, {
        left: pct(b.x0 - 0.004), top: pct(b.y0 - 0.002), width: pct(b.x1 - b.x0 + 0.008), height: pct(b.y1 - b.y0 + 0.004),
      });
      page.append(m);
    }
    return page;
  }

  // left and right edge of the printed text, as fractions of the page width
  function textSpan(words) {
    if (!words.length) return [0.06, 0.94];
    return [Math.max(0, Math.min(...words.map(w => w[1])) - 0.02), Math.min(1, Math.max(...words.map(w => w[3])) + 0.02)];
  }

  // trim the empty side margins so the page's text fills the panel width
  function clipToText(page, words) {
    const [left, right] = textSpan(words);
    const clip = el('div', 'clip');
    clip.style.aspectRatio = `${right - left} / ${A4}`;
    Object.assign(page.style, { left: pct(-left / (right - left)), width: pct(1 / (right - left)) });
    clip.append(page);
    return clip;
  }

  // the page at a readable size, cut to the width of its text and to the part around the quote
  function pageCrop(conf, n, words, boxes, width, height = 200) {
    const [left, right] = textSpan(words);
    const w = width / (right - left);
    const h = w * A4;
    let cy = 0.1 * h + height / 2; // no quote: the top of the text block
    if (boxes && boxes.length) cy = h * (Math.min(...boxes.map(b => b.y0)) + Math.max(...boxes.map(b => b.y1))) / 2;
    const top = Math.max(0, Math.min(h - height, cy - height / 2));
    const page = pageView(conf, n, boxes, true);
    Object.assign(page.style, { width: `${w}px`, height: `${h}px`, transform: `translate(${-left * w}px, ${-top}px)` });
    const crop = el('div', 'crop');
    crop.style.height = `${height}px`;
    crop.append(page);
    return crop;
  }

  // ---- hover snapshot --------------------------------------------------------------------------------
  const popup = $('#popup');
  let current = null;
  let showTimer = 0;
  let hideTimer = 0;

  async function snapshot(c, quote) {
    if (c.kind === 'figure') {
      const img = new Image();
      img.className = 'snap-img';
      img.alt = c.path;
      img.addEventListener('load', () => current && place(current));
      img.addEventListener('error', () => img.replaceWith(errorNote(c, new Error('not found'))));
      img.src = c.path;
      return img;
    }
    if (c.kind === 'paper') {
      const conf = CONFIG.pdfs[c.path];
      if (!conf) return note('Click to open the PDF.');
      const n = c.page || 1;
      const words = await pageWords(conf, n);
      const boxes = findQuote(words, quote);
      const frag = document.createDocumentFragment();
      if (quote) frag.append(el('blockquote', 'snap-quote', `“${quote}”`));
      if (boxes === null) frag.append(note(`These words were not found on p. ${n}. Check the quote or the page number.`, true));
      frag.append(pageCrop(conf, n, words, boxes, popup.clientWidth));
      return frag;
    }
    const text = await load(c.path);
    if (c.ext === 'csv' || c.ext === 'tsv') return csvView(text, c, true);
    const lines = highlighted(text, c.ext);
    if (c.from > lines.length) return note(`L${c.from} is past the end of this file (${lines.length} lines).`, true);
    const start = c.from ? Math.max(1, c.from - 1) : 1; // one line of context on each side
    const last = c.from ? Math.min(lines.length, c.to + 1) : lines.length;
    const end = Math.min(last, start + CONFIG.snapshotLines - 1);
    const box = codeView(lines, c, start, end);
    if (last > end) box.append(el('div', 'more', `… ${last - end} more line${last - end > 1 ? 's' : ''}`));
    return box;
  }

  function headNode(c) {
    const [label, cls] = kindOf(c);
    const head = el('div', 'popup-head');
    head.append(el('span', `kind ${cls}`, label), el('span', 'path', c.path));
    if (where(c)) head.append(el('span', 'loc', where(c)));
    return head;
  }

  async function showPopup(a) {
    current = a;
    const body = el('div', 'popup-body');
    body.append(note('Loading…'));
    popup.replaceChildren(headNode(a._cite), body, el('div', 'popup-foot', 'Click to open beside the report'));
    popup.hidden = false;
    place(a);
    let node;
    try {
      node = await snapshot(a._cite, a.dataset.quote);
    } catch (err) {
      node = errorNote(a._cite, err);
    }
    if (current !== a) return;
    body.replaceChildren(node);
    place(a);
  }

  function place(a) {
    const r = a.getBoundingClientRect();
    const vw = document.documentElement.clientWidth;
    const vh = window.innerHeight;
    const w = popup.offsetWidth;
    const h = popup.offsetHeight;
    let top = r.bottom + 8;
    if (top + h > vh - 8 && r.top - h - 8 >= 8) top = r.top - h - 8;
    popup.style.left = `${Math.max(8, Math.min(r.left, vw - w - 8))}px`;
    popup.style.top = `${Math.max(8, top)}px`;
  }

  function scheduleShow(a, delay) {
    clearTimeout(hideTimer);
    clearTimeout(showTimer);
    showTimer = setTimeout(() => showPopup(a), delay);
  }

  function scheduleHide() {
    clearTimeout(showTimer);
    hideTimer = setTimeout(hidePopup, 160);
  }

  function hidePopup() {
    clearTimeout(showTimer);
    popup.hidden = true;
    current = null;
  }

  popup.addEventListener('mouseenter', () => clearTimeout(hideTimer));
  popup.addEventListener('mouseleave', scheduleHide);
  window.addEventListener('scroll', hidePopup, { passive: true });

  // ---- side panel --------------------------------------------------------------------------------
  const panel = $('#panel');
  const panelBody = $('#panel-body');
  let panelToken = 0;

  async function fullView(c, quote) {
    if (c.kind === 'figure') {
      const img = new Image();
      img.className = 'full-img';
      img.alt = c.path;
      img.addEventListener('error', () => img.replaceWith(errorNote(c, new Error('not found'))));
      img.src = c.path;
      return img;
    }
    if (c.kind === 'paper') {
      const conf = CONFIG.pdfs[c.path];
      if (!conf) {
        const frame = el('iframe', 'pdf-frame');
        frame.title = c.path;
        frame.src = c.path + (c.page ? `#page=${c.page}` : '');
        return frame;
      }
      const all = await load(conf.words, 'json');
      const n = Math.min(Math.max(1, c.page || 1), all.length);
      const boxes = findQuote(all[n - 1] || [], quote);
      const box = el('div', 'pages');
      if (boxes === null) box.append(note(`These words were not found on p. ${n}: “${quote}”`, true));
      for (let i = 1; i <= all.length; i++) {
        const page = pageView(conf, i, i === n ? boxes : null, Math.abs(i - n) <= 1);
        if (i === n) page.dataset.target = '';
        box.append(el('div', 'page-label', `p. ${i}`), clipToText(page, all[i - 1] || []));
      }
      return box;
    }
    const text = await load(c.path);
    if (text.length > 5e6) return note('This file is too large to show here. Use “Open on GitHub”.');
    if (c.ext === 'csv' || c.ext === 'tsv') return csvView(text, c, false);
    return codeView(highlighted(text, c.ext), c);
  }

  async function openPanel(a) {
    const c = a._cite;
    const [label, cls] = kindOf(c);
    hidePopup();
    document.querySelectorAll('a.cite.active').forEach(x => x.classList.remove('active'));
    a.classList.add('active');
    $('#panel-kind').className = `kind ${cls}`;
    $('#panel-kind').textContent = label;
    $('#panel-path').textContent = c.path + (where(c) ? `  ${where(c)}` : '');
    $('#panel-path').title = c.path;
    const open = $('#panel-open');
    if (c.kind === 'paper') {
      open.href = c.path + (c.page ? `#page=${c.page}` : '');
      open.textContent = 'Open PDF';
    } else if (c.kind === 'figure') {
      open.href = c.path;
      open.textContent = 'Open image';
    } else {
      open.href = githubUrl(c);
      open.textContent = 'Open on GitHub';
    }
    panel.hidden = false;
    document.body.classList.add('panel-open');
    panelBody.replaceChildren(note('Loading…'));
    const token = ++panelToken;
    let node;
    try {
      node = await fullView(c, a.dataset.quote);
    } catch (err) {
      node = errorNote(c, err);
    }
    if (token !== panelToken) return;
    panelBody.replaceChildren(node);
    panelBody.scrollTop = 0;
    panelBody.scrollLeft = 0;
    // bring the cited part a third of the way down the panel
    const target = panelBody.querySelector('.hit') || panelBody.querySelector('.mark') || panelBody.querySelector('[data-target]');
    if (target) {
      const offset = target.getBoundingClientRect().top - panelBody.getBoundingClientRect().top;
      panelBody.scrollTop = offset - panelBody.clientHeight / 3;
    }
  }

  function closePanel() {
    panelToken++;
    panel.hidden = true;
    document.body.classList.remove('panel-open');
    document.querySelectorAll('a.cite.active').forEach(x => x.classList.remove('active'));
  }

  $('#panel-close').addEventListener('click', closePanel);
  document.addEventListener('keydown', e => {
    if (e.key !== 'Escape') return;
    if (!popup.hidden) hidePopup();
    else if (!panel.hidden) closePanel();
  });

  // ---- report --------------------------------------------------------------------------------
  function bindCitation(a, c) {
    const [label, cls] = kindOf(c);
    a._cite = c;
    a.classList.add('cite', cls);
    a.dataset.kind = label;
    if (a.title) {
      a.dataset.quote = a.title; // shown in the snapshot instead of as a browser tooltip
      a.removeAttribute('title');
    }
    a.setAttribute('aria-label', `${a.textContent} (${label}: ${c.path}${where(c) ? ` ${where(c)}` : ''})`);
    a.addEventListener('mouseenter', () => scheduleShow(a, 180));
    a.addEventListener('mouseleave', scheduleHide);
    a.addEventListener('focus', () => scheduleShow(a, 0));
    a.addEventListener('blur', scheduleHide);
    a.addEventListener('click', e => {
      if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return; // ctrl-click opens the file in a new tab
      e.preventDefault();
      openPanel(a);
    });
  }

  async function init() {
    $('#repo-link').href = `https://github.com/${CONFIG.repo}`;
    const main = $('#report');
    let md;
    try {
      md = await load(CONFIG.report);
    } catch (err) {
      main.replaceChildren(note(`Could not load ${CONFIG.report} (${err.message}). Open this page through a web server: run “python -m http.server” in the repository root and go to http://localhost:8000/exp3_report/.`, true));
      return;
    }
    main.innerHTML = marked.parse(md, { gfm: true }); // REPORT.md is your own text, so its HTML is trusted
    if (window.hljs) main.querySelectorAll('pre code').forEach(b => hljs.highlightElement(b));
    for (const a of main.querySelectorAll('a[href]')) {
      const href = a.getAttribute('href');
      const c = parseCite(href);
      if (c) bindCitation(a, c);
      else if (/^https?:/i.test(href)) {
        a.target = '_blank';
        a.rel = 'noopener';
      }
    }
    const h1 = main.querySelector('h1');
    if (h1) document.title = h1.textContent;
  }

  init();
})();
