/* Review editor.

   Ported from the auto_voc_labeller prototype, which had the interaction model
   right: draw on a canvas at native resolution, drag the body to move and the
   corners to resize, and keep the hands on the keyboard — Space saves and
   advances, which is the whole loop when there are two thousand images to get
   through.

   Two things differ from the prototype. Boxes carry a species each rather than
   all being forced to one, because the models here are multi-class. And every
   box remembers whether a model proposed it or a person drew it: touching a box
   flips it to `human`, which is what lets you ask later how much of a dataset
   was actually checked. */

(function () {
  const A = window.ANNOTATE;
  const $ = function (id) { return document.getElementById(id); };
  const canvas = $('canvas'), ctx = canvas.getContext('2d');

  let idx = -1, boxes = [], selected = -1, meta = {};
  let img = new Image(), dragging = false, mode = null, start = {}, orig = {};
  let states = A.states.slice();
  let saving = Promise.resolve();

  // ------------------------------------------------------------- list --
  function renderList() {
    const host = $('image-list');
    host.innerHTML = A.images.map(function (name, i) {
      const cls = 'ann-item' + (i === idx ? ' sel' : '') + ' st-' + states[i];
      const mark = states[i] === 'reviewed' ? '✓' : (states[i] === 'predicted' ? '·' : '');
      return '<div class="' + cls + '" data-open="' + i + '">'
        + '<span class="mark">' + mark + '</span>'
        + '<span class="nm">' + escapeHtml(name) + '</span></div>';
    }).join('') || '<p class="muted small">Nothing matches this filter.</p>';

    const reviewed = states.filter(function (s) { return s === 'reviewed'; }).length;
    $('review-count').textContent = reviewed + ' / ' + A.total;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>'"]/g, function (c) {
      return {'&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'}[c];
    });
  }

  // ------------------------------------------------------------ labels --
  /* Species are suggestions, never a fixed list. A new animal has to be
     typeable the moment it appears, which is exactly when no model knows it
     yet, so every species field is a combobox backed by one shared <datalist>.
     Typing a name adds it to that list for the rest of the session, and the
     server records it against the batch so it survives a reload. */
  function ensureLabel(name) {
    name = String(name || '').trim();
    if (!name || A.classes.indexOf(name) >= 0) return name;
    A.classes.push(name);
    A.classes.sort(function (a, b) { return a.localeCompare(b); });
    const list = $('species');
    const option = document.createElement('option');
    option.value = name;
    list.appendChild(option);
    return name;
  }

  function defaultLabel() {
    return ensureLabel($('box-label').value) || A.classes[0] || 'animal';
  }

  // ------------------------------------------------------------ canvas --
  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (img.complete && img.naturalWidth) ctx.drawImage(img, 0, 0);
    const scale = Math.max(2, canvas.width / 600);
    ctx.lineWidth = scale;
    ctx.font = Math.max(16, canvas.width / 70) + 'px system-ui';

    boxes.forEach(function (b, i) {
      // Colour carries meaning, not decoration: amber is a guess nobody has
      // confirmed, green is a box a person stands behind, and the selection is
      // always the brightest thing on screen.
      const colour = i === selected ? '#ffd400'
        : (b.source === 'human' ? '#29a745' : '#d99a12');
      ctx.strokeStyle = colour;
      ctx.fillStyle = colour;
      ctx.strokeRect(b.xmin, b.ymin, b.xmax - b.xmin, b.ymax - b.ymin);
      const tag = (i + 1) + ' ' + (b.label || '')
        + (b.source === 'model' && b.score != null ? ' ' + Number(b.score).toFixed(2) : '');
      ctx.fillText(tag, b.xmin + 4, Math.max(20, b.ymin - 6));
      if (i === selected) {
        handles(b).forEach(function (h) { ctx.fillRect(h.x - 6, h.y - 6, 12, 12); });
      }
    });
    renderTable();
  }

  function renderTable() {
    $('box-table').innerHTML = boxes.map(function (b, i) {
      const w = Math.round(b.xmax - b.xmin), h = Math.round(b.ymax - b.ymin);
      return '<tr' + (i === selected ? ' class="sel"' : '') + ' data-row="' + i + '">'
        + '<td>' + (i + 1) + '</td>'
        + '<td><input data-label="' + i + '" list="species" autocomplete="off"'
        + ' value="' + escapeHtml(b.label || '') + '"></td>'
        + '<td class="num small">' + (b.score != null ? Number(b.score).toFixed(2) : '-') + '</td>'
        + '<td><span class="badge ' + (b.source === 'human' ? 'good' : 'warn') + '">'
        + (b.source === 'human' ? 'you' : 'AI') + '</span></td>'
        + '<td class="num small muted">' + w + '×' + h + '</td>'
        + '<td class="num"><button class="btn danger" data-drop="' + i
        + '" style="padding:2px 8px">✕</button></td></tr>';
    }).join('');
  }

  function handles(b) {
    return [{n: 'tl', x: b.xmin, y: b.ymin}, {n: 'tr', x: b.xmax, y: b.ymin},
            {n: 'bl', x: b.xmin, y: b.ymax}, {n: 'br', x: b.xmax, y: b.ymax}];
  }

  function at(ev) {
    const r = canvas.getBoundingClientRect();
    return {x: (ev.clientX - r.left) * canvas.width / r.width,
            y: (ev.clientY - r.top) * canvas.height / r.height};
  }

  function hit(p) {
    const grab = Math.max(10, canvas.width / 120);
    for (let i = boxes.length - 1; i >= 0; i--) {
      const b = boxes[i];
      for (const h of handles(b)) {
        if (Math.abs(p.x - h.x) < grab && Math.abs(p.y - h.y) < grab) return {i: i, mode: h.n};
      }
      if (p.x >= b.xmin && p.x <= b.xmax && p.y >= b.ymin && p.y <= b.ymax) {
        return {i: i, mode: 'move'};
      }
    }
    return null;
  }

  function normalise(b) {
    b.xmin = Math.max(0, Math.min(canvas.width, b.xmin));
    b.xmax = Math.max(0, Math.min(canvas.width, b.xmax));
    b.ymin = Math.max(0, Math.min(canvas.height, b.ymin));
    b.ymax = Math.max(0, Math.min(canvas.height, b.ymax));
    if (b.xmin > b.xmax) { const t = b.xmin; b.xmin = b.xmax; b.xmax = t; }
    if (b.ymin > b.ymax) { const t = b.ymin; b.ymin = b.ymax; b.ymax = t; }
  }

  /* Any edit makes the box yours. Without this the provenance record would say
     a model drew a box a person spent a minute correcting. */
  function claim(b) { b.source = 'human'; b.score = null; }

  canvas.addEventListener('mousedown', function (ev) {
    const p = at(ev), h = hit(p);
    start = p;
    if (h) {
      selected = h.i; mode = h.mode; orig = Object.assign({}, boxes[selected]);
    } else {
      selected = boxes.length;
      boxes.push({label: defaultLabel(), score: null, source: 'human',
                  xmin: p.x, ymin: p.y, xmax: p.x + 1, ymax: p.y + 1});
      mode = 'br'; orig = Object.assign({}, boxes[selected]);
    }
    dragging = true;
    draw();
  });

  canvas.addEventListener('mousemove', function (ev) {
    if (!dragging) return;
    const p = at(ev), b = boxes[selected];
    if (!b) return;
    if (mode === 'move') {
      const dx = p.x - start.x, dy = p.y - start.y;
      b.xmin = orig.xmin + dx; b.xmax = orig.xmax + dx;
      b.ymin = orig.ymin + dy; b.ymax = orig.ymax + dy;
    } else {
      if (mode.indexOf('l') >= 0) b.xmin = p.x;
      if (mode.indexOf('r') >= 0) b.xmax = p.x;
      if (mode.indexOf('t') >= 0) b.ymin = p.y;
      if (mode.indexOf('b') >= 0) b.ymax = p.y;
    }
    normalise(b);
    claim(b);
    draw();
  });

  ['mouseup', 'mouseleave'].forEach(function (e) {
    canvas.addEventListener(e, function () { dragging = false; mode = null; });
  });

  // ------------------------------------------------------------- data --
  function url(rel) {
    return '/api/batches/' + A.batchId + '/images/' + encodeURIComponent(rel).replace(/%2F/g, '/');
  }

  function open(i) {
    if (i < 0 || i >= A.images.length) return;
    idx = i; selected = -1; boxes = [];
    const name = A.images[i];
    $('current-name').textContent = name;
    renderList();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const wanted = fetch(url(name)).then(function (r) { return r.json(); });
    img.onload = function () {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      draw();
      if (i + 1 < A.images.length) {          // warm the next one
        new Image().src = '/api/batches/' + A.batchId + '/file/'
          + encodeURIComponent(A.images[i + 1]).replace(/%2F/g, '/');
      }
    };
    img.src = '/api/batches/' + A.batchId + '/file/'
      + encodeURIComponent(name).replace(/%2F/g, '/');

    wanted.then(function (d) {
      meta = d;
      boxes = d.boxes || [];
      boxes.forEach(function (b) { ensureLabel(b.label); });
      draw();
    });
  }

  function save(advance) {
    if (idx < 0) return Promise.resolve();
    const at_ = idx, name = A.images[idx];
    const payload = {boxes: JSON.parse(JSON.stringify(boxes)),
                     width: canvas.width, height: canvas.height};

    states[at_] = 'reviewed';
    renderList();
    $('editor-status').textContent = 'Saving…';

    // Queued rather than awaited, so Space-Space-Space through a folder never
    // waits on the network. The list already shows the optimistic state.
    const request = fetch(url(name), {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    }).then(function (r) {
      if (!r.ok) throw new Error(r.status);
      $('editor-status').textContent = 'Saved ' + name;
    }).catch(function () {
      states[at_] = 'predicted';
      renderList();
      $('editor-status').textContent = 'Could not save ' + name;
    });
    saving = saving.then(function () { return request; });

    if (advance) open(at_ + 1);
    return request;
  }

  // ---------------------------------------------------------- controls --
  $('image-list').addEventListener('click', function (ev) {
    const el = ev.target.closest('[data-open]');
    if (el) open(Number(el.dataset.open));
  });

  $('box-table').addEventListener('change', function (ev) {
    const field = ev.target.closest('[data-label]');
    if (!field) return;
    const b = boxes[Number(field.dataset.label)];
    const name = ensureLabel(field.value);
    if (!name) { field.value = b.label || ''; return; }   // never save a blank species
    b.label = name;
    claim(b);
    draw();
  });

  $('box-table').addEventListener('click', function (ev) {
    const drop = ev.target.closest('[data-drop]');
    if (drop) {
      boxes.splice(Number(drop.dataset.drop), 1);
      selected = -1;
      draw();
      return;
    }
    const row = ev.target.closest('[data-row]');
    if (row) { selected = Number(row.dataset.row); draw(); }
  });

  $('add-box').addEventListener('click', function () {
    const w = canvas.width || 100, h = canvas.height || 100;
    boxes.push({label: defaultLabel(), score: null, source: 'human',
                xmin: w * 0.4, ymin: h * 0.4, xmax: w * 0.6, ymax: h * 0.6});
    selected = boxes.length - 1;
    draw();
  });

  $('del-box').addEventListener('click', function () {
    if (selected >= 0) { boxes.splice(selected, 1); selected = -1; draw(); }
  });

  function deleteImage() {
    if (idx < 0) return;
    const name = A.images[idx];
    if (!confirm('Remove ' + name + ' from this batch?\n\n'
                 + 'Its annotation goes too. The original file in the folder you '
                 + 'imported from is not touched.')) return;

    const at_ = idx;
    fetch(url(name), {method: 'DELETE'})
      .then(function (r) {
        if (!r.ok) throw new Error(r.status);
        A.images.splice(at_, 1);
        states.splice(at_, 1);
        A.total -= 1;
        $('editor-status').textContent = 'Removed ' + name;
        if (!A.images.length) {
          idx = -1; boxes = []; selected = -1;
          $('current-name').textContent = 'No images left';
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          renderTable();
          renderList();
          return;
        }
        // Land on the next image rather than jumping to the start: deleting is
        // something you do while working through a folder in order.
        idx = -1;
        open(Math.min(at_, A.images.length - 1));
      })
      .catch(function () { $('editor-status').textContent = 'Could not remove ' + name; });
  }

  $('del-image').addEventListener('click', deleteImage);
  $('save').addEventListener('click', function () { save(true); });
  $('prev').addEventListener('click', function () { open(idx - 1); });
  $('next').addEventListener('click', function () { open(idx + 1); });

  document.addEventListener('keydown', function (ev) {
    const tag = (ev.target.tagName || '').toLowerCase();
    if (['input', 'select', 'textarea'].indexOf(tag) >= 0) return;
    if (ev.code === 'Space') { ev.preventDefault(); save(true); }
    else if (ev.key === 'ArrowRight') open(idx + 1);
    else if (ev.key === 'ArrowLeft') open(idx - 1);
    // Shift qualifies it, because losing an image to a stray Delete while
    // aiming at a box would be the wrong kind of surprise.
    else if (ev.key === 'Delete' && ev.shiftKey) deleteImage();
    else if (ev.key === 'Delete') $('del-box').click();
    else if (ev.ctrlKey && ev.key.toLowerCase() === 's') { ev.preventDefault(); save(false); }
  });

  // Any species already on the images comes from the server's list; seed the
  // control with the commonest so the first new box is one keystroke, not two.
  $('box-label').value = A.classes[0] || '';
  $('box-label').addEventListener('change', function () { ensureLabel($('box-label').value); });
  renderList();
  if (A.images.length) open(0);
})();
