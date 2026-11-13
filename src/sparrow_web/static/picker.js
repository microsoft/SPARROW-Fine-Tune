/* Folder picker.
   Shared by "new project" (read-only roots) and "publish a batch" (writable
   roots). The two modes differ only by the `writable` flag, which selects which
   allowlist the server checks — keeping them in one file is what stops the two
   pages drifting into disagreeing about what a valid folder looks like. */

window.Picker = (function () {
  const LABELS = {
    '/browse/c': 'C: drive', '/browse/d': 'D: drive', '/browse/e': 'E: drive',
    '/browse/home': 'Linux home', '/browse': 'This machine',
    '/write/c': 'C: drive', '/write/d': 'D: drive', '/write/e': 'E: drive',
    '/write/home': 'Linux home', '/write': 'This machine',
    '/workspace/datasets': 'Configured dataset folder'
  };

  function fmt(n, partial) { return n.toLocaleString() + (partial ? '+' : ''); }
  function label(path, name) { return LABELS[path] || name; }

  function mount(opts) {
    const writable = !!opts.writable;
    // In writable mode any folder is a valid destination, including an empty
    // one — you are choosing where to put files, not finding files.
    const selectable = opts.selectable || function (e) { return e.images > 0; };
    let current = null;

    const el = function (id) { return document.getElementById(id); };

    function query(path) {
      const p = new URLSearchParams();
      if (path) p.set('path', path);
      if (writable) p.set('writable', '1');
      return '/api/browse?' + p;
    }

    function render(data) {
      current = data;
      el('crumbs').innerHTML = data.breadcrumbs.map(function (c, i) {
        return (i ? ' / ' : '') + '<a href="#" data-go="' + c.path + '">'
          + label(c.path, c.name) + '</a>';
      }).join('');

      const here = data.here;
      const summary = el('here-summary');
      if (summary) {
        summary.textContent = here.images
          ? fmt(here.images, here.partial) + ' images, ' + fmt(here.xmls, here.partial)
            + ' annotations here'
          : (writable ? 'Empty folder' : 'No images directly in this folder');
      }
      const useHere = el('use-here');
      if (useHere) useHere.disabled = !(writable || here.images);

      let rows = '';
      if (data.parent) {
        rows += '<tr class="nav"><td colspan="3"><a href="#" data-go="' + data.parent
          + '">↑ Up one level</a></td></tr>';
      }
      if (!data.entries.length && !here.images) {
        rows += '<tr><td colspan="3" class="muted small">This folder is empty.</td></tr>';
      }
      data.entries.forEach(function (e) {
        const pick = selectable(e);
        rows += '<tr>'
          + '<td><a href="#" data-go="' + e.path + '">'
          + (e.is_dataset ? '🗂 ' : '📁 ') + label(e.path, e.name) + '</a>'
          + (e.subdirs ? '<span class="small muted"> · ' + e.subdirs + ' subfolder'
              + (e.subdirs > 1 ? 's' : '') + '</span>' : '')
          + '</td>'
          + '<td class="num small">' + (e.images ? fmt(e.images, e.partial) + ' img' : '') + '</td>'
          + '<td class="num">' + (pick
              ? '<button class="btn" type="button" data-pick="' + e.path + '">Use</button>' : '')
          + '</td></tr>';
      });
      el('browser').innerHTML = '<table class="browser"><tbody>' + rows + '</tbody></table>';
    }

    function go(path) {
      el('browser').innerHTML = '<p class="muted small">Loading…</p>';
      fetch(query(path))
        .then(function (r) { return r.json(); })
        .then(function (d) {
          if (d.error) {
            el('browser').innerHTML = '<div class="notice bad">' + d.error + '</div>';
            return;
          }
          render(d);
        })
        .catch(function () {
          el('browser').innerHTML =
            '<div class="notice bad">Could not read that folder.</div>';
        });
    }

    document.addEventListener('click', function (ev) {
      const goTo = ev.target.closest('[data-go]');
      if (goTo) { ev.preventDefault(); go(goTo.dataset.go); return; }
      const p = ev.target.closest('[data-pick]');
      if (p) { ev.preventDefault(); opts.onPick(p.dataset.pick); }
    });
    const useHere = el('use-here');
    if (useHere) useHere.addEventListener('click', function () {
      if (current) opts.onPick(current.path);
    });

    go(opts.start || '');
    return { go: go };
  }

  return { mount: mount, label: label };
})();
