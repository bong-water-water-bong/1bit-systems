// docs.js — sidebar filter + group collapse + copy-buttons + right TOC scrollspy.
// Loads /docs/index.json once for search.
(function () {
  var $q = document.getElementById('q');
  var body = document.body;
  var curSlug = body.getAttribute('data-slug');

  // ── sidebar: mark current, collapse groups, auto-open current group ──
  var sidebar = document.querySelector('.doc-sidebar');
  if (sidebar) {
    var groups = sidebar.querySelectorAll('.group');
    var isIndex = curSlug === 'index' || !curSlug;
    for (var gi = 0; gi < groups.length; gi++) {
      var g = groups[gi];
      var h = g.querySelector('h4');
      var hasCur = curSlug && g.querySelector('li[data-slug="' + curSlug + '"]');
      if (!isIndex && !hasCur) g.classList.add('collapsed');
      if (h) {
        h.addEventListener('click', (function (el) {
          return function () { el.classList.toggle('collapsed'); };
        })(g));
      }
    }
    if (curSlug) {
      var li = sidebar.querySelector('li[data-slug="' + curSlug + '"]');
      if (li) {
        li.classList.add('active');
        var a = li.querySelector('a');
        if (a) a.classList.add('current');
      }
    }
  }

  // ── hamburger + backdrop (mobile drawer) ──
  var topbar = document.querySelector('.doc-topbar');
  if (topbar && sidebar) {
    var burger = document.createElement('button');
    burger.className = 'doc-burger';
    burger.setAttribute('aria-label', 'menu');
    burger.textContent = '\u2630';
    topbar.insertBefore(burger, topbar.firstChild);
    var backdrop = document.createElement('div');
    backdrop.className = 'doc-backdrop';
    document.body.appendChild(backdrop);
    function closeDrawer() { body.classList.remove('sidebar-open'); }
    burger.addEventListener('click', function () { body.classList.toggle('sidebar-open'); });
    backdrop.addEventListener('click', closeDrawer);
    sidebar.addEventListener('click', function (e) {
      if (e.target.tagName === 'A') closeDrawer();
    });
  }

  // ── copy-to-clipboard on <pre> ──
  var pres = document.querySelectorAll('.doc-main pre');
  for (var pi = 0; pi < pres.length; pi++) {
    var pre = pres[pi];
    var code = pre.querySelector('code');
    if (code) {
      var cls = (code.className || '').match(/language-(\S+)/);
      if (cls) pre.setAttribute('data-lang', cls[1]);
    }
    var btn = document.createElement('button');
    btn.className = 'doc-copy';
    btn.type = 'button';
    btn.textContent = 'copy';
    btn.addEventListener('click', (function (p, b) {
      return function () {
        var txt = (p.querySelector('code') || p).innerText;
        (navigator.clipboard ? navigator.clipboard.writeText(txt) : Promise.reject())
          .then(function () {
            b.textContent = 'copied'; b.classList.add('copied');
            setTimeout(function () { b.textContent = 'copy'; b.classList.remove('copied'); }, 1400);
          })
          .catch(function () { b.textContent = 'err'; });
      };
    })(pre, btn));
    pre.appendChild(btn);
  }

  // ── right-rail TOC from h2/h3 in .doc-main ──
  var main = document.querySelector('.doc-main:not(.doc-landing)');
  if (main) {
    var heads = main.querySelectorAll('h2[id], h3[id]');
    if (heads.length > 1) {
      var toc = document.createElement('aside');
      toc.className = 'doc-toc';
      var title = document.createElement('p');
      title.className = 'toc-title';
      title.textContent = 'on this page';
      toc.appendChild(title);
      var ul = document.createElement('ul');
      for (var hi = 0; hi < heads.length; hi++) {
        var hd = heads[hi];
        var item = document.createElement('li');
        item.className = 'lvl-' + hd.tagName.charAt(1);
        var link = document.createElement('a');
        link.href = '#' + hd.id;
        link.textContent = hd.textContent.replace(/^>\s*/, '');
        item.appendChild(link);
        ul.appendChild(item);
      }
      toc.appendChild(ul);
      var shell = document.querySelector('.doc-shell');
      if (shell) shell.appendChild(toc);
      var links = toc.querySelectorAll('a');
      function spy() {
        var y = window.scrollY + 120;
        var cur = heads[0];
        for (var i = 0; i < heads.length; i++) {
          if (heads[i].offsetTop <= y) cur = heads[i];
        }
        for (var j = 0; j < links.length; j++) {
          links[j].classList.toggle('active', links[j].getAttribute('href') === '#' + cur.id);
        }
      }
      window.addEventListener('scroll', spy, { passive: true });
      spy();
    }
  }

  // ── search filter (unchanged behaviour) ──
  if (!$q) return;
  var index = null;
  var indexReady = fetch('/docs/index.json', { cache: 'no-cache' })
    .then(function (r) { return r.ok ? r.json() : []; })
    .then(function (j) { index = Array.isArray(j) ? j : []; })
    .catch(function () { index = []; });

  function matches(entry, q) {
    if (!q) return true;
    q = q.toLowerCase();
    if (entry.title && entry.title.toLowerCase().indexOf(q) !== -1) return true;
    if (entry.slug && entry.slug.indexOf(q) !== -1) return true;
    var kws = entry.keywords || [];
    for (var i = 0; i < kws.length; i++) {
      if (kws[i].indexOf(q) !== -1) return true;
    }
    return false;
  }

  function apply(q) {
    var items = document.querySelectorAll('.doc-sidebar li[data-slug]');
    var cards = document.querySelectorAll('.doc-card');
    var allowed = {};
    if (!q) {
      for (var i = 0; i < items.length; i++) items[i].classList.remove('hidden');
      for (var j = 0; j < cards.length; j++) cards[j].classList.remove('hidden');
      hideEmptyGroups();
      return;
    }
    if (index) {
      for (var k = 0; k < index.length; k++) {
        if (matches(index[k], q)) allowed[index[k].slug] = true;
      }
    } else {
      for (var m = 0; m < items.length; m++) {
        var a = items[m].querySelector('a');
        if (a && a.textContent.toLowerCase().indexOf(q.toLowerCase()) !== -1) {
          allowed[items[m].getAttribute('data-slug')] = true;
        }
      }
    }
    for (var x = 0; x < items.length; x++) {
      var slug = items[x].getAttribute('data-slug');
      if (allowed[slug]) items[x].classList.remove('hidden');
      else items[x].classList.add('hidden');
    }
    for (var y = 0; y < cards.length; y++) {
      var href = cards[y].getAttribute('href') || '';
      var slugMatch = href.match(/\/docs\/([^.]+)\.html/);
      var cardSlug = slugMatch ? slugMatch[1] : '';
      if (allowed[cardSlug]) cards[y].classList.remove('hidden');
      else cards[y].classList.add('hidden');
    }
    hideEmptyGroups();
  }

  function hideEmptyGroups() {
    var groups = document.querySelectorAll('.doc-sidebar .group');
    for (var i = 0; i < groups.length; i++) {
      var visible = groups[i].querySelectorAll('li:not(.hidden)').length;
      if (visible === 0) groups[i].classList.add('empty');
      else groups[i].classList.remove('empty');
    }
    var landingGroups = document.querySelectorAll('.doc-group');
    for (var j = 0; j < landingGroups.length; j++) {
      var vis = landingGroups[j].querySelectorAll('.doc-card:not(.hidden)').length;
      if (vis === 0) landingGroups[j].classList.add('empty');
      else landingGroups[j].classList.remove('empty');
    }
  }

  var t = null;
  $q.addEventListener('input', function (e) {
    var q = (e.target.value || '').trim();
    clearTimeout(t);
    if (!index) {
      indexReady.then(function () { apply(q); });
      return;
    }
    t = setTimeout(function () { apply(q); }, 30);
  });

  document.addEventListener('keydown', function (e) {
    if (e.key === '/' && document.activeElement !== $q) {
      e.preventDefault();
      $q.focus();
      $q.select();
    } else if (e.key === 'Escape' && document.activeElement === $q) {
      $q.value = '';
      apply('');
      $q.blur();
    }
  });
})();
