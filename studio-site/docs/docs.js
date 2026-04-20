// docs.js — tiny client-side filter for sidebar + landing cards.
// Loads /docs/index.json once, filters on every input keystroke.
(function () {
  var $q = document.getElementById('q');
  if (!$q) return;
  var body = document.body;
  var curSlug = body.getAttribute('data-slug');

  // Mark the current page in the sidebar.
  if (curSlug) {
    var li = document.querySelector('.doc-sidebar li[data-slug="' + curSlug + '"]');
    if (li) {
      li.classList.add('active');
      var a = li.querySelector('a');
      if (a) a.classList.add('current');
    }
  }

  var index = null;
  var indexReady = fetch('/docs/index.json', { cache: 'no-cache' })
    .then(function (r) { return r.ok ? r.json() : []; })
    .then(function (j) { index = Array.isArray(j) ? j : []; })
    .catch(function () { index = []; });

  // Normalise once; searching against keywords[] + title.
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
      // Everything visible.
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
      // Index not ready — fall back to title-only substring.
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

  // Keyboard: "/" focuses, Esc clears.
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
