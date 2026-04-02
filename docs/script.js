(() => {
  const setYear = () => {
    const yearNode = document.getElementById('last-updated');
    if (yearNode) yearNode.textContent = String(new Date().getFullYear());
  };

  const loadPartials = async () => {
    const mounts = Array.from(document.querySelectorAll('.partial-mount[data-partial]'));
    for (const mount of mounts) {
      const partialPath = mount.getAttribute('data-partial');
      if (!partialPath) continue;
      try {
        const response = await fetch(partialPath, { cache: 'no-store' });
        if (!response.ok) throw new Error(`Failed to load ${partialPath}: ${response.status}`);
        mount.innerHTML = await response.text();
      } catch (error) {
        const message = window.location.protocol === 'file:'
          ? `Failed to load ${partialPath}. Open the page through a local web server.`
          : `Failed to load ${partialPath}.`;
        mount.innerHTML = `<section class="section"><div class="container"><div class="partial-error">${message}</div></div></section>`;
        console.error(error);
      }
      mount.removeAttribute('data-partial');
    }
  };

  const initActiveNav = () => {
    const navLinks = Array.from(document.querySelectorAll('.page-contents-nav a[href^="#"]'));
    const targets = navLinks
      .map((link) => ({ link, id: link.getAttribute('href')?.replace('#', '') || '' }))
      .filter(({ id }) => id)
      .map(({ link, id }) => ({ link, id, node: document.getElementById(id) }))
      .filter(({ node }) => node);

    if (!targets.length) return;

    const setActive = (activeId) => {
      navLinks.forEach((link) => {
        link.classList.remove('active');
        link.removeAttribute('aria-current');
      });
      const active = targets.find(({ id }) => id === activeId);
      if (active) {
        active.link.classList.add('active');
        active.link.setAttribute('aria-current', 'true');
      }
    };

    navLinks.forEach((link) => {
      const id = link.getAttribute('href')?.replace('#', '') || '';
      link.addEventListener('click', () => {
        if (id) setActive(id);
      });
    });

    window.addEventListener('hashchange', () => {
      const hashId = window.location.hash.replace('#', '');
      if (hashId) setActive(hashId);
    });

    const initialHash = window.location.hash.replace('#', '');
    if (initialHash) setActive(initialHash);

    if (!('IntersectionObserver' in window)) return;

    const observer = new IntersectionObserver((entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      const currentId = visible.target.getAttribute('id');
      if (currentId) setActive(currentId);
    }, { rootMargin: '-20% 0px -55% 0px', threshold: [0.05, 0.2, 0.4] });

    targets.forEach(({ node }) => observer.observe(node));
  };

  const initCopyCitation = () => {
    const copyBtn = document.getElementById('copy-citation');
    const status = document.getElementById('copy-status');
    if (!copyBtn) return;
    const setStatus = (message) => { if (status) status.textContent = message; };
    copyBtn.addEventListener('click', async () => {
      const targetId = copyBtn.getAttribute('data-copy-target');
      const target = targetId ? document.getElementById(targetId) : null;
      if (!target) return;
      try {
        await navigator.clipboard.writeText(target.textContent || '');
        setStatus('Copied.');
      } catch {
        setStatus('Clipboard blocked. Copy manually.');
      }
      window.setTimeout(() => setStatus(''), 1800);
    });
  };

  let playbackObserver;
  let previewObserver;
  const getPlaybackObserver = () => {
    if (playbackObserver || !('IntersectionObserver' in window)) return playbackObserver;
    playbackObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        const video = entry.target;
        if (!(video instanceof HTMLVideoElement)) return;
        if (!entry.isIntersecting && !video.paused) video.pause();
      });
    }, { threshold: 0.1 });
    return playbackObserver;
  };

  const ensureReplayVideo = (player) => {
    const src = (player.dataset.videoSrc || '').trim();
    let video = player.querySelector('video');
    if (!src) return video;
    if (!video) {
      video = document.createElement('video');
      player.prepend(video);
    }
    if (video.dataset.initialized === 'true') return video;
    video.className = 'replay-video';
    video.controls = true;
    video.playsInline = true;
    video.preload = 'none';
    video.setAttribute('aria-label', player.dataset.posterTitle || 'Replay video');
    if (player.dataset.posterImage) video.poster = player.dataset.posterImage;
    let source = video.querySelector('source');
    if (!source) {
      source = document.createElement('source');
      video.appendChild(source);
    }
    source.src = src;
    source.type = player.dataset.videoType || 'video/mp4';
    video.dataset.initialized = 'true';
    video.addEventListener('play', () => player.classList.add('is-activated'));
    const observer = getPlaybackObserver();
    if (observer) observer.observe(video);
    return video;
  };

  const primeReplayVideo = (video) => {
    if (!video || video.dataset.previewLoaded === 'true') return;
    video.preload = 'metadata';
    const markPreviewLoaded = () => {
      video.dataset.previewLoaded = 'true';
      video.classList.add('has-frame');
    };
    video.addEventListener('loadeddata', markPreviewLoaded, { once: true });
    video.load();
  };

  const getPreviewObserver = () => {
    if (previewObserver || !('IntersectionObserver' in window)) return previewObserver;
    previewObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const video = entry.target;
        if (video instanceof HTMLVideoElement) {
          primeReplayVideo(video);
          previewObserver.unobserve(video);
        }
      });
    }, { rootMargin: '240px 0px', threshold: 0.01 });
    return previewObserver;
  };

  const initReplayPlayers = () => {
    const players = document.querySelectorAll('.replay-player[data-video-src]');
    players.forEach((player) => {
      const video = ensureReplayVideo(player);
      if (!video) return;
      const preview = getPreviewObserver();
      if (preview) {
        preview.observe(video);
      } else {
        primeReplayVideo(video);
      }
      const trigger = player.querySelector('.replay-launch');
      if (!trigger) return;
      trigger.addEventListener('click', () => {
        player.classList.add('is-activated');
        primeReplayVideo(video);
        const playPromise = video.play();
        if (playPromise && typeof playPromise.catch === 'function') playPromise.catch(() => {});
      });
    });
  };

  const initMediaFallbacks = () => {
    document.querySelectorAll('.media-thumb').forEach((thumb) => {
      const img = thumb.querySelector('img');
      if (!img) return;
      const markLoaded = () => {
        if (img.naturalWidth > 0) thumb.classList.add('loaded');
      };
      if (img.complete) markLoaded();
      img.addEventListener('load', markLoaded);
      img.addEventListener('error', () => thumb.classList.remove('loaded'));
    });
  };

  const restoreHashTarget = () => {
    if (!window.location.hash) return;
    const target = document.querySelector(window.location.hash);
    if (!target) return;
    window.requestAnimationFrame(() => {
      target.scrollIntoView({ block: 'start' });
    });
  };

  const boot = async () => {
    await loadPartials();
    restoreHashTarget();
    setYear();
    initActiveNav();
    initCopyCitation();
    initReplayPlayers();
    initMediaFallbacks();
  };

  boot();
})();
