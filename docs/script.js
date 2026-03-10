(() => {
  const navLinks = Array.from(document.querySelectorAll('.nav-list a'));
  const sections = navLinks
    .map((link) => document.querySelector(link.getAttribute('href')))
    .filter(Boolean);

  const setYear = () => {
    const yearNode = document.getElementById('last-updated');
    if (yearNode) {
      yearNode.textContent = String(new Date().getFullYear());
    }
  };

  const initActiveNav = () => {
    if (!sections.length || !navLinks.length) return;

    const mapById = new Map(
      navLinks.map((link) => [link.getAttribute('href').replace('#', ''), link]),
    );

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          const currentId = entry.target.getAttribute('id');
          navLinks.forEach((link) => link.classList.remove('active'));
          const activeLink = mapById.get(currentId);
          if (activeLink) activeLink.classList.add('active');
        });
      },
      { rootMargin: '-30% 0px -55% 0px', threshold: 0.05 },
    );

    sections.forEach((section) => observer.observe(section));
  };

  const initCopyCitation = () => {
    const copyBtn = document.getElementById('copy-citation');
    const status = document.getElementById('copy-status');
    if (!copyBtn) return;

    const setStatus = (message) => {
      if (status) status.textContent = message;
    };

    copyBtn.addEventListener('click', async () => {
      const targetId = copyBtn.getAttribute('data-copy-target');
      const target = targetId ? document.getElementById(targetId) : null;
      if (!target) return;

      const text = target.textContent || '';
      try {
        await navigator.clipboard.writeText(text);
        setStatus('Copied.');
      } catch {
        setStatus('Clipboard blocked. Copy manually.');
      }
      window.setTimeout(() => setStatus(''), 1800);
    });
  };

  const initMediaFallbacks = () => {
    const imageThumbs = document.querySelectorAll('.media-thumb');
    imageThumbs.forEach((thumb) => {
      const img = thumb.querySelector('img');
      if (!img) return;
      const markLoaded = () => {
        if (img.naturalWidth > 0) thumb.classList.add('loaded');
      };
      if (img.complete) markLoaded();
      img.addEventListener('load', markLoaded);
      img.addEventListener('error', () => thumb.classList.remove('loaded'));
    });

    const replayFrames = document.querySelectorAll('.replay-frame');
    replayFrames.forEach((frame) => {
      const video = frame.querySelector('video');
      if (!video) return;
      video.addEventListener('loadeddata', () => frame.classList.add('loaded'));
      video.addEventListener('error', () => frame.classList.remove('loaded'));
    });
  };

  setYear();
  initActiveNav();
  initCopyCitation();
  initMediaFallbacks();
})();
