  // ============================================
    // HERO SLIDER
    // ============================================
    (function () {
        const slides = document.querySelectorAll('.slide');
        const dots = document.querySelectorAll('.slider-dot');
        const totalSlides = slides.length;
        let currentSlide = 0;
        let autoplayTimer = null;
        const AUTOPLAY_INTERVAL = 7000;

        function showSlide(index) {
            slides.forEach((s, i) => {
                const isActive = i === index;
                s.classList.toggle('active', isActive);

                const video = s.querySelector('video');
                if (video && video.dataset.loaded === 'true') {
                    if (isActive) {
                        try { video.currentTime = 0; } catch (e) {}
                        const p = video.play();
                        if (p !== undefined) p.catch(() => {});
                    } else {
                        video.pause();
                    }
                }
            });
            dots.forEach((d, i) => d.classList.toggle('active', i === index));
            currentSlide = index;
        }

        window.changeSlide = function (direction) {
            const next = (currentSlide + direction + totalSlides) % totalSlides;
            showSlide(next);
            resetAutoplay();
        };

        window.goToSlide = function (index) {
            showSlide(index);
            resetAutoplay();
        };

        function startAutoplay() {
            autoplayTimer = setInterval(() => changeSlide(1), AUTOPLAY_INTERVAL);
        }

        function resetAutoplay() {
            if (autoplayTimer) clearInterval(autoplayTimer);
            startAutoplay();
        }

        const slider = document.getElementById('heroSlider');
        if (slider) {
            slider.addEventListener('mouseenter', () => {
                if (autoplayTimer) clearInterval(autoplayTimer);
            });
            slider.addEventListener('mouseleave', startAutoplay);

            let touchStartX = 0, touchEndX = 0;
            slider.addEventListener('touchstart', (e) => {
                touchStartX = e.changedTouches[0].screenX;
            }, { passive: true });
            slider.addEventListener('touchend', (e) => {
                touchEndX = e.changedTouches[0].screenX;
                const diff = touchStartX - touchEndX;
                if (Math.abs(diff) > 50) changeSlide(diff > 0 ? 1 : -1);
            }, { passive: true });
        }

        document.querySelectorAll('.slide video').forEach((video, idx) => {
            const source = video.querySelector('source');
            const src = source ? (source.getAttribute('src') || '').trim() : '';
            if (src) {
                video.dataset.loaded = 'true';
                if (idx === 0) {
                    const p = video.play();
                    if (p !== undefined) p.catch(() => {});
                }
            }
        });

        startAutoplay();
    })();

    // ============================================
    // CARRUSEL DE ESTILOS
    // ============================================
    function scrollLabels(direction) {
        const track = document.getElementById('labelsTrack');
        if (!track) return;
        track.scrollBy({ left: direction * 278, behavior: 'smooth' });
    }

    // Re-renderizar iconos Lucide (por si el base.html no lo hace)
    if (window.lucide) lucide.createIcons();