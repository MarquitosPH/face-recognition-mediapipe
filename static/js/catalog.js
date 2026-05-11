let currentFilter = 'all';

function setFilter(filter) {
    currentFilter = filter;
    document.querySelectorAll('.filter-chip').forEach(chip => {
        chip.classList.toggle('active', chip.dataset.filter === filter);
    });
    applyFilters();
}

function applyFilters() {
    const query = document.getElementById('searchInput').value.toLowerCase().trim();
    const sortBy = document.getElementById('sortSelect').value;
    const grid = document.getElementById('catalogGrid');
    const emptyState = document.getElementById('emptyState');

    const cards = Array.from(document.querySelectorAll('.product-card'));
    let visibleCount = 0;

    cards.forEach(card => {
        const name = card.dataset.name || '';
        const brand = card.dataset.brand || '';
        const style = card.dataset.style || '';
        const tags = card.dataset.tags || '';

        const matchesSearch = !query ||
            name.includes(query) || brand.includes(query) || style.includes(query);

        const matchesFilter = currentFilter === 'all' ||
            style.includes(currentFilter) || tags.includes(currentFilter);

        const visible = matchesSearch && matchesFilter;
        card.style.display = visible ? '' : 'none';
        if (visible) visibleCount++;
    });

    if (sortBy !== 'default') {
        const visibleCards = cards.filter(c => c.style.display !== 'none');
        visibleCards.sort((a, b) => {
            if (sortBy === 'name-asc') return a.dataset.name.localeCompare(b.dataset.name);
            if (sortBy === 'name-desc') return b.dataset.name.localeCompare(a.dataset.name);
            if (sortBy === 'compat-desc') return parseInt(b.dataset.compat) - parseInt(a.dataset.compat);
            return 0;
        });
        visibleCards.forEach(c => grid.appendChild(c));
    }

    document.getElementById('resultCount').textContent = visibleCount;
    grid.style.display = visibleCount === 0 ? 'none' : 'grid';
    emptyState.style.display = visibleCount === 0 ? 'block' : 'none';
}

function toggleFav(btn) {
    btn.classList.toggle('active');
}

// ============================================
// COMPATIBILIDAD PERSONALIZADA
// Solo se muestra si el usuario completó el análisis facial.
// El % se calcula dinámicamente según si su forma de rostro
// está en la lista compatible_faces del lente.
// ============================================
(function applyCompatibility() {
    const faceShape   = sessionStorage.getItem('lm_face_shape');
    const confidence  = sessionStorage.getItem('lm_confidence');

    // Sin análisis previo → no mostrar nada
    if (!faceShape || !confidence) return;

    const userShape = faceShape.trim();

    document.querySelectorAll('.product-compat').forEach(el => {
        let compatFaces = [];
        try { compatFaces = JSON.parse(el.dataset.faces || '[]'); } catch (_) {}

        const isMatch  = compatFaces.some(f => f.trim().toLowerCase() === userShape.toLowerCase());
        const baseScore = parseInt(el.dataset.compat, 10) || 0;

        // Calcular score real: si coincide → valor del DB, si no → penalizar
        const score = isMatch ? baseScore : Math.max(10, Math.round(baseScore * 0.35));

        // Clase del indicador de color
        const dot      = el.querySelector('.compat-dot');
        const textSpan = el.querySelector('.compat-text');

        dot.className = 'compat-dot' +
            (score >= 85 ? '' : score >= 60 ? ' medium' : ' low');

        textSpan.textContent = `${score}% compatible`;
        el.style.display = 'flex';
    });
})();

if (window.lucide) lucide.createIcons();