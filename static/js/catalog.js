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
// MOSTRAR COMPATIBILIDAD SOLO SI HAY ANÁLISIS PREVIO
// (Nielsen #1 visibilidad real, #5 prevenir falsa información)
// ============================================
const savedFaceShape = localStorage.getItem('lm_face_shape');
if (savedFaceShape) {
    document.querySelectorAll('.product-compat').forEach(el => {
        el.style.display = 'flex';
    });
}

if (window.lucide) lucide.createIcons();