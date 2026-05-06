let activeFilters = { style: 'all', material: 'all', gender: 'all' };

function filterBy(type, value, el) {
    activeFilters[type] = value;
    el.parentElement.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
    el.classList.add('active');
    applyFilters();
    updateClearButton();
}

function clearAllFilters() {
    activeFilters = { style: 'all', material: 'all', gender: 'all' };
    document.querySelectorAll('.filter-options').forEach(group => {
        group.querySelectorAll('.filter-chip').forEach((c, idx) => {
            c.classList.toggle('active', idx === 0);
        });
    });
    applyFilters();
    updateClearButton();
}

function updateClearButton() {
    const anyActive = activeFilters.style !== 'all' ||
                      activeFilters.material !== 'all' ||
                      activeFilters.gender !== 'all';
    document.getElementById('clearBtn').classList.toggle('visible', anyActive);
}

function applyFilters() {
    const sortBy = document.getElementById('sortSelect').value;
    const grid = document.getElementById('glassesGrid');
    const emptyState = document.getElementById('emptyState');

    const cards = Array.from(document.querySelectorAll('.product-card'));
    let visible = 0;

    cards.forEach(card => {
        let show = true;
        if (activeFilters.style !== 'all' && card.dataset.style !== activeFilters.style) show = false;
        if (activeFilters.material !== 'all' && card.dataset.material !== activeFilters.material) show = false;
        if (activeFilters.gender !== 'all' && card.dataset.gender !== activeFilters.gender) show = false;
        card.style.display = show ? '' : 'none';
        if (show) visible++;
    });

    if (sortBy !== 'default') {
        const visibleCards = cards.filter(c => c.style.display !== 'none');
        visibleCards.sort((a, b) => {
            if (sortBy === 'name-asc') return a.dataset.name.localeCompare(b.dataset.name);
            if (sortBy === 'name-desc') return b.dataset.name.localeCompare(a.dataset.name);
            return 0;
        });
        visibleCards.forEach(c => grid.appendChild(c));
    } else {
        const visibleCards = cards.filter(c => c.style.display !== 'none');
        visibleCards.sort((a, b) => parseInt(b.dataset.compat) - parseInt(a.dataset.compat));
        visibleCards.forEach(c => grid.appendChild(c));
    }

    grid.appendChild(emptyState);

    document.getElementById('resultsCount').textContent = visible;
    emptyState.style.display = visible === 0 ? 'block' : 'none';
}

function toggleFav(btn) {
    btn.classList.toggle('active');
}

function toggleMobileFilters() {
    document.getElementById('sidebarFilters').classList.toggle('mobile-open');
}

const savedShape = localStorage.getItem('lm_face_shape');
if (savedShape) {
    document.getElementById('faceBanner').style.display = 'block';
    document.getElementById('faceShapeLabel').textContent = savedShape;
}

if (window.lucide) lucide.createIcons();