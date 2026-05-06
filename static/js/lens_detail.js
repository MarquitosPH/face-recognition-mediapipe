// ============================================
// STAGE (galería sincronizada en ambos paneles)
// ============================================
let currentStage = 0;
const lifestyleImgs = document.querySelectorAll('.stage-img-lifestyle');
const productImgs = document.querySelectorAll('.stage-img-product');
const stageDots = document.querySelectorAll('.stage-dot');
const totalStages = lifestyleImgs.length;

function goToStage(index) {
    if (index < 0 || index >= totalStages) return;
    currentStage = index;

    // Para crear efecto Mykita: alternamos qué imagen va en cada panel
    // Panel izquierdo muestra la imagen actual, derecho muestra la siguiente
    lifestyleImgs.forEach((img, i) => img.classList.toggle('active', i === index));
    productImgs.forEach((img, i) => {
        const productIdx = (index + 1) % totalStages;
        img.classList.toggle('active', i === productIdx);
    });
    stageDots.forEach((d, i) => d.classList.toggle('active', i === index));
}

function navStage(direction) {
    const next = (currentStage + direction + totalStages) % totalStages;
    goToStage(next);
}

// Inicializar el panel derecho con la segunda imagen (efecto Mykita)
if (totalStages > 1) {
    productImgs.forEach((img, i) => img.classList.toggle('active', i === 1));
}

// Teclado
document.addEventListener('keydown', (e) => {
    if (totalStages <= 1) return;
    if (e.key === 'ArrowLeft') navStage(-1);
    if (e.key === 'ArrowRight') navStage(1);
});

// Swipe táctil en el stage
const stageGrid = document.querySelector('.stage-grid');
if (stageGrid && totalStages > 1) {
    let touchStartX = 0;
    stageGrid.addEventListener('touchstart', (e) => {
        touchStartX = e.changedTouches[0].screenX;
    }, { passive: true });
    stageGrid.addEventListener('touchend', (e) => {
        const diff = touchStartX - e.changedTouches[0].screenX;
        if (Math.abs(diff) > 50) navStage(diff > 0 ? 1 : -1);
    }, { passive: true });
}

// ============================================
// TABS
// ============================================
function switchTab(tab) {
    document.querySelectorAll('.specs-tab').forEach(t => {
        t.classList.toggle('active', t.dataset.tab === tab);
    });
    document.querySelectorAll('.specs-content').forEach(c => {
        c.classList.toggle('active', c.id === `tab-${tab}`);
    });
}

// ============================================
// FAVORITO
// ============================================
function toggleFav(btn) {
    btn.classList.toggle('active');
}

// ============================================
// MATCH CONTEXTUAL CON ROSTRO DETECTADO
// (Nielsen #10: ayuda contextual - solo si hay análisis previo)
// ============================================
const savedShape = localStorage.getItem('lm_face_shape');
const compatibility = parseInt({{ glass.compatibility|default(0) }});

if (savedShape) {
    // 1. Activar pill de match con la compatibilidad real
    const matchPill = document.getElementById('matchPill');
    const matchPillText = document.getElementById('matchPillText');
    if (matchPill && matchPillText) {
        if (compatibility >= 85) {
            matchPillText.textContent = `Top Match · ${compatibility}%`;
        } else if (compatibility) {
            matchPillText.textContent = `${compatibility}% compatible`;
        }
        matchPill.classList.add('visible');
    }

    // 2. Buscar coincidencia con rostro guardado
    const chips = document.querySelectorAll('.face-chip');
    let isCompatible = false;

    chips.forEach(chip => {
        if (chip.dataset.face && chip.dataset.face.toLowerCase() === savedShape.toLowerCase()) {
            chip.classList.add('highlight');
            isCompatible = true;
        }
    });

    if (isCompatible) {
        const block = document.getElementById('faceMatchBlock');
        const text = document.getElementById('faceMatchText');
        if (block && text) {
            text.textContent = `Compatible con tu rostro ${savedShape}`;
            block.classList.add('visible');
        }
    }
}

if (window.lucide) lucide.createIcons();