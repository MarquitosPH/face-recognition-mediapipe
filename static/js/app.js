/* ════════════════════════════════════════════════
   OptiMatch — app.js  Shared JavaScript
   ════════════════════════════════════════════════ */

const API = {
    async uploadPhoto(file) {
        const fd = new FormData();
        fd.append('file', file);
        const res = await fetch('/api/upload-photo', { method: 'POST', body: fd });
        return res.json();
    },

    async getGlasses() {
        const res = await fetch('/api/glasses');
        return res.json();
    },

    async getGlass(id) {
        const res = await fetch(`/api/glasses/${id}`);
        return res.json();
    },

    async getRecommendations(faceShape) {
        const res = await fetch(`/api/recommendations/${faceShape}`);
        return res.json();
    }
};

/* ── Utility helpers ──────────────────────── */
function debounce(fn, ms = 300) {
    let t;
    return (...args) => {
        clearTimeout(t);
        t = setTimeout(() => fn(...args), ms);
    };
}

function showToast(msg, type = 'info') {
    const toast = document.createElement('div');
    toast.style.cssText = `
        position:fixed;bottom:24px;left:50%;transform:translateX(-50%);
        padding:12px 24px;border-radius:8px;font-size:0.88rem;font-weight:600;
        color:white;z-index:9999;animation:fadeIn 0.3s ease;
        background:${type==='error'?'#EF4444':type==='success'?'#22C55E':'#3B82F6'};
        box-shadow:0 4px 20px rgba(0,0,0,0.2);
    `;
    toast.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}
