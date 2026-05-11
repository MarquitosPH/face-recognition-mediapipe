let cameraStream = null;
let analyzingTimer = null;

const MESSAGES = [
    'Detectando rostro...',
    'Analizando rasgos faciales...',
    'Calculando proporciones...',
    'Encontrando tu estilo ideal...'
];

function setState(state) {
    document.querySelectorAll('.action-group').forEach(g => {
        g.classList.toggle('active', g.dataset.state === state);
    });
}

function triggerFileInput() { document.getElementById('fileInput').click(); }

async function startCapture() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720, facingMode: 'user' }
        });
        cameraStream = stream;
        const video = document.getElementById('cameraFeed');
        video.srcObject = stream;
        video.style.display = 'block';
        document.getElementById('capturePlaceholder').style.display = 'none';
        document.getElementById('uploadPreview').style.display = 'none';
        document.getElementById('faceGuide').classList.add('visible');
        hideError();
        setState('camera');
    } catch (err) {
        showError('No se pudo acceder a la cámara. Verifica los permisos en tu navegador.');
    }
}

function cancelCapture() { stopCamera(); resetToInitial(); }

function takePhoto() {
    const video = document.getElementById('cameraFeed');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    canvas.toBlob(async (blob) => {
        stopCamera();
        const preview = document.getElementById('uploadPreview');
        preview.src = URL.createObjectURL(blob);
        preview.style.display = 'block';
        document.getElementById('faceGuide').classList.remove('visible');
        await analyzeImage(blob);
    }, 'image/jpeg', 0.9);
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
    }
    const video = document.getElementById('cameraFeed');
    video.style.display = 'none';
    video.srcObject = null;
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    if (!file.type.startsWith('image/')) {
        showError('Por favor selecciona una imagen válida (JPG o PNG).');
        return;
    }
    const preview = document.getElementById('uploadPreview');
    preview.src = URL.createObjectURL(file);
    preview.style.display = 'block';
    document.getElementById('capturePlaceholder').style.display = 'none';
    document.getElementById('cameraFeed').style.display = 'none';
    document.getElementById('faceGuide').classList.remove('visible');
    hideError();
    analyzeImage(file);
}

async function analyzeImage(blob) {
    setState('analyzing');
    showAnalyzingOverlay();
    try {
        const [result] = await Promise.all([
            API.uploadPhoto(blob),
            new Promise(r => setTimeout(r, 2400))
        ]);
        hideAnalyzingOverlay();
        if (result && result.success) {
            showResult(result);
        } else {
            setState('initial');
            resetToInitial();
            showError(result?.error || 'No se detectó un rostro. Intenta con mejor iluminación mirando de frente.');
        }
    } catch (err) {
        hideAnalyzingOverlay();
        setState('initial');
        resetToInitial();
        showError('Error al analizar la imagen. Intenta de nuevo.');
    }
}

function showAnalyzingOverlay() {
    const overlay = document.getElementById('analyzingOverlay');
    const textEl = document.getElementById('analyzingText');
    overlay.classList.add('active');
    let idx = 0;
    textEl.textContent = MESSAGES[0];
    analyzingTimer = setInterval(() => {
        idx = (idx + 1) % MESSAGES.length;
        textEl.style.opacity = '0';
        setTimeout(() => {
            textEl.textContent = MESSAGES[idx];
            textEl.style.opacity = '1';
        }, 200);
    }, 900);
}

function hideAnalyzingOverlay() {
    document.getElementById('analyzingOverlay').classList.remove('active');
    if (analyzingTimer) clearInterval(analyzingTimer);
    analyzingTimer = null;
}

function showResult(result) {
    document.getElementById('resultType').textContent = result.face_shape || '—';
    document.getElementById('resultConf').textContent = `Confianza: ${result.confidence}%`;
    document.getElementById('resultDesc').textContent = result.description || '';
    const charsEl = document.getElementById('resultChars');
    charsEl.innerHTML = '';
    (result.characteristics || []).forEach(c => {
        const span = document.createElement('span');
        span.className = 'char-badge';
        span.textContent = c;
        charsEl.appendChild(span);
    });
    const card = document.getElementById('resultCard');
    card.style.display = 'block';
    requestAnimationFrame(() => card.classList.add('visible'));
    document.getElementById('btnResults').href = `/resultados?face=${encodeURIComponent(result.face_shape)}`;
    try {
        sessionStorage.setItem('lm_face_shape', result.face_shape);
        sessionStorage.setItem('lm_confidence', result.confidence);
    } catch (e) {}
    setState('result');
    setTimeout(() => {
        card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 300);
}

function resetCapture() { stopCamera(); resetToInitial(); }

function resetToInitial() {
    document.getElementById('uploadPreview').style.display = 'none';
    document.getElementById('uploadPreview').src = '';
    document.getElementById('cameraFeed').style.display = 'none';
    document.getElementById('capturePlaceholder').style.display = 'flex';
    document.getElementById('faceGuide').classList.remove('visible');
    const card = document.getElementById('resultCard');
    card.classList.remove('visible');
    setTimeout(() => { card.style.display = 'none'; }, 400);
    document.getElementById('fileInput').value = '';
    hideError();
    setState('initial');
}

function showError(text) {
    const errEl = document.getElementById('errorMsg');
    document.getElementById('errorText').textContent = text;
    errEl.classList.add('visible');
}

function hideError() { document.getElementById('errorMsg').classList.remove('visible'); }

window.addEventListener('beforeunload', stopCamera);

if (window.lucide) lucide.createIcons();