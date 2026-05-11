document.addEventListener('DOMContentLoaded', () => {
    const shape = sessionStorage.getItem('lm_face_shape');
    const conf = sessionStorage.getItem('lm_confidence');
    if (shape) {
        document.getElementById('noAnalysis').style.display = 'none';
        document.getElementById('analysisData').style.display = 'block';
        document.getElementById('aType').textContent = shape;
        document.getElementById('aConf').textContent = `Confianza: ${conf || '?'}%`;

        const descs = {
            'Ovalado': 'Rostro equilibrado con proporciones armónicas. Los pómulos son tu zona más ancha y el mentón se suaviza gradualmente.',
            'Redondo': 'Rostro con líneas suaves, casi tan ancho como largo. Tus pómulos son amplios y el mentón es redondeado.',
            'Cuadrado': 'Rostro con mandíbula angular marcada. Frente, pómulos y mandíbula presentan un ancho similar.',
            'Rectángulo': 'Rostro notablemente más largo que ancho con ángulos definidos y mandíbula pronunciada.',
            'Triángulo': 'Tu mandíbula es la zona más ancha del rostro, con frente más estrecha.',
            'Corazón': 'Frente amplia con mentón puntiagudo. La mandíbula es más estrecha que la zona superior.',
            'Diamante': 'Pómulos prominentes dominan el ancho facial. Frente y mandíbula son más estrechas.',
        };
        document.getElementById('aDesc').textContent = descs[shape] || '';

        const chars = {
            'Ovalado': ['Equilibrado', 'Pómulos prominentes', 'Mentón suave', 'Frente proporcionada'],
            'Redondo': ['Líneas suaves', 'Pómulos amplios', 'Mentón redondeado', 'Ancho ≈ alto'],
            'Cuadrado': ['Mandíbula angular', 'Frente amplia', 'Mentón cuadrado', 'Líneas definidas'],
            'Rectángulo': ['Cara alargada', 'Mandíbula definida', 'Frente amplia', 'Ángulos marcados'],
            'Triángulo': ['Mandíbula ancha', 'Frente estrecha', 'Base amplia', 'Mentón ancho'],
            'Corazón': ['Frente amplia', 'Mentón puntiagudo', 'Pómulos altos', 'Forma de V invertida'],
            'Diamante': ['Pómulos dominantes', 'Frente estrecha', 'Mentón definido', 'Forma angular'],
        };
        const charsEl = document.getElementById('aChars');
        (chars[shape] || []).forEach(c => {
            charsEl.innerHTML += `<span class="badge badge-blue" style="font-size:0.85rem;padding:8px 14px;">${c}</span>`;
        });
    }
});