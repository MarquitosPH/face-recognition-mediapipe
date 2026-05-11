  const shape = sessionStorage.getItem('lm_face_shape');
    if (shape) {
        setTimeout(() => { window.location.href = `/resultados?face=${shape}`; }, 2000);
    }