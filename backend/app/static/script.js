async function analyze() {
    const imageFile = document.getElementById('imageInput').files[0];
    const text = document.getElementById('textInput').value;  // we need a text input in HTML
    const formData = new FormData();
    formData.append('file', imageFile);
    formData.append('text', text);

    const response = await fetch('/analyze', {
        method: 'POST',
        body: formData
    });
    const data = await response.json();
    document.getElementById('results').innerHTML = `
        <h3>Defect</h3>
        <pre>${JSON.stringify(data.defect, null, 2)}</pre>
        <h3>Root Cause</h3>
        <pre>${JSON.stringify(data.root_cause, null, 2)}</pre>
        <h3>CLIP Similarity</h3>
        <p>${data.clip_similarity}</p>
    `;
}