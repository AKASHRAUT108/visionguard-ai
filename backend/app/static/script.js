async function analyze() {
    const imageFile = document.getElementById('imageInput').files[0];
    const text = document.getElementById('textInput').value;
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
        <p>${data.clip_similarity.toFixed(3)}</p>
    `;
}

// async function ask() {
//     const question = document.getElementById('questionInput').value;
//     const formData = new FormData();
//     formData.append('question', question);

//     const response = await fetch('/chat', {
//         method: 'POST',
//         body: formData
//     });
//     const data = await response.json();
//     document.getElementById('chatAnswer').innerText = data.answer;
// }

async function getPrediction() {
    const response = await fetch('/predict_failure');
    const data = await response.json();
    document.getElementById('predictionResult').innerText = `Failure probability: ${data.failure_probability * 100}%`;
}