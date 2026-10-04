document.addEventListener('DOMContentLoaded', () => {
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const resumeInput = document.getElementById('resumeInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const errorBox = document.getElementById('errorBox');
    
    // Result elements
    const resultsSection = document.getElementById('resultsSection');
    const predictedRole = document.getElementById('predictedRole');
    const confidence = document.getElementById('confidence');
    const topPredictionsList = document.getElementById('topPredictionsList');
    const scoreCirclePath = document.getElementById('scoreCirclePath');
    const finalScorePct = document.getElementById('finalScorePct');
    const breakdownBars = document.getElementById('breakdownBars');
    const detectedSkills = document.getElementById('detectedSkills');
    const missingSkills = document.getElementById('missingSkills');

    let currentFile = null;

    // Drag and drop handlers
    uploadZone.addEventListener('click', () => fileInput.click());
    
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        currentFile = file;
        const p = uploadZone.querySelector('p');
        p.innerHTML = `Selected: <strong>${file.name}</strong><br><span>Click to change</span>`;
        resumeInput.value = ''; // Clear text if file selected
    }

    resumeInput.addEventListener('input', () => {
        if (resumeInput.value.trim().length > 0 && currentFile) {
            currentFile = null;
            const p = uploadZone.querySelector('p');
            p.innerHTML = `Click to upload or drag and drop<br><span>PDF, PNG, JPG (Max 10MB)</span>`;
            fileInput.value = '';
        }
    });

    function setLoading(isLoading) {
        analyzeBtn.disabled = isLoading;
        analyzeBtn.textContent = isLoading ? 'Analyzing...' : 'Analyze Resume';
        const loader = document.getElementById('loader');
        if (loader) {
            loader.classList.toggle('hidden', !isLoading);
        }
    }

    analyzeBtn.addEventListener('click', async () => {
        errorBox.classList.add('hidden');
        resultsSection.classList.add('hidden');

        const text = resumeInput.value.trim();
        if (!text && !currentFile) {
            showError("Please paste a resume or upload a file.");
            return;
        }

        setLoading(true);

        try {
            let result;
            if (currentFile) {
                const formData = new FormData();
                formData.append('upload', currentFile, currentFile.name);

                const response = await fetch('/predict-file', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const err = await response.json().catch(() => ({}));
                    throw new Error(err.detail || "File analysis failed.");
                }
                result = await response.json();
            } else {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ resume_text: text })
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || "Analysis failed");
                }
                result = await response.json();
            }

            renderResults(result);

        } catch (err) {
            showError(err.message || 'Analysis failed.');
        } finally {
            setLoading(false);
        }
    });

    function renderResults(data) {
        // Primary Prediction
        predictedRole.textContent = data.predicted_role;
        const confPct = data.confidence ? Math.round(data.confidence * 100) : 0;
        confidence.textContent = `${confPct}% Confidence`;

        // Top Predictions
        topPredictionsList.innerHTML = '';
        if (data.top_predictions && data.top_predictions.length > 1) {
            data.top_predictions.slice(1).forEach(pred => {
                const div = document.createElement('div');
                div.className = 'pred-row';
                const pConf = Math.round(pred.confidence * 100);
                div.innerHTML = `<span>${pred.role}</span><span>${pConf}%</span>`;
                topPredictionsList.appendChild(div);
            });
        }

        // Screening Score
        if (data.screening_breakdown) {
            const finalScore = data.screening_breakdown.final_score_pct;
            finalScorePct.textContent = `${finalScore}%`;
            // stroke-dasharray="0, 100" -> "finalScore, 100"
            setTimeout(() => {
                scoreCirclePath.setAttribute('stroke-dasharray', `${finalScore}, 100`);
            }, 100);

            // Bars
            breakdownBars.innerHTML = '';
            const comps = data.screening_breakdown.components;
            const labels = {
                'role_match': 'Role Alignment',
                'skills_match': 'Required Skills',
                'experience_match': 'Experience Indicator',
                'education_match': 'Education Indicator'
            };
            
            for (const [key, val] of Object.entries(comps)) {
                const div = document.createElement('div');
                div.className = 'bar-row';
                div.innerHTML = `
                    <div class="bar-label">
                        <span>${labels[key]}</span>
                        <span>${val.score_pct}%</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: 0%"></div>
                    </div>
                `;
                breakdownBars.appendChild(div);
                
                // Animate bar
                setTimeout(() => {
                    div.querySelector('.bar-fill').style.width = `${val.score_pct}%`;
                }, 100);
            }
        }

        // Skills
        detectedSkills.innerHTML = '';
        if (data.detected_skills && data.detected_skills.length) {
            data.detected_skills.forEach(skill => {
                const s = document.createElement('span');
                s.className = 'skill-badge';
                s.textContent = skill;
                detectedSkills.appendChild(s);
            });
        } else {
            detectedSkills.innerHTML = '<span class="text-secondary">No technical skills detected.</span>';
        }

        missingSkills.innerHTML = '';
        if (data.missing_common_skills && data.missing_common_skills.length) {
            data.missing_common_skills.forEach(skill => {
                const s = document.createElement('span');
                s.className = 'skill-badge';
                s.textContent = skill;
                missingSkills.appendChild(s);
            });
        } else {
            missingSkills.innerHTML = '<span class="text-secondary">None.</span>';
        }

        resultsSection.classList.remove('hidden');
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    function showError(msg) {
        errorBox.textContent = msg;
        errorBox.classList.remove('hidden');
    }
});