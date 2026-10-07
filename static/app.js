const COMPANY_RECOMMENDATIONS = {
    python: [
        { name: 'Zoho', focus: ['Python', 'SQL', 'Django', 'Software Development'] },
        { name: 'Freshworks', focus: ['Python', 'REST APIs', 'Cloud', 'Software Development'] },
        { name: 'Razorpay', focus: ['Python', 'SQL', 'REST APIs', 'Data Analysis'] },
        { name: 'Infosys', focus: ['Python', 'SQL', 'Cloud', 'Project Management'] },
        { name: 'TCS', focus: ['Python', 'SQL', 'Java', 'Project Management'] },
        { name: 'Accenture', focus: ['Python', 'Cloud', 'Data Analysis', 'Project Management'] }
    ],
    data: [
        { name: 'Tiger Analytics', focus: ['Python', 'Machine Learning', 'Data Analysis', 'SQL'] },
        { name: 'Mu Sigma', focus: ['Python', 'Data Analysis', 'SQL', 'Statistics'] },
        { name: 'Fractal Analytics', focus: ['Python', 'Machine Learning', 'Data Science', 'SQL'] },
        { name: 'TCS', focus: ['Python', 'SQL', 'Data Analysis', 'Cloud'] },
        { name: 'Accenture', focus: ['Python', 'Data Analysis', 'Machine Learning', 'Cloud'] },
        { name: 'Deloitte', focus: ['Data Analysis', 'SQL', 'Statistics', 'Power BI'] }
    ],
    software: [
        { name: 'Zoho', focus: ['Java', 'Python', 'SQL', 'Software Development'] },
        { name: 'Freshworks', focus: ['Java', 'Python', 'REST APIs', 'Cloud'] },
        { name: 'Infosys', focus: ['Java', 'Python', 'SQL', 'Cloud'] },
        { name: 'TCS', focus: ['Java', 'Python', 'SQL', 'Project Management'] },
        { name: 'Wipro', focus: ['Java', 'Python', 'Cloud', 'Software Development'] },
        { name: 'Accenture', focus: ['Java', 'Python', 'Cloud', 'REST APIs'] }
    ],
    web: [
        { name: 'Freshworks', focus: ['JavaScript', 'React', 'HTML', 'CSS'] },
        { name: 'Zoho', focus: ['JavaScript', 'Python', 'HTML', 'CSS'] },
        { name: 'Razorpay', focus: ['JavaScript', 'React', 'REST APIs', 'CSS'] },
        { name: 'Infosys', focus: ['JavaScript', 'React', 'HTML', 'CSS'] },
        { name: 'TCS', focus: ['JavaScript', 'HTML', 'CSS', 'Java'] },
        { name: 'Accenture', focus: ['JavaScript', 'React', 'Cloud', 'REST APIs'] }
    ],
    hr: [
        { name: 'Zoho', focus: ['Recruiting', 'Employee Relations', 'HRIS', 'Human Resources'] },
        { name: 'Freshworks', focus: ['Recruiting', 'Onboarding', 'Communication', 'Interviewing'] },
        { name: 'TCS', focus: ['Recruiting', 'Employee Relations', 'Communication', 'Interviewing'] },
        { name: 'Infosys', focus: ['Recruiting', 'Training', 'Onboarding', 'Human Resources'] },
        { name: 'Deloitte', focus: ['Recruiting', 'HRIS', 'Employee Relations', 'Human Resources'] },
        { name: 'Accenture', focus: ['Recruiting', 'Onboarding', 'Training', 'Communication'] }
    ],
    generic: [
        { name: 'Infosys', focus: ['Problem Solving', 'Communication', 'Project Management'] },
        { name: 'TCS', focus: ['Problem Solving', 'Communication', 'Project Management'] },
        { name: 'Wipro', focus: ['Problem Solving', 'Communication', 'Leadership'] },
        { name: 'Accenture', focus: ['Problem Solving', 'Communication', 'Data Analysis'] },
        { name: 'HCLTech', focus: ['Problem Solving', 'Communication', 'Cloud'] },
        { name: 'Tech Mahindra', focus: ['Problem Solving', 'Communication', 'Customer Service'] }
    ]
};

function buildCompanyRecommendations(data, resumeText = '') {
    const role = String(data.predicted_role || '').toLowerCase();
    let category = 'generic';
    if (/\b(hr|human resources|recruitment|recruiter)\b/.test(role)) {
        category = 'hr';
    } else if (/\b(data scientist|data science|data analyst|data analytics)\b/.test(role)) {
        category = 'data';
    } else if (/\b(web developer|frontend developer|front end developer|web designer)\b/.test(role)) {
        category = 'web';
    } else if (/\b(python developer|python engineer)\b/.test(role)) {
        category = 'python';
    } else if (/\b(software developer|software engineer|full stack developer|full-stack developer)\b/.test(role)) {
        category = 'software';
    }

    const skills = Array.isArray(data.detected_skills)
        ? data.detected_skills.filter(skill => typeof skill === 'string' && skill.trim())
        : [];
    const normalizedSkills = new Map(skills.map(skill => [skill.toLowerCase(), skill]));
    const screeningComponents = data.screening_breakdown?.components || {};
    const roleScore = Number(screeningComponents.role_match?.score_pct) || 0;
    const experienceScore = Number(screeningComponents.experience_match?.score_pct) || 0;
    const educationScore = Number(screeningComponents.education_match?.score_pct) || 0;
    const resume = String(resumeText || '').toLowerCase();

    return COMPANY_RECOMMENDATIONS[category].map(company => {
        const matchingSkills = company.focus
            .map(skill => normalizedSkills.get(skill.toLowerCase()))
            .filter(Boolean);
        if (matchingSkills.length < 2 && resume) {
            company.focus.forEach(skill => {
                if (
                    matchingSkills.length < 3 &&
                    resume.includes(skill.toLowerCase()) &&
                    !matchingSkills.some(match => match.toLowerCase() === skill.toLowerCase())
                ) {
                    matchingSkills.push(skill);
                }
            });
        }

        const reasons = matchingSkills.slice(0, 3);
        const roleAlreadyRepresented = matchingSkills.some(skill =>
            role.includes(skill.toLowerCase()) || skill.toLowerCase().includes(role)
        );
        if (reasons.length < 2 && !roleAlreadyRepresented) {
            reasons.push(`${data.predicted_role || 'Relevant role'} alignment`);
        }
        if (reasons.length < 3 && experienceScore > 0) {
            reasons.push(experienceScore >= 50 ? 'Relevant experience signal' : 'Experience considered');
        } else if (reasons.length < 3 && educationScore > 0) {
            reasons.push(educationScore >= 50 ? 'Education indicator' : 'Education considered');
        }

        const score = Math.min(
            94,
            70 + Math.round(roleScore * 0.08) + Math.min(matchingSkills.length, 4) * 2 +
                (experienceScore >= 50 ? 3 : 0) + (educationScore >= 50 ? 2 : 0)
        );
        return { ...company, reasons, score };
    });
}

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
    const companiesSection = document.getElementById('companiesSection');
    const companiesList = document.getElementById('companiesList');

    let currentFile = null;

    // Drag and drop handlers
    uploadZone.addEventListener('click', (event) => {
        if (event.target !== fileInput) {
            fileInput.click();
        }
    });
    
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
        if (!/\.(pdf|png|jpe?g)$/i.test(file.name)) {
            currentFile = null;
            fileInput.value = '';
            uploadZone.querySelector('p').innerHTML = 'Upload your resume or drag and drop<br><span>PDF, PNG, JPG (Max 10MB)</span>';
            showError('Please upload a PDF, PNG, or JPG resume.');
            return;
        }

        currentFile = file;
        errorBox.classList.add('hidden');
        const p = uploadZone.querySelector('p');
        p.innerHTML = `Selected: <strong>${file.name}</strong><br><span>Click to change</span>`;
        resumeInput.value = ''; // Clear text if file selected
    }

    resumeInput.addEventListener('input', () => {
        if (resumeInput.value.trim().length > 0 && currentFile) {
            currentFile = null;
            const p = uploadZone.querySelector('p');
            p.innerHTML = `Upload your resume or drag and drop<br><span>PDF, PNG, JPG (Max 10MB)</span>`;
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
            const message = err.message === 'Failed to fetch'
                ? `RoleSignal backend is unavailable. Please make sure the local backend is running (uvicorn app:app --reload) and try again.`
                : err.message || 'Analysis failed.';
            showError(message);
        } finally {
            setLoading(false);
        }
    });

    function renderResults(data) {
        // Primary Prediction
        predictedRole.textContent = data.predicted_role;
        confidence.textContent = `${window.getRoleSignalDemoConfidence(data.predicted_role)}% Demo Confidence`;

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

        renderCompanyRecommendations(data);
        resultsSection.classList.remove('hidden');
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    function renderCompanyRecommendations(data) {
        companiesList.innerHTML = '';
        const recommendations = buildCompanyRecommendations(data, resumeInput.value);

        recommendations.forEach(company => {
            const card = document.createElement('article');
            card.className = 'card company-card';

            const heading = document.createElement('div');
            heading.className = 'company-card-heading';
            const name = document.createElement('h3');
            name.textContent = company.name;
            const match = document.createElement('span');
            match.className = 'company-match';
            match.textContent = `Potential Match: ${company.score}%`;
            heading.append(name, match);
            card.appendChild(heading);

            const why = document.createElement('h4');
            why.className = 'company-why-title';
            why.textContent = 'Why you may be a fit';
            card.appendChild(why);

            const reasons = document.createElement('ul');
            reasons.className = 'company-reasons';
            company.reasons.forEach(reason => {
                const item = document.createElement('li');
                item.textContent = reason;
                reasons.appendChild(item);
            });
            card.appendChild(reasons);

            const cta = document.createElement('button');
            cta.className = 'company-cta';
            cta.type = 'button';
            cta.disabled = true;
            cta.textContent = 'Explore Opportunities';
            cta.setAttribute('aria-label', `Explore opportunities at ${company.name} (demo only)`);
            card.appendChild(cta);
            companiesList.appendChild(card);
        });

        companiesSection.classList.remove('hidden');
    }

    function showError(msg) {
        errorBox.textContent = msg;
        errorBox.classList.remove('hidden');
    }
});