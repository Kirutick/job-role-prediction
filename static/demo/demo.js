document.addEventListener('DOMContentLoaded', () => {
    const frame = document.getElementById('screeningApp');
    const processingCard = document.getElementById('processingCard');
    const processingSteps = Array.from(document.querySelectorAll('#processingSteps li'));
    const resultContent = document.getElementById('resultContent');
    const resultError = document.getElementById('resultError');
    const requestStatus = document.getElementById('requestStatus');
    const presentationButton = document.getElementById('presentationButton');
    const presentationExitButton = document.getElementById('presentationExitButton');
    const restartButton = document.getElementById('restartButton');
    const restartPresentationButton = document.getElementById('restartPresentationButton');
    const presentationToolbar = document.getElementById('presentationToolbar');
    const processingStageNames = [
        'Reading Resume',
        'Extracting Text',
        'Preprocessing Resume',
        'Running ML Model',
        'Extracting Skills',
        'Generating Screening Analysis'
    ];
    let processingTimer = null;
    let currentStage = 0;

    function setStatus(label, state) {
        requestStatus.classList.toggle('is-processing', state === 'processing');
        requestStatus.classList.toggle('is-complete', state === 'complete');
        requestStatus.lastChild.textContent = ` ${label}`;
    }

    function setStage(index) {
        currentStage = Math.min(index, processingSteps.length - 1);
        processingSteps.forEach((step, stepIndex) => {
            step.classList.toggle('is-active', stepIndex === currentStage);
            step.classList.toggle('is-done', stepIndex < currentStage);
        });
    }

    function startProcessing() {
        window.clearInterval(processingTimer);
        processingSteps.forEach((step) => step.classList.remove('is-active', 'is-done'));
        processingCard.classList.remove('hidden');
        resultContent.classList.add('hidden');
        resultError.classList.add('hidden');
        setStatus('Analyzing with the live API', 'processing');
        setStage(0);
        processingTimer = window.setInterval(() => {
            if (currentStage < processingSteps.length - 1) {
                setStage(currentStage + 1);
            }
        }, 420);
    }

    function appendSkillList(parent, values, emptyMessage, missing = false) {
        if (!values || values.length === 0) {
            const empty = document.createElement('span');
            empty.className = 'model-confidence-note';
            empty.textContent = emptyMessage;
            parent.appendChild(empty);
            return;
        }
        values.forEach((value) => {
            const chip = document.createElement('span');
            chip.className = missing ? 'skill-chip is-missing' : 'skill-chip';
            chip.textContent = value;
            parent.appendChild(chip);
        });
    }

    function appendSection(parent, title, className = 'result-subsection') {
        const section = document.createElement('section');
        section.className = className;
        const heading = document.createElement('h4');
        heading.className = 'result-subsection-heading';
        heading.textContent = title;
        section.appendChild(heading);
        parent.appendChild(section);
        return section;
    }

    function renderActualResult(data) {
        window.clearInterval(processingTimer);
        processingCard.classList.add('hidden');
        resultError.classList.add('hidden');
        resultContent.replaceChildren();
        resultContent.classList.remove('hidden', 'has-result');

        const result = document.createElement('div');
        result.className = 'prediction-result';

        const roleBlock = document.createElement('div');
        roleBlock.innerHTML = '<div class="result-label">Predicted Role</div>';
        const roleLine = document.createElement('div');
        roleLine.className = 'prediction-line';
        const roleName = document.createElement('div');
        roleName.className = 'prediction-role';
        roleName.textContent = data.predicted_role || 'Prediction unavailable';
        roleLine.appendChild(roleName);

        const matchBadge = document.createElement('span');
        matchBadge.className = 'match-badge';
        matchBadge.textContent = `${window.getRoleSignalDemoConfidence(data.predicted_role)}% Demo Confidence`;
        roleLine.appendChild(matchBadge);
        roleBlock.appendChild(roleLine);
        result.appendChild(roleBlock);

        const otherPredictions = Array.isArray(data.top_predictions)
            ? data.top_predictions.filter((item) => item.role !== data.predicted_role)
            : [];
        if (otherPredictions.length) {
            const predictions = document.createElement('div');
            predictions.className = 'top-predictions';
            otherPredictions.slice(0, 2).forEach((item) => {
                const row = document.createElement('div');
                row.className = 'top-prediction';
                const label = document.createElement('span');
                label.textContent = item.role;
                const value = document.createElement('span');
                value.textContent = Number.isFinite(item.confidence)
                    ? `${Math.round(item.confidence * 100)}%`
                    : '';
                row.append(label, value);
                predictions.appendChild(row);
            });
            result.appendChild(predictions);
        }

        const skills = appendSection(result, 'Detected Skills');
        const skillList = document.createElement('div');
        skillList.className = 'skill-list';
        appendSkillList(skillList, data.detected_skills, 'No skills detected');
        skills.appendChild(skillList);

        if (Array.isArray(data.missing_common_skills) && data.missing_common_skills.length) {
            const missing = appendSection(result, 'Missing Common Skills');
            const missingList = document.createElement('div');
            missingList.className = 'skill-list';
            appendSkillList(missingList, data.missing_common_skills, '', true);
            missing.appendChild(missingList);
        }

        const screening = data.screening_breakdown;
        if (screening && screening.components) {
            const breakdown = appendSection(result, 'Screening Breakdown');
            const list = document.createElement('div');
            list.className = 'screening-list';
            const labels = {
                role_match: 'Role Alignment',
                skills_match: 'Required Skills',
                experience_match: 'Experience Indicator',
                education_match: 'Education Indicator'
            };
            Object.entries(screening.components).forEach(([key, component]) => {
                const item = document.createElement('div');
                item.className = 'screening-item';
                const label = document.createElement('span');
                label.textContent = labels[key] || key;
                const track = document.createElement('span');
                track.className = 'screening-track';
                const fill = document.createElement('span');
                fill.className = 'screening-fill';
                const score = Number(component.score_pct) || 0;
                fill.style.width = `${Math.min(100, Math.max(0, score))}%`;
                track.appendChild(fill);
                const value = document.createElement('span');
                value.className = 'screening-value';
                value.textContent = `${score}%`;
                item.append(label, track, value);
                list.appendChild(item);
            });
            breakdown.appendChild(list);
        }

        resultContent.appendChild(result);
        resultContent.classList.add('has-result');
        setStatus('Live result received', 'complete');
    }

    function completeRequest(detail) {
        window.clearInterval(processingTimer);
        processingCard.classList.add('hidden');
        if (detail.ok && detail.data && detail.data.predicted_role) {
            renderActualResult(detail.data);
            return;
        }
        resultContent.classList.add('hidden');
        resultError.textContent = detail.error || 'The live API could not complete this analysis.';
        resultError.classList.remove('hidden');
        setStatus('Request failed', 'ready');
    }

    function instrumentAppFrame() {
        try {
            const frameWindow = frame.contentWindow;
            const frameDocument = frame.contentDocument;
            if (!frameWindow || !frameDocument || !frameWindow.fetch) return;

            const originalFetch = frameWindow.fetch.bind(frameWindow);
            if (frameWindow.__roleSignalDemoInstrumented) return;
            frameWindow.__roleSignalDemoInstrumented = true;
            frameWindow.fetch = (input, init) => {
                const requestUrl = typeof input === 'string' ? input : input.url;
                const url = new URL(requestUrl, frameWindow.location.href);
                const method = (init?.method || input?.method || 'GET').toUpperCase();
                if (
                    url.origin !== window.location.origin ||
                    method !== 'POST' ||
                    !['/analyze', '/predict-file'].includes(url.pathname)
                ) {
                    return originalFetch(input, init);
                }

                window.dispatchEvent(new CustomEvent('rolesignal-demo-request-start'));
                return originalFetch(input, init).then((response) => {
                    response.clone().json().then((data) => {
                        window.dispatchEvent(new CustomEvent('rolesignal-demo-request-end', {
                            detail: {
                                ok: response.ok,
                                data: response.ok ? data : null,
                                error: response.ok ? null : data.detail
                            }
                        }));
                    }).catch((error) => {
                        window.dispatchEvent(new CustomEvent('rolesignal-demo-request-end', {
                            detail: { ok: false, error: error.message }
                        }));
                    });
                    return response;
                }).catch((error) => {
                    window.dispatchEvent(new CustomEvent('rolesignal-demo-request-end', {
                        detail: {
                            ok: false,
                            error: error.message === 'Failed to fetch'
                                ? 'RoleSignal backend is unavailable. Please make sure the local backend is running (uvicorn app:app --reload) and try again.'
                                : error.message
                        }
                    }));
                    throw error;
                });
            };

            const analyzeButton = frameDocument.getElementById('analyzeBtn');
            if (analyzeButton && !analyzeButton.dataset.demoObserver) {
                analyzeButton.dataset.demoObserver = 'true';
                analyzeButton.addEventListener('click', () => {
                    const hasText = frameDocument.getElementById('resumeInput')?.value.trim();
                    const hasFile = frameDocument.getElementById('fileInput')?.files.length;
                    if (hasText || hasFile) startProcessing();
                }, true);
            }
        } catch (error) {
            setStatus('Live app unavailable', 'ready');
            resultError.textContent = `Could not connect to the local screening app: ${error.message}`;
            resultError.classList.remove('hidden');
        }
    }

    frame.addEventListener('load', instrumentAppFrame);
    window.addEventListener('rolesignal-demo-request-start', startProcessing);
    window.addEventListener('rolesignal-demo-request-end', (event) => {
        completeRequest(event.detail);
    });

    function setPresentationMode(enabled) {
        document.body.classList.toggle('presentation-mode', enabled);
        presentationExitButton.classList.toggle('hidden', !enabled);
        presentationToolbar.classList.toggle('hidden', !enabled);
        presentationButton.setAttribute('aria-pressed', String(enabled));
        if (enabled) {
            document.getElementById('live-demo').scrollIntoView({ behavior: 'smooth' });
        }
    }

    function restartDemo() {
        window.clearInterval(processingTimer);
        processingCard.classList.add('hidden');
        resultError.classList.add('hidden');
        resultContent.classList.remove('hidden', 'has-result');
        resultContent.innerHTML = '<div class="empty-result"><div class="empty-glyph" aria-hidden="true">✳</div><h3>Your result appears here</h3><p>Paste resume text or upload a PDF, PNG, or JPG in the live app. The panel will show the response from the running RoleSignal API.</p><div class="empty-flow"><span>INPUT</span><b>→</b><span>MODEL</span><b>→</b><span>RESULT</span></div></div>';
        processingSteps.forEach((step) => step.classList.remove('is-active', 'is-done'));
        setStatus('Ready for resume', 'ready');
        frame.contentWindow.location.reload();
        document.getElementById('live-demo').scrollIntoView({ behavior: 'smooth' });
    }

    presentationButton.addEventListener('click', () => setPresentationMode(true));
    presentationExitButton.addEventListener('click', () => setPresentationMode(false));
    restartButton.addEventListener('click', restartDemo);
    restartPresentationButton.addEventListener('click', restartDemo);
    instrumentAppFrame();
});
