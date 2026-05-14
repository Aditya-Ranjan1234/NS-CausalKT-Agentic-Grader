let activePractice = null;
let currentIndex = 0;
let currentQuestion = null;
let results = [];

document.addEventListener('DOMContentLoaded', async () => {
    document.getElementById('seedQuestion').value = '';
    document.getElementById('qaTopic').textContent = 'Enter a question or topic to begin.';

    await loadUsers();

    document.getElementById('generateBtn').addEventListener('click', generatePractice);
    document.getElementById('prevBtn').addEventListener('click', previousQuestion);
    document.getElementById('nextBtn').addEventListener('click', nextQuestion);
    document.getElementById('currentAnswer').addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            nextQuestion();
        }
    });
    document.getElementById('backTutorBtn').addEventListener('click', () => {
        window.location.href = 'tutor.html';
    });

    localStorage.removeItem('qa_seed_question');
    localStorage.removeItem('qa_topic');
});

async function loadUsers() {
    const response = await fetch('/api/tutor/users');
    const users = response.ok ? await response.json() : [];
    const selectedUserId = localStorage.getItem('qa_user_id') || 'aditya_ranjan';
    const select = document.getElementById('userSelect');

    select.innerHTML = '';
    users.forEach(user => {
        const option = document.createElement('option');
        option.value = user.id;
        option.textContent = user.name;
        option.selected = user.id === selectedUserId;
        select.appendChild(option);
    });

    select.addEventListener('change', () => {
        localStorage.setItem('qa_user_id', select.value);
    });
}

async function generatePractice() {
    const seedQuestion = document.getElementById('seedQuestion').value.trim();
    if (!seedQuestion) {
        showResult('Enter a question or topic first.');
        return;
    }

    const userId = document.getElementById('userSelect').value || 'aditya_ranjan';
    localStorage.setItem('qa_user_id', userId);

    showResult('Generating practice questions...');

    const response = await fetch('/api/tutor/practice/generate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question: seedQuestion, user_id: userId})
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        showResult(error.error || 'Could not generate practice questions. Please try again.');
        return;
    }

    activePractice = await response.json();
    currentIndex = 0;
    results = [];
    currentQuestion = activePractice.question;
    renderCurrentQuestion();
    document.getElementById('qaResult').style.display = 'none';
}

function renderCurrentQuestion() {
    if (!activePractice) return;

    const question = currentQuestion;
    document.getElementById('questionCard').style.display = 'block';
    document.getElementById('qaTopic').textContent = `Topic: ${activePractice.concept}`;
    const source = activePractice.source === 'ai' ? 'real-time generator' : activePractice.source;
    document.getElementById('qaConceptHint').textContent = `Generated from: ${activePractice.seed_question} (${source})`;
    document.getElementById('qaProgress').textContent = `Question ${currentIndex + 1} of ${activePractice.total_questions} - ${question.difficulty || 'medium'}`;
    document.getElementById('questionPrompt').textContent = question.prompt;
    document.getElementById('currentAnswer').value = '';
    document.getElementById('prevBtn').disabled = true;
    document.getElementById('nextBtn').textContent = currentIndex === activePractice.total_questions - 1 ? 'Submit Answer' : 'Submit & Next';
    document.getElementById('currentAnswer').focus();
}

function previousQuestion(event) {
    if (event) event.preventDefault();
}

function nextQuestion(event) {
    if (event) event.preventDefault();
    submitCurrentAnswer();
}

async function submitCurrentAnswer() {
    if (!activePractice) return;
    const answer = document.getElementById('currentAnswer').value.trim();
    if (!answer) {
        showResult('Type an answer before continuing.');
        return;
    }

    const userId = document.getElementById('userSelect').value || 'aditya_ranjan';
    document.getElementById('nextBtn').disabled = true;
    showResult('Evaluating answer...');

    const response = await fetch('/api/tutor/practice/answer', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            user_id: userId,
            practice_id: activePractice.practice_id,
            answer
        })
    });

    document.getElementById('nextBtn').disabled = false;

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        showResult(error.error || 'Could not evaluate the answer. Please try again.');
        return;
    }

    const result = await response.json();

    if (!result.complete) {
        results.push(result.evaluation);
        currentIndex = result.question_index;
        currentQuestion = result.question;
        const outcome = result.evaluation.correct ? 'Correct' : 'Needs review';
        showResult(`${outcome}. ${result.evaluation.feedback || ''}<br>Next difficulty: <strong>${result.next_difficulty}</strong>`);
        renderCurrentQuestion();
        return;
    }

    const feedback = (result.checked || []).map((item, index) => {
        const marker = item.correct ? 'Correct' : 'Needs review';
        const penalty = item.authenticity_penalty ? ' - authenticity penalty applied' : '';
        const feedbackText = item.feedback ? ` - ${item.feedback}` : '';
        return `<li>${index + 1}. ${marker}${penalty}${feedbackText}</li>`;
    }).join('');

    const scoreText = Number.isInteger(result.score) ? result.score : result.score.toFixed(1);
    showResult(`
        <strong>Score: ${scoreText}/5</strong><br>
        ${result.message}<br>
        Added graph node: <strong>${result.node.id}</strong>
        <ul class="qa-feedback-list">${feedback}</ul>
    `);
}

function showResult(html) {
    const result = document.getElementById('qaResult');
    result.innerHTML = html;
    result.style.display = 'block';
}
