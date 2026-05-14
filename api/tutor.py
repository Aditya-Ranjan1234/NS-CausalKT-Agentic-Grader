import json
import os
import uuid

try:
    from openai import OpenAI
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USERS_PATH = os.path.join(ROOT, "agentic_ui", "backend", "users.json")
PRACTICE_SESSIONS = {}

client = None
if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def response(body, status=200):
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS"
        },
        "body": json.dumps(body)
    }


def request_path(request):
    return getattr(request, "path", "") or getattr(request, "url", "")


def request_json(request):
    try:
        if hasattr(request, "get_json"):
            return request.get_json() or {}
        body = getattr(request, "body", None)
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        return json.loads(body or "{}")
    except Exception:
        return {}


def load_users():
    try:
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def infer_concept(text):
    value = (text or "").lower()
    aliases = [
        ("Componendo and Dividendo", ["componendo", "dividendo"]),
        ("Quadratic Functions", ["quadratic", "parabola", "factor"]),
        ("Linear Equations", ["linear", "equation", "solve for x"]),
        ("Fractions", ["fraction", "numerator", "denominator"]),
        ("Decimals", ["decimal", "place value"]),
        ("Algebra Basics", ["algebra"]),
        ("Sign Management", ["negative", "positive", "sign"]),
        ("Basic Operations", ["addition", "subtraction", "multiply", "division"])
    ]
    for concept, keys in aliases:
        if any(key in value for key in keys):
            return concept
    words = [w.strip(".,?!:;") for w in value.split() if len(w.strip(".,?!:;")) > 3]
    return " ".join(words[:2]).title() if words else "Practice Concept"


def make_question(topic, difficulty, seed, previous):
    if not client:
        return None, "The question generator is not configured."
    try:
        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "Create exactly 1 specific short-answer assessment question based only on the requested topic. "
                    "Return JSON only with concept and question. question must contain prompt, expected_answer, "
                    "acceptable_answers, difficulty, and rubric. Adjust difficulty exactly as requested."
                )},
                {"role": "user", "content": json.dumps({
                    "topic": topic,
                    "difficulty": difficulty,
                    "seed": seed,
                    "previous_results": previous
                })}
            ],
            response_format={"type": "json_object"}
        )
        data = json.loads(result.choices[0].message.content)
        q = data.get("question") or data
        return {
            "prompt": q.get("prompt") or q.get("question"),
            "expected_answer": q.get("expected_answer") or q.get("answer"),
            "acceptable_answers": q.get("acceptable_answers") or [],
            "difficulty": q.get("difficulty") or difficulty,
            "rubric": q.get("rubric") or "Evaluate semantic correctness."
        }, None
    except Exception as e:
        return None, str(e)


def evaluate_answer(topic, question, answer):
    if not client:
        return None, "The evaluator is not configured."
    try:
        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "Evaluate the student's answer semantically. Apply an authenticity penalty for unnaturally "
                    "over-polished or AI-like short answers; cap score at 0.5 if penalty applies. "
                    "Return JSON with correct, score, feedback, misconception, authenticity_penalty."
                )},
                {"role": "user", "content": json.dumps({
                    "concept": topic,
                    "question": question,
                    "student_answer": answer
                })}
            ],
            response_format={"type": "json_object"}
        )
        data = json.loads(result.choices[0].message.content)
        score = max(0, min(1, float(data.get("score", 0))))
        if data.get("authenticity_penalty"):
            score = min(score, 0.5)
        return {
            "correct": bool(data.get("correct")) or score >= 0.75,
            "score": score,
            "feedback": data.get("feedback", ""),
            "misconception": data.get("misconception", ""),
            "authenticity_penalty": bool(data.get("authenticity_penalty"))
        }, None
    except Exception as e:
        return None, str(e)


def next_difficulty(current, score):
    levels = ["easy", "medium", "hard"]
    idx = levels.index(current if current in levels else "medium")
    if score >= 0.75:
        idx = min(2, idx + 1)
    elif score <= 0.25:
        idx = max(0, idx - 1)
    return levels[idx]


def graph_for_user(user):
    weak = set(user.get("weak_areas", []))
    learned = set(user.get("concepts_learned", []))
    concepts = [
        "Fractions", "Decimals", "Basic Operations", "Linear Equations",
        "Sign Management", "Algebra Basics", "Quadratic Functions"
    ]
    for name in list(user.get("mastery", {}).keys()) + user.get("weak_areas", []):
        if name not in concepts:
            concepts.append(name)
    for node in user.get("practice_nodes", []):
        if node.get("concept") and node["concept"] not in concepts:
            concepts.append(node["concept"])
    nodes = [{"id": c, "status": "weak" if c in weak else ("mastered" if c in learned else "pending")} for c in concepts]
    for node in user.get("practice_nodes", []):
        nodes.append({"id": node.get("id", "Practice Node"), "status": node.get("status", "pending")})
    edges = [
        {"source": "Basic Operations", "target": "Linear Equations"},
        {"source": "Fractions", "target": "Linear Equations"},
        {"source": "Decimals", "target": "Linear Equations"},
        {"source": "Linear Equations", "target": "Sign Management"},
        {"source": "Linear Equations", "target": "Algebra Basics"},
        {"source": "Algebra Basics", "target": "Quadratic Functions"}
    ]
    for node in user.get("practice_nodes", []):
        if node.get("concept") and node.get("id"):
            edges.append({"source": node["concept"], "target": node["id"]})
    recommended = user.get("weak_areas", [None])[0] or user.get("learning_goals", ["Linear Equations"])[0]
    return {"nodes": nodes, "edges": edges, "recommended_next": recommended}


def handler(request):
    method = getattr(request, "method", "GET")
    if method == "OPTIONS":
        return response({})

    path = request_path(request)
    users = load_users()

    if path.endswith("/users"):
        return response(list(users.values()))

    if path.endswith("/profile"):
        user_id = getattr(request, "args", {}).get("user_id", "aditya_ranjan") if hasattr(request, "args") else "aditya_ranjan"
        user = users.get(user_id)
        return response(user or {"error": "User not found"}, 200 if user else 404)

    if path.endswith("/graph"):
        user_id = getattr(request, "args", {}).get("user_id", "aditya_ranjan") if hasattr(request, "args") else "aditya_ranjan"
        return response(graph_for_user(users.get(user_id, {})))

    if path.endswith("/practice/generate"):
        data = request_json(request)
        seed = data.get("question", "")
        topic = infer_concept(seed)
        question, error = make_question(topic, "medium", seed, [])
        if error:
            return response({"error": f"Question generation failed: {error}"}, 400)
        practice_id = str(uuid.uuid4())
        PRACTICE_SESSIONS[practice_id] = {"seed": seed, "topic": topic, "difficulty": "medium", "questions": [question], "results": []}
        return response({"practice_id": practice_id, "seed_question": seed, "concept": topic, "source": "ai", "question_index": 0, "total_questions": 5, "question": {"prompt": question["prompt"], "difficulty": question["difficulty"]}})

    if path.endswith("/practice/answer"):
        data = request_json(request)
        session = PRACTICE_SESSIONS.get(data.get("practice_id"))
        if not session:
            return response({"error": "Practice session expired. Generate questions again."}, 400)
        evaluation, error = evaluate_answer(session["topic"], session["questions"][-1], data.get("answer", ""))
        if error:
            return response({"error": f"Answer evaluation failed: {error}"}, 400)
        current = session["questions"][-1].get("difficulty", session["difficulty"])
        session["results"].append({**evaluation, "prompt": session["questions"][-1]["prompt"], "answer": data.get("answer", ""), "difficulty": current})
        if len(session["results"]) < 5:
            difficulty = next_difficulty(current, evaluation["score"])
            question, error = make_question(session["topic"], difficulty, session["seed"], session["results"])
            if error:
                return response({"error": f"Next-question generation failed: {error}"}, 400)
            session["questions"].append(question)
            return response({"complete": False, "evaluation": evaluation, "question_index": len(session["results"]), "total_questions": 5, "next_difficulty": difficulty, "question": {"prompt": question["prompt"], "difficulty": difficulty}})
        raw_score = sum(item["score"] for item in session["results"])
        mastery = raw_score / 5
        node = {"id": f"Practice: {session['topic']}", "concept": session["topic"], "score": round(raw_score, 2), "mastery": mastery, "status": "mastered" if mastery >= 0.8 else ("weak" if mastery < 0.6 else "intermediate")}
        return response({"complete": True, "score": round(raw_score, 2), "checked": session["results"], "node": node, "message": "Practice complete. The learning graph will reflect this result locally."})

    if path.endswith("/chat"):
        return response({"reply": "Tutor chat is available in local mode for this build.", "updated_mastery": False})

    return response({"error": "Unknown tutor endpoint"}, 404)
