import os
import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import PyPDF2
from PIL import Image
import base64
from openai import OpenAI
import json
import uuid
import difflib
from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
agentic_ui_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(agentic_ui_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

dotenv_path = os.path.join(project_root, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path, override=True)
    print("Loaded .env file from:", dotenv_path)
else:
    print(".env file not found at:", dotenv_path)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

client = None
PRACTICE_SESSIONS = {}
LAST_PRACTICE_ERROR = None
KT_RUNTIME = None


def load_openai_client():
    global client
    # Manually parse .env to bypass existing environment variables
    api_key = os.getenv('OPENAI_API_KEY')
    
    dotenv_path = os.path.join(project_root, '.env')
    if os.path.exists(dotenv_path):
        with open(dotenv_path, 'r') as f:
            for line in f:
                if line.startswith('OPENAI_API_KEY='):
                    api_key = line.split('=', 1)[1].strip()
                    break
    
    if api_key:
        api_key = api_key.strip()
        client = OpenAI(api_key=api_key)
        print(f"AI client initialized. Key starts with {api_key[:10]} and ends with {api_key[-4:]}")
    else:
        print("OPENAI_API_KEY not found!")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def normalize_topic_name(value):
    return str(value or "").lower().replace("_", " ").replace("-", " ").strip()


def get_text_for_math_detection(result_json):
    return json.dumps(result_json, ensure_ascii=False).lower()


def is_math_kt_candidate(result_json):
    math_keywords = [
        'algebra', 'equation', 'math', 'geometry', 'fraction', 'function',
        'calculation', 'statistics', 'graph', 'line', 'slope', 'percent',
        'probability', 'integer', 'decimal', 'triangle', 'angle', 'theorem'
    ]
    detected_text = get_text_for_math_detection(result_json)
    return any(kw in detected_text for kw in math_keywords)


def load_kt_runtime():
    global KT_RUNTIME
    if KT_RUNTIME is not None:
        return KT_RUNTIME

    import torch
    from dataset import preprocess_assist09
    from models.nscausalkt import NSCausalKT

    csv_path = os.path.join(project_root, 'data', 'skill_builder_data.csv')
    checkpoint_path = os.path.join(project_root, 'checkpoints', 'latest.pt')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing ASSISTments data at {csv_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing trained NS-CausalKT checkpoint at {checkpoint_path}")

    df, num_q, num_c, skill_map = preprocess_assist09(csv_path)
    skill_to_question = df.groupby('concept_id')['question_id'].first().to_dict()
    inv_skill_map = {v: k for k, v in skill_map.items()}

    edges = [[i, i + 1] for i in range(1, num_c - 1)]
    device_name = os.getenv('KT_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_name)
    edge_index = torch.tensor(edges, dtype=torch.long).t().to(device)

    model = NSCausalKT(num_q, num_c, edge_index, d_model=256, n_layers=2).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    KT_RUNTIME = {
        "torch": torch,
        "model": model,
        "device": device,
        "num_q": num_q,
        "num_c": num_c,
        "skill_map": skill_map,
        "inv_skill_map": inv_skill_map,
        "skill_to_question": skill_to_question,
        "checkpoint_path": checkpoint_path
    }
    return KT_RUNTIME


def match_kt_skill(concept, skill_map):
    requested = normalize_topic_name(concept)
    if not requested:
        return None

    scored = []
    for skill_name, concept_id in skill_map.items():
        candidate = normalize_topic_name(skill_name)
        if requested == candidate:
            score = 1.0
        elif requested in candidate or candidate in requested:
            score = 0.92
        else:
            score = difflib.SequenceMatcher(None, requested, candidate).ratio()
        scored.append((score, skill_name, concept_id))

    best = max(scored, key=lambda item: item[0])
    if best[0] < 0.35:
        return None
    return {"requested": concept, "matched_skill": best[1], "concept_id": int(best[2]), "match_score": round(best[0], 3)}


def correctness_to_model_answer(value):
    if isinstance(value, bool):
        score = 1.0 if value else 0.0
    else:
        try:
            score = float(value)
        except (TypeError, ValueError):
            text = str(value or "").lower()
            score = 1.0 if text in {"correct", "true", "yes"} else (0.5 if "partial" in text else 0.0)
    return 2 if score >= 0.5 else 1


def get_kt_parameters_from_analysis(result_json):
    kt_params = result_json.get("kt_parameters") or {}
    interactions = kt_params.get("interactions") or []

    if not interactions:
        for item in result_json.get("corrections", []):
            interactions.append({
                "question_number": item.get("question"),
                "concept": item.get("title") or item.get("description"),
                "correctness": 1
            })
        for item in result_json.get("mistakes", []):
            interactions.append({
                "question_number": item.get("question"),
                "concept": item.get("title") or item.get("description"),
                "correctness": 0
            })

    target_concept = kt_params.get("target_concept")
    if not target_concept:
        focus_areas = result_json.get("focus_areas") or result_json.get("weaknesses") or []
        if focus_areas:
            target_concept = focus_areas[0].get("title") or focus_areas[0].get("description")
    if not target_concept and interactions:
        target_concept = interactions[-1].get("concept")

    return {
        "student_id": kt_params.get("student_id", "uploaded_answer_sheet"),
        "target_concept": target_concept,
        "interactions": interactions[:200],
        "source": "gpt_extracted_from_answer_sheet"
    }


def run_nscausalkt_from_parameters(kt_parameters):
    runtime = load_kt_runtime()
    torch = runtime["torch"]
    model = runtime["model"]
    device = runtime["device"]
    skill_map = runtime["skill_map"]
    inv_skill_map = runtime["inv_skill_map"]
    skill_to_question = runtime["skill_to_question"]
    num_c = runtime["num_c"]

    mapped_interactions = []
    for interaction in kt_parameters.get("interactions", []):
        match = match_kt_skill(interaction.get("concept"), skill_map)
        if not match:
            continue
        concept_id = match["concept_id"]
        mapped_interactions.append({
            **interaction,
            **match,
            "question_id": int(skill_to_question.get(concept_id, 1)),
            "model_answer": correctness_to_model_answer(interaction.get("correctness"))
        })

    if not mapped_interactions:
        return {
            "active": False,
            "error": "GPT did not provide any math concepts that could be mapped to the trained ASSISTments skill space.",
            "kt_parameters": kt_parameters
        }

    target_match = match_kt_skill(kt_parameters.get("target_concept"), skill_map)
    if not target_match:
        target_match = {
            "requested": kt_parameters.get("target_concept"),
            "matched_skill": mapped_interactions[-1]["matched_skill"],
            "concept_id": mapped_interactions[-1]["concept_id"],
            "match_score": mapped_interactions[-1]["match_score"]
        }
    target_c_id = int(target_match["concept_id"])

    seq_len = min(len(mapped_interactions), 200)
    q_values = [item["question_id"] for item in mapped_interactions[-seq_len:]]
    c_values = [item["concept_id"] for item in mapped_interactions[-seq_len:]]
    a_values = [item["model_answer"] for item in mapped_interactions[-seq_len:]]

    q = torch.tensor([q_values], dtype=torch.long, device=device)
    c = torch.tensor([c_values], dtype=torch.long, device=device)
    a = torch.tensor([a_values], dtype=torch.long, device=device)
    delta_t = torch.zeros((1, seq_len, 1), dtype=torch.float, device=device)
    target_c = torch.full((1, seq_len, 1), target_c_id, dtype=torch.long, device=device)

    with torch.no_grad():
        y_hat, y_base, mu_tilde, h_t = model(q, a, c, delta_t, target_c=target_c)
        final_idx = seq_len - 1
        predicted_probability = float(y_hat[0, final_idx, 0].detach().cpu().item())
        base_probability = float(y_base[0, final_idx, 0].detach().cpu().item())
        final_mastery = mu_tilde[0, final_idx, :].detach().cpu()

        encountered = sorted(set(c_values + [target_c_id]))
        mastery_by_concept = []
        for concept_id in encountered:
            if 0 <= concept_id < num_c:
                mastery_by_concept.append({
                    "concept_id": int(concept_id),
                    "skill": str(inv_skill_map.get(concept_id, "Unknown")),
                    "mastery": round(float(final_mastery[concept_id].item()), 4)
                })

        weakest = sorted(mastery_by_concept, key=lambda item: item["mastery"])[:5]
        strongest = sorted(mastery_by_concept, key=lambda item: item["mastery"], reverse=True)[:5]

        intervention = None
        if weakest:
            weakest_id = weakest[0]["concept_id"]
            mu_do_1 = mu_tilde[:, final_idx:final_idx + 1, :].clone()
            mu_do_1[..., weakest_id] = 1.0
            mu_do_1 = model.csl.propagate_concept_graph(mu_do_1, model.edge_index, model.scl.W_sym, model.adj_matrix)
            tc = torch.full((1, 1, 1), target_c_id, dtype=torch.long, device=device)
            concat_intervened = torch.cat([h_t[:, final_idx:final_idx + 1, :], torch.gather(mu_do_1, 2, tc)], dim=-1)
            dummy_ace = torch.zeros((1, 1, 1), device=device)
            y_intervened = torch.sigmoid(model.W_p(torch.cat([concat_intervened, dummy_ace], dim=-1)))
            intervention = {
                "intervened_concept": weakest[0]["skill"],
                "do_value": 1.0,
                "target_skill": target_match["matched_skill"],
                "original_probability": round(predicted_probability, 4),
                "intervened_probability": round(float(y_intervened[0, 0, 0].detach().cpu().item()), 4)
            }

    return {
        "active": True,
        "model": "NS-CausalKT",
        "checkpoint": os.path.relpath(runtime["checkpoint_path"], project_root),
        "input_parameters": {
            "q": q_values,
            "a": a_values,
            "c": c_values,
            "delta_t": [0.0] * seq_len,
            "target_c": target_c_id,
            "seq_len": seq_len
        },
        "gpt_kt_parameters": kt_parameters,
        "concept_mapping": mapped_interactions,
        "target_mapping": target_match,
        "prediction": {
            "target_skill": target_match["matched_skill"],
            "passing_probability": round(predicted_probability, 4),
            "base_probability": round(base_probability, 4)
        },
        "weakest_concepts": weakest,
        "strongest_concepts": strongest,
        "counterfactual": intervention
    }


def analyze_with_gpt4o_mini(file_paths, model_insights=None):
    if not client:
        print("No AI client, returning sample data")
        return {
            "overall_score": "75",
            "correct": 5,
            "incorrect": 2,
            "partial": 1,
            "summary": "This is a sample summary. Configure the API key to use real analysis.",
            "mistakes": [
                {"title": "Algebraic Error", "question": 2, "description": "Sign error in quadratic equation solution.", "correction": "Should be -b ± sqrt(b²-4ac) over 2a"}
            ],
            "corrections": [
                {"title": "Correct Application", "question": 1, "description": "Properly used the Pythagorean theorem."}
            ],
            "strengths": [
                {"title": "Geometry Understanding", "question": 3, "description": "Strong grasp of triangle properties."}
            ],
            "weaknesses": [
                {"title": "Sign Management", "question": 2, "description": "Needs practice with negative numbers."}
            ],
            "focus_areas": [
                {"title": "Quadratic Formula", "description": "Review the quadratic formula and sign conventions."}
            ],
            "kt_parameters": {
                "student_id": "sample",
                "target_concept": "Quadratic Functions",
                "interactions": [
                    {"question_number": 1, "concept": "Pythagorean Theorem", "correctness": 1.0},
                    {"question_number": 2, "concept": "Algebraic Solving", "correctness": 0.0},
                    {"question_number": 3, "concept": "Linear Equations", "correctness": 1.0}
                ]
            },
            "bounding_boxes": [
                {"page": 1, "x": 100, "y": 150, "width": 200, "height": 80, "type": "error", "description": "Quadratic formula mistake"}
            ]
        }
    
    messages = [
        {"role": "system", "content": "You are an expert educational assistant that grades answer sheets and returns JSON only."}
    ]
    
    if len(file_paths) > 0:
        user_content = [
            {
                "type": "text",
                "text": f"""You are an expert teacher grading a student's answer sheet. Analyze the following images and provide detailed feedback.

Additional model insights (from NS-CausalKT):
{model_insights or "None"}

Please return a JSON response with the following structure:
{{
    "overall_score": "string (percentage)",
    "correct": number,
    "incorrect": number,
    "partial": number,
    "summary": "detailed summary of performance",
    "mistakes": [
        {{"title": "short title", "question": number, "description": "detailed description", "correction": "how to fix it"}}
    ],
    "corrections": [
        {{"title": "short title", "question": number, "description": "what was done correctly"}}
    ],
    "strengths": [
        {{"title": "short title", "question": number, "description": "strength details"}}
    ],
    "weaknesses": [
        {{"title": "short title", "question": number, "description": "weakness details"}}
    ],
    "focus_areas": [
        {{"title": "area title", "description": "what to focus on"}}
    ],
    "kt_parameters": {{
        "student_id": "uploaded_answer_sheet",
        "target_concept": "math concept to predict next, or null for non-math",
        "interactions": [
            {{"question_number": 1, "concept": "closest specific math concept tested", "correctness": 1.0}}
        ]
    }},
    "bounding_boxes": [
        {{"page": 1, "x": 100, "y": 150, "width": 200, "height": 80, "type": "error/warning/success", "description": "brief description"}}
    ]
}}

For kt_parameters, include only academic interactions visible in the answer sheet. Use correctness 1.0 for correct, 0.5 for partial, and 0.0 for incorrect. If the work is not math, return an empty interactions list and null target_concept."""
            }
        ]
        
        for file_path in file_paths:
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                base64_image = encode_image(file_path)
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
        
        messages.append({"role": "user", "content": user_content})
    else:
        prompt = f"""You are an expert teacher grading a student's answer sheet. Analyze the following content and provide detailed feedback.

Additional model insights (from NS-CausalKT):
{model_insights or "None"}

Please return a JSON response with the following structure:
{{
    "overall_score": "string (percentage)",
    "correct": number,
    "incorrect": number,
    "partial": number,
    "summary": "detailed summary of performance",
    "mistakes": [
        {{"title": "short title", "question": number, "description": "detailed description", "correction": "how to fix it"}}
    ],
    "corrections": [
        {{"title": "short title", "question": number, "description": "what was done correctly"}}
    ],
    "strengths": [
        {{"title": "short title", "question": number, "description": "strength details"}}
    ],
    "weaknesses": [
        {{"title": "short title", "question": number, "description": "weakness details"}}
    ],
    "focus_areas": [
        {{"title": "area title", "description": "what to focus on"}}
    ],
    "kt_parameters": {{
        "student_id": "uploaded_answer_sheet",
        "target_concept": "math concept to predict next, or null for non-math",
        "interactions": [
            {{"question_number": 1, "concept": "closest specific math concept tested", "correctness": 1.0}}
        ]
    }},
    "bounding_boxes": [
        {{"page": 1, "x": 100, "y": 150, "width": 200, "height": 80, "type": "error/warning/success", "description": "brief description"}}
    ]
}}

For kt_parameters, include only academic interactions visible in the answer sheet. Use correctness 1.0 for correct, 0.5 for partial, and 0.0 for incorrect. If the work is not math, return an empty interactions list and null target_concept."""
        messages.append({"role": "user", "content": prompt})
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        result_json = json.loads(response.choices[0].message.content)
        result_json['kt_active'] = is_math_kt_candidate(result_json)
        return result_json
    except Exception as e:
        print(f"CRITICAL: AI API error: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to sample data due to above error")
        return {
            "overall_score": "75",
            "correct": 5,
            "incorrect": 2,
            "partial": 1,
            "summary": "This is a sample summary. Check your API key for real analysis.",
            "mistakes": [
                {"title": "Algebraic Error", "question": 2, "description": "Sign error in quadratic equation solution.", "correction": "Should be -b ± sqrt(b²-4ac) over 2a"}
            ],
            "corrections": [
                {"title": "Correct Application", "question": 1, "description": "Properly used the Pythagorean theorem."}
            ],
            "strengths": [
                {"title": "Geometry Understanding", "question": 3, "description": "Strong grasp of triangle properties."}
            ],
            "weaknesses": [
                {"title": "Sign Management", "question": 2, "description": "Needs practice with negative numbers."}
            ],
            "focus_areas": [
                {"title": "Quadratic Formula", "description": "Review the quadratic formula and sign conventions."}
            ],
            "kt_parameters": {
                "student_id": "sample",
                "target_concept": "Quadratic Functions",
                "interactions": [
                    {"question_number": 1, "concept": "Pythagorean Theorem", "correctness": 1.0},
                    {"question_number": 2, "concept": "Algebraic Solving", "correctness": 0.0},
                    {"question_number": 3, "concept": "Linear Equations", "correctness": 1.0}
                ]
            },
            "bounding_boxes": [
                {"page": 1, "x": 100, "y": 150, "width": 200, "height": 80, "type": "error", "description": "Quadratic formula mistake"}
            ]
        }


@app.route('/')
def index():
    return send_from_directory(agentic_ui_dir, 'index.html')


@app.route('/favicon.ico')
def favicon():
    return '', 204


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(agentic_ui_dir, path)


@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    return jsonify({
        'auc_score': 0.8137,
        'rmse': 0.4001,
        'accuracy': 0.7668,
        'latent_orthogonality': 0.89,
        'forgetting_rate': 0.045,
        'epochs_trained': 30,
        'training_curves': {'epochs': [1,2,3,4,5,6,7,8,9,10], 'auc': [0.5,0.6,0.7,0.75,0.78,0.79,0.8,0.805,0.81,0.8137], 'rmse': [0.5,0.48,0.45,0.43,0.42,0.41,0.405,0.402,0.401,0.4001], 'acc': [0.5,0.6,0.7,0.73,0.75,0.755,0.76,0.762,0.765,0.7668]},
        'sample_students': [
            {'student_id': 1, 'predictions': [0.8, 0.7, 0.9, 0.6, 0.85], 'actuals': [1,0,1,0,1], 'sequence_length':5},
            {'student_id': 2, 'predictions': [0.6, 0.9, 0.75, 0.55, 0.7], 'actuals': [0,1,1,0,0], 'sequence_length':5},
            {'student_id':3, 'predictions': [0.9,0.85,0.95,0.8,0.88], 'actuals': [1,1,1,1,1], 'sequence_length':5}
        ],
        'concepts': [
            {'name': 'Linear Equations', 'proficiency': 75},
            {'name': 'Quadratic Functions', 'proficiency':82},
            {'name': 'Graph Theory Basics', 'proficiency':68},
            {'name': 'Probability Distributions', 'proficiency':79}
        ]
    })


@app.route('/api/graphs/<domain>', methods=['GET'])
def get_graph_data(domain):
    safe_domain = domain.upper()
    if safe_domain not in {"DSA", "ML"}:
        return jsonify({"error": "Unknown graph domain"}), 404

    graph_path = os.path.join(project_root, "Graph", safe_domain, "knowledge_graph.json")
    if not os.path.exists(graph_path):
        return jsonify({"error": f"Graph data not found for {safe_domain}"}), 404

    try:
        with open(graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/causal', methods=['GET'])
def get_causal_data():
    return jsonify({
        'ace_ranking': [
            {'name': 'Concept_Retention', 'value': 0.942},
            {'name': 'Time_On_Task', 'value': 0.780},
            {'name': 'Inter_Skill_Dependency', 'value':0.554},
            {'name': 'Anxiety_Factor', 'value':-0.210}
        ],
        'topic_graph': {
            'nodes': [
                {'id': 'Number Line', 'proficiency': 0.92},
                {'id': 'Decimals', 'proficiency': 0.88},
                {'id': 'Fractions', 'proficiency': 0.85},
                {'id': 'Percents', 'proficiency': 0.81},
                {'id': 'Linear Equations', 'proficiency': 0.64},
                {'id': 'Algebra Basics', 'proficiency': 0.58},
                {'id': 'Geometry Basics', 'proficiency': 0.72},
                {'id': 'Quadratic Functions', 'proficiency': 0.42},
                {'id': 'Statistics', 'proficiency': 0.78}
            ],
            'edges': [
                {'from': 'Number Line', 'to': 'Fractions', 'strength': 0.88},
                {'from': 'Number Line', 'to': 'Decimals', 'strength': 0.92},
                {'from': 'Decimals', 'to': 'Percents', 'strength': 0.95},
                {'from': 'Fractions', 'to': 'Percents', 'strength': 0.91},
                {'from': 'Fractions', 'to': 'Linear Equations', 'strength': 0.82},
                {'from': 'Linear Equations', 'to': 'Algebra Basics', 'strength': 0.91},
                {'from': 'Geometry Basics', 'to': 'Algebra Basics', 'strength': 0.55},
                {'from': 'Algebra Basics', 'to': 'Quadratic Functions', 'strength': 0.75},
                {'from': 'Statistics', 'to': 'Decimals', 'strength': 0.45}
            ]
        },
        'benchmark_data': [
            {'model': 'NS-CausalKT-v2', 'auc_baseline': 0.824, 'auc_causal':0.891, 'rmse_delta':-0.062, 'efficiency_gain':6.1},
            {'model': 'AKT (Baseline)', 'auc_baseline':0.791, 'auc_causal':0.791, 'rmse_delta':0.000, 'efficiency_gain':0.0},
            {'model': 'DKT (Baseline)', 'auc_baseline':0.805, 'auc_causal':0.812, 'rmse_delta':0.002, 'efficiency_gain':1.2}
        ]
    })


@app.route('/api/analyze', methods=['POST'])
def analyze():
    print("Analyze endpoint called!")
    try:
        files = request.files.getlist('files')
        print(f"Number of files: {len(files)}")
        if not files:
            return jsonify({"error": "No files uploaded"}), 400
        
        image_paths = []
        
        for file in files:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            
            if not file.filename.lower().endswith('.pdf'):
                image_paths.append(file_path)
        
        # GPT grades the sheet and extracts a structured KT payload.
        results = analyze_with_gpt4o_mini(image_paths)

        if results.get("kt_active"):
            kt_parameters = get_kt_parameters_from_analysis(results)
            try:
                results["ns_causalkt"] = run_nscausalkt_from_parameters(kt_parameters)
                results["kt_active"] = bool(results["ns_causalkt"].get("active"))
            except Exception as kt_error:
                print(f"NS-CausalKT inference error: {kt_error}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                results["ns_causalkt"] = {
                    "active": False,
                    "error": str(kt_error),
                    "gpt_kt_parameters": kt_parameters
                }
                results["kt_active"] = False
        else:
            results["ns_causalkt"] = {
                "active": False,
                "reason": "No math topic detected by GPT output.",
                "gpt_kt_parameters": get_kt_parameters_from_analysis(results)
            }
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500
# --- AI Tutor Endpoints ---
def get_user_profile(user_id="aditya_ranjan"):
    users_file = os.path.join(script_dir, 'users.json')
    try:
        with open(users_file, 'r') as f:
            users = json.load(f)
            return users.get(user_id)
    except Exception:
        return None

def save_user_profile(user_id, data):
    users_file = os.path.join(script_dir, 'users.json')
    try:
        with open(users_file, 'r') as f:
            users = json.load(f)
        users[user_id] = data
        with open(users_file, 'w') as f:
            json.dump(users, f, indent=4)
        return True
    except Exception:
        return False

def infer_practice_concept(seed_question):
    text = (seed_question or "").lower()
    phrase_aliases = [
        ("Chinese Language", ["learning chinese", "learn chinese", "chinese language", "mandarin", "chinese"]),
        ("Componendo and Dividendo", ["componendo and dividendo", "componendo", "dividendo"]),
        ("Quadratic Functions", ["quadratic functions", "quadratic equation", "quadratics"]),
        ("Linear Equations", ["linear equations", "linear equation", "solve for x"]),
        ("Algebra Basics", ["algebra basics", "basic algebra", "algebra"]),
        ("Sign Management", ["sign management", "negative numbers", "positive and negative", "sign errors"]),
        ("Basic Operations", ["basic operations", "arithmetic operations"])
    ]
    for concept, aliases in phrase_aliases:
        if any(alias in text for alias in aliases):
            return concept
    concept_keywords = get_concept_keywords()
    for concept, keywords in concept_keywords:
        if any(keyword in text for keyword in keywords):
            return concept
    stop_words = {
        "what", "when", "where", "which", "with", "about", "explain", "solve",
        "practice", "question", "questions", "generate", "please", "learn",
        "study", "help", "this", "that", "into", "from", "does", "mean"
    }
    words = [
        word.strip(".,?!:;")
        for word in text.split()
        if len(word.strip(".,?!:;")) > 3 and word.strip(".,?!:;") not in stop_words
    ]
    return " ".join(words[:2]).title() if words else "Practice Concept"

def get_concept_keywords():
    return [
        ("Fractions", ["fraction", "fractions", "numerator", "denominator", "simplify fraction", "mixed number"]),
        ("Decimals", ["decimal", "decimals", "place value", "tenths", "hundredths"]),
        ("Quadratic Functions", ["quadratic", "parabola", "factor", "roots", "vertex", "x^2"]),
        ("Linear Equations", ["linear", "equation", "equations", "solve for x", "variable", "coefficient"]),
        ("Basic Operations", ["addition", "subtraction", "multiply", "multiplication", "division", "operation"]),
        ("Componendo and Dividendo", ["componendo", "dividendo", "ratio", "proportion"]),
        ("Chinese Language", ["chinese", "mandarin", "pinyin", "hanzi", "tones"])
    ]

def get_supported_practice_topics():
    return [
        "Linear Equations",
        "Fractions",
        "Decimals",
        "Quadratic Functions",
        "Basic Operations",
        "Componendo and Dividendo",
        "Algebra Basics",
        "Sign Management",
        "Chinese Language"
    ]

def get_topic_terms(seed_question, concept):
    text = (seed_question or "").lower()
    terms = set()
    for known_concept, keywords in get_concept_keywords():
        if known_concept == concept:
            terms.update(keyword.lower() for keyword in keywords)
    for word in text.replace("?", " ").replace(",", " ").split():
        clean = word.strip(".,?!:;()[]{}")
        if len(clean) > 4:
            terms.add(clean)
    for part in concept.lower().split():
        if len(part) > 3:
            terms.add(part)
    return terms

def questions_match_topic(seed_question, concept, questions):
    terms = get_topic_terms(seed_question, concept)
    if not terms:
        return True
    matched = 0
    for question in questions:
        text = f"{question.get('prompt', '')} {question.get('answer', '')}".lower()
        if any(term in text for term in terms):
            matched += 1
    return matched >= 3

def build_ai_practice_question(seed_question, concept, difficulty, previous_results=None, user=None):
    global LAST_PRACTICE_ERROR
    LAST_PRACTICE_ERROR = None
    if not client:
        LAST_PRACTICE_ERROR = "AI client is not initialized."
        return None

    try:
        profile_context = json.dumps({
            "mastery": (user or {}).get("mastery", {}),
            "weak_areas": (user or {}).get("weak_areas", []),
            "learning_goals": (user or {}).get("learning_goals", [])
        })
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate assessment questions in real time. "
                        "Create exactly 1 specific, non-generic short-answer question based only on the user's requested topic. "
                        "Do not ask meta questions like naming the topic or writing an important term. "
                        "The question must test actual knowledge or skill in the topic. "
                        "For math, ask solvable computation/application questions. "
                        "For languages, ask translation, vocabulary, grammar, script, pronunciation, or usage questions. "
                        "For any other subject, ask precise factual or applied questions appropriate to that subject. "
                        "Adjust difficulty exactly to the requested difficulty. "
                        "Return JSON only with keys concept and question. "
                        "question must be an object with prompt, expected_answer, acceptable_answers, difficulty, and rubric. "
                        "The expected_answer is for evaluator reference only; the UI must not use exact string matching."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"REQUESTED_TOPIC: {concept}\n"
                        f"DIFFICULTY: {difficulty}\n"
                        f"Student profile: {profile_context}\n"
                        f"Student question/topic text: {seed_question}\n"
                        f"Previous results in this session: {json.dumps(previous_results or [])}"
                    )
                }
            ],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        question = data.get("question") or data.get("item") or data
        if not isinstance(question, dict):
            LAST_PRACTICE_ERROR = "AI response did not contain a question object."
            return None

        prompt = str(question.get("prompt") or question.get("question") or "").strip()
        expected_answer = question.get("expected_answer") or question.get("answer") or question.get("correct_answer") or ""
        expected_answer = ", ".join(str(value) for value in expected_answer) if isinstance(expected_answer, list) else str(expected_answer).strip()
        acceptable = question.get("acceptable_answers") or [expected_answer]
        if isinstance(acceptable, str):
            acceptable = [acceptable]
        rubric = str(question.get("rubric") or "Evaluate whether the answer is semantically correct for the prompt.").strip()

        if prompt and expected_answer:
            return {
                "prompt": prompt,
                "expected_answer": expected_answer,
                "acceptable_answers": [str(value).strip() for value in acceptable if str(value).strip()],
                "difficulty": question.get("difficulty") or difficulty,
                "rubric": rubric
            }
        LAST_PRACTICE_ERROR = "AI response did not contain a valid prompt and expected_answer."
    except Exception as e:
        LAST_PRACTICE_ERROR = str(e)
        print(f"Practice question AI generation failed: {e}")

    return None

def evaluate_answer_with_openai(concept, question, user_answer):
    global LAST_PRACTICE_ERROR
    LAST_PRACTICE_ERROR = None
    if not client:
        LAST_PRACTICE_ERROR = "AI client is not initialized."
        return None

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You evaluate student answers. Do not require exact string matching. "
                        "Grade semantic correctness using the prompt, expected answer, acceptable alternatives, and rubric. "
                        "Apply an authenticity penalty when an answer appears AI-generated or unnaturally over-polished: "
                        "for example, it is much more verbose than needed, uses generic textbook phrasing, gives a polished paragraph "
                        "for a short-answer prompt, avoids the student's own working, or mirrors the expected answer with excessive explanation. "
                        "Do not penalize concise correct answers. If the authenticity penalty applies, cap score at 0.5 even if the content is correct. "
                        "Return JSON only with keys correct, score, feedback, misconception, and authenticity_penalty. "
                        "correct is boolean. score is 0, 0.5, or 1. authenticity_penalty is boolean."
                    )
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "concept": concept,
                        "prompt": question.get("prompt"),
                        "expected_answer": question.get("expected_answer"),
                        "acceptable_answers": question.get("acceptable_answers", []),
                        "rubric": question.get("rubric"),
                        "student_answer": user_answer
                    })
                }
            ],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        score = float(data.get("score", 1 if data.get("correct") else 0))
        score = max(0, min(1, score))
        authenticity_penalty = bool(data.get("authenticity_penalty"))
        if authenticity_penalty:
            score = min(score, 0.5)
        return {
            "correct": bool(data.get("correct")) or score >= 0.75,
            "score": score,
            "feedback": str(data.get("feedback") or "").strip(),
            "misconception": str(data.get("misconception") or "").strip(),
            "authenticity_penalty": authenticity_penalty
        }
    except Exception as e:
        LAST_PRACTICE_ERROR = str(e)
        print(f"Practice answer evaluation failed: {e}")
        return None

def next_difficulty(current_difficulty, evaluation):
    levels = ["easy", "medium", "hard"]
    current = current_difficulty if current_difficulty in levels else "medium"
    idx = levels.index(current)
    score = evaluation.get("score", 1 if evaluation.get("correct") else 0)
    if score >= 0.75:
        idx = min(len(levels) - 1, idx + 1)
    elif score <= 0.25:
        idx = max(0, idx - 1)
    return levels[idx]

@app.route('/api/tutor/users', methods=['GET'])
def tutor_users():
    users_file = os.path.join(script_dir, 'users.json')
    try:
        with open(users_file, 'r') as f:
            users = json.load(f)
            return jsonify(list(users.values()))
    except Exception as e:
        return jsonify([])

@app.route('/api/tutor/profile', methods=['GET'])
def tutor_profile():
    user_id = request.args.get('user_id', 'aditya_ranjan')
    user = get_user_profile(user_id)
    if user:
        return jsonify(user)
    return jsonify({"error": "User not found"}), 404

@app.route('/api/tutor/goal', methods=['POST'])
def update_goal():
    data = request.json
    goal = data.get('goal')
    user = get_user_profile()
    if user and goal:
        user['learning_goals'] = [goal]
        save_user_profile(user['id'], user)
        return jsonify({"success": True, "user": user})
    return jsonify({"error": "Failed to update goal"}), 400

@app.route('/api/tutor/chat', methods=['POST'])
def tutor_chat():
    data = request.json
    message = data.get('message')
    user = get_user_profile()
    
    if not client:
        return jsonify({
            "reply": "I am operating in offline mode. Please set OPENAI_API_KEY to enable AI chat.",
            "updated_mastery": False
        })
        
    messages = [
        {"role": "system", "content": "You are an AI Tutor. The student's current profile is: " + json.dumps(user) + ". Provide short, encouraging, and pedagogically sound advice. If they ask to learn a concept, explain it briefly."},
        {"role": "user", "content": message}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        reply = response.choices[0].message.content
        
        # Log interaction
        user['interaction_history'].append({"user": message, "bot": reply})
        save_user_profile(user['id'], user)
        
        return jsonify({
            "reply": reply,
            "updated_mastery": False # In a real system, you'd extract intent and update mastery here
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/tutor/practice/generate', methods=['POST'])
def generate_practice():
    data = request.json or {}
    seed_question = data.get('question', '')
    if not seed_question.strip():
        return jsonify({"error": "Enter a question or topic first."}), 400

    selected_topic = infer_practice_concept(seed_question)
    user_id = data.get('user_id', 'aditya_ranjan')
    user = get_user_profile(user_id) or {}

    if not client:
        return jsonify({
            "error": "The question generator is not configured. Set the API key in .env and restart Flask."
        }), 400

    question = build_ai_practice_question(seed_question, selected_topic, "medium", [], user)
    if not question:
        return jsonify({
            "error": f"Question generation failed: {LAST_PRACTICE_ERROR or 'unknown error'}"
        }), 400

    practice_id = str(uuid.uuid4())
    PRACTICE_SESSIONS[practice_id] = {
        "seed_question": seed_question,
        "concept": selected_topic,
        "current_difficulty": "medium",
        "questions": [question],
        "results": []
    }

    return jsonify({
        "practice_id": practice_id,
        "seed_question": seed_question,
        "concept": selected_topic,
        "source": "ai",
        "question_index": 0,
        "total_questions": 5,
        "question": {"prompt": question["prompt"], "difficulty": question.get("difficulty", "medium")}
    })

@app.route('/api/tutor/practice/answer', methods=['POST'])
def answer_practice():
    data = request.json or {}
    practice_id = data.get('practice_id')
    user_answer = data.get('answer', '')
    practice = PRACTICE_SESSIONS.get(practice_id)
    if not practice:
        return jsonify({"error": "Practice session expired. Generate questions again before submitting."}), 400

    concept = practice["concept"]
    question = practice["questions"][-1]
    evaluation = evaluate_answer_with_openai(concept, question, user_answer)
    if not evaluation:
        return jsonify({
            "error": f"Answer evaluation failed: {LAST_PRACTICE_ERROR or 'unknown error'}"
        }), 400

    result = {
        "prompt": question["prompt"],
        "answer": user_answer,
        "expected_answer": question.get("expected_answer"),
        "correct": evaluation["correct"],
        "score": evaluation["score"],
        "feedback": evaluation["feedback"],
        "misconception": evaluation["misconception"],
        "authenticity_penalty": evaluation.get("authenticity_penalty", False),
        "difficulty": question.get("difficulty", practice.get("current_difficulty", "medium"))
    }
    practice["results"].append(result)
    answered_count = len(practice["results"])

    if answered_count < 5:
        next_level = next_difficulty(result["difficulty"], evaluation)
        practice["current_difficulty"] = next_level
        user_id = data.get('user_id', 'aditya_ranjan')
        user = get_user_profile(user_id) or {}
        next_question = build_ai_practice_question(
            practice["seed_question"],
            concept,
            next_level,
            practice["results"],
            user
        )
        if not next_question:
            return jsonify({
                "error": f"Next-question generation failed: {LAST_PRACTICE_ERROR or 'unknown error'}"
            }), 400
        practice["questions"].append(next_question)
        return jsonify({
            "complete": False,
            "evaluation": evaluation,
            "question_index": answered_count,
            "total_questions": 5,
            "next_difficulty": next_level,
            "question": {"prompt": next_question["prompt"], "difficulty": next_question.get("difficulty", next_level)}
        })

    return finish_practice_session(data.get('user_id', 'aditya_ranjan'), practice_id)

def finish_practice_session(user_id, practice_id):
    practice = PRACTICE_SESSIONS.get(practice_id)
    if not practice:
        return jsonify({"error": "Practice session expired."}), 400

    concept = practice["concept"]
    checked = practice["results"]
    user = get_user_profile(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    raw_score = sum(item.get("score", 0) for item in checked)
    score = round(raw_score, 2)
    mastery_value = raw_score / 5
    practice_node = {
        "id": f"Practice: {concept}",
        "concept": concept,
        "score": score,
        "mastery": mastery_value,
        "status": "mastered" if mastery_value >= 0.8 else ("weak" if mastery_value < 0.6 else "intermediate")
    }

    user.setdefault("practice_nodes", [])
    user["practice_nodes"] = [node for node in user["practice_nodes"] if node.get("id") != practice_node["id"]]
    user["practice_nodes"].append(practice_node)
    user.setdefault("mastery", {})[concept] = mastery_value

    if mastery_value >= 0.8:
        if concept not in user.setdefault("concepts_learned", []):
            user["concepts_learned"].append(concept)
        if concept in user.setdefault("weak_areas", []):
            user["weak_areas"].remove(concept)
        message = "Strong work. This concept was marked as mastered in your learning path."
    elif mastery_value < 0.6:
        if concept not in user.setdefault("weak_areas", []):
            user["weak_areas"].append(concept)
        message = "This concept needs more practice, so it was added as a weak node."
    else:
        if concept in user.setdefault("weak_areas", []):
            user["weak_areas"].remove(concept)
        message = "You are making progress. The concept was added as an intermediate node."

    practice_record = {
        "type": "q_answer_practice",
        "concept": concept,
        "score": score,
        "total": 5,
        "mastery": mastery_value,
        "node_id": practice_node["id"],
        "questions": checked
    }
    user.setdefault("practice_history", []).append(practice_record)
    user.setdefault("interaction_history", []).append({
        "user": f"Q Answer practice: {concept}",
        "bot": f"Score {score}/5. {message}"
    })

    save_user_profile(user_id, user)

    return jsonify({
        "complete": True,
        "score": score,
        "checked": checked,
        "node": practice_node,
        "message": message
    })

@app.route('/api/tutor/graph', methods=['GET'])
def tutor_graph():
    user_id = request.args.get('user_id', 'aditya_ranjan')
    user = get_user_profile(user_id) or {}
    weak_areas = user.get('weak_areas', [])
    learned = set(user.get('concepts_learned', []))
    weak = set(weak_areas)
    
    # Dynamic Prerequisite Graph based on NS-CausalKT Logic
    # This is a sample personalized DAG showing what's mastered, pending, and current focus
    concept_ids = [
        "Fractions",
        "Decimals",
        "Basic Operations",
        "Linear Equations",
        "Sign Management",
        "Algebra Basics",
        "Quadratic Functions"
    ]
    for topic in get_supported_practice_topics():
        if topic not in concept_ids:
            concept_ids.append(topic)
    for topic in user.get("mastery", {}).keys():
        if topic not in concept_ids:
            concept_ids.append(topic)
    for topic in weak_areas:
        if topic not in concept_ids:
            concept_ids.append(topic)
    for node in user.get("practice_nodes", []):
        topic = node.get("concept")
        if topic and topic not in concept_ids:
            concept_ids.append(topic)
    nodes = []
    for concept in concept_ids:
        if concept in weak:
            status = "weak"
        elif concept in learned:
            status = "mastered"
        else:
            status = "pending"
        nodes.append({"id": concept, "status": status})
    practice_nodes = user.get("practice_nodes", [])
    for node in practice_nodes:
        nodes.append({
            "id": node.get("id", "Practice Node"),
            "status": node.get("status", "pending")
        })
    
    edges = [
        {"source": "Basic Operations", "target": "Linear Equations"},
        {"source": "Fractions", "target": "Linear Equations"},
        {"source": "Decimals", "target": "Linear Equations"},
        {"source": "Linear Equations", "target": "Sign Management"},
        {"source": "Linear Equations", "target": "Algebra Basics"},
        {"source": "Algebra Basics", "target": "Quadratic Functions"},
        {"source": "Fractions", "target": "Componendo and Dividendo"},
        {"source": "Basic Operations", "target": "Sign Management"}
    ]
    for node in practice_nodes:
        concept = node.get("concept")
        node_id = node.get("id")
        if concept in concept_ids and node_id:
            edges.append({"source": concept, "target": node_id})
    
    learning_goals = user.get('learning_goals', [])
    recommended_next = weak_areas[0] if weak_areas else (learning_goals[0] if learning_goals else "Linear Equations")
    return jsonify({"nodes": nodes, "edges": edges, "recommended_next": recommended_next})

if __name__ == '__main__':
    load_openai_client()
    app.run(host='0.0.0.0', port=5000, debug=True)
