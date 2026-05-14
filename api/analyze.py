import json
import os

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

client = None
if HAS_OPENAI and os.getenv('OPENAI_API_KEY'):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def handler(request):
    if not client:
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                "overall_score": "75",
                "correct": 5,
                "incorrect": 2,
                "partial": 1,
                "summary": "This is a sample summary. Set OPENAI_API_KEY environment variable in Vercel to use real GPT-4o Mini analysis.",
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
                "bounding_boxes": [
                    {"page": 1, "x": 100, "y": 150, "width": 200, "height": 80, "type": "error", "description": "Quadratic formula mistake"}
                ]
            })
        }
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            "overall_score": "75",
            "correct": 5,
            "incorrect": 2,
            "partial": 1,
            "summary": "Sample analysis. Full GPT integration would require server-side file upload handling.",
            "mistakes": [],
            "corrections": [],
            "strengths": [],
            "weaknesses": [],
            "focus_areas": [],
            "bounding_boxes": []
        })
    }
