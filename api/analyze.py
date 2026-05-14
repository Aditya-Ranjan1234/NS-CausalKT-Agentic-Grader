import json
from http.server import BaseHTTPRequestHandler


def sample_analysis():
    return {
        "overall_score": "75",
        "correct": 5,
        "incorrect": 2,
        "partial": 1,
        "summary": "Sample analysis. Full file-based grading is available in the local Flask backend.",
        "mistakes": [
            {
                "title": "Algebraic Error",
                "question": 2,
                "description": "Sign error in quadratic equation solution.",
                "correction": "Review sign handling in the formula."
            }
        ],
        "corrections": [],
        "strengths": [],
        "weaknesses": [],
        "focus_areas": [],
        "bounding_boxes": []
    }


class handler(BaseHTTPRequestHandler):
    def _send(self, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        self.end_headers()

    def do_OPTIONS(self):
        self._send(200)
        self.wfile.write(b'{}')

    def do_GET(self):
        self._send(200)
        self.wfile.write(json.dumps(sample_analysis()).encode('utf-8'))

    def do_POST(self):
        self._send(200)
        self.wfile.write(json.dumps(sample_analysis()).encode('utf-8'))
