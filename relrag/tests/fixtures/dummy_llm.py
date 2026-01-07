from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading
import time

class DummyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Construct a valid note response
            # NoteGenerator expects: 
            # [{"subj": "...", "pred": "...", "obj": "...", "evidence": "...", "meta": {...}}]
            # Or the original entity/attribute format which validator normalizes.
            # Let's provide the "normalized" format directly or close to it to ensure success.
            # Based on NoteValidator, it patches "pred" if missing using attribute name.
            
            response_content = json.dumps([
                {
                    "subj": "Albert Einstein",
                    "obj": "Ulm",
                    "pred": "born_in",
                    "subj_type": "PERSON",
                    "obj_type": "PLACE",
                    "evidence": "Albert Einstein was born in Ulm.",
                    "meta": {
                        "confidence": 1.0,
                        "attribute": {
                            "name": "born_in",
                            "values": [{"value": "Ulm", "confidence": 1.0}]
                        },
                        "subject_profile": {"type": "PERSON"}
                    }
                }
            ])
            
            # Wrap in FINAL tag as required by prompt
            full_content = f"FINAL: \n{response_content}"

            response = {
                "id": "chatcmpl-dummy",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "test-model",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_content
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20
                }
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run(port=8001):
    server_address = ('', port)
    httpd = HTTPServer(server_address, DummyHandler)
    print(f"Starting dummy LLM on port {port}...")
    httpd.serve_forever()

if __name__ == "__main__":
    run()
