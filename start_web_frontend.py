import http.server
import socketserver
import os
from pathlib import Path

# Change to the web-frontend directory
web_dir = Path("web-frontend")

if __name__ == "__main__":
    os.chdir(web_dir)

    PORT = 8080

    Handler = http.server.SimpleHTTPRequestHandler

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Flight Delay Prediction Web Frontend running at http://localhost:{PORT}")
        print("Make sure the backend API is running at http://localhost:8000")
        print("Press Ctrl+C to stop the server")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")