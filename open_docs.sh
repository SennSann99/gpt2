#!/usr/bin/env bash
set -e
PORT=8080
URL="http://localhost:$PORT"
echo "ドキュメントサーバーを起動します (Port: $PORT)..."

(
  sleep 2
  if command -v open > /dev/null; then
    open "$URL"
  elif command -v xdg-open > /dev/null; then
    xdg-open "$URL"
  elif command -v start > /dev/null; then
    start "$URL"
  else
    echo "ブラウザを自動で開けませんでした。手動で $URL にアクセスしてください。"
  fi
) &

echo "サーバーがキャッシュ無効化モードで起動しました。終了するには Ctrl+C を押してください。"

python3 -c "
import http.server
import socketserver
import sys

class NoCacheHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # より強力なキャッシュ無効化ヘッダー
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

# Ctrl+C後にポートをすぐに解放するための設定
class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

PORT = int('$PORT')
DIRECTORY = 'docs'

class Handler(NoCacheHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

try:
    with ReusableTCPServer(('', PORT), Handler) as httpd:
        httpd.serve_forever()
except KeyboardInterrupt:
    print('\nサーバーを安全に停止し、ポートを解放しました。')
    sys.exit(0)
"