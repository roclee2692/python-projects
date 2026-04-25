"""
支持 HTTP Range 请求的文件服务器（用于断点续传大文件）
运行: python range_http_server.py [port]
"""
import http.server
import os
import sys


class RangeHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """在标准 SimpleHTTPRequestHandler 基础上增加 Range 支持"""

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()

        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        fs = os.fstat(f.fileno())
        file_size = fs.st_size

        range_header = self.headers.get("Range")
        if range_header:
            # 解析 Range: bytes=start-end
            try:
                range_spec = range_header.strip().replace("bytes=", "")
                start_str, end_str = range_spec.split("-")
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else file_size - 1
                end = min(end, file_size - 1)
                length = end - start + 1

                f.seek(start)
                self.send_response(206, "Partial Content")
                self.send_header("Content-Type", self.guess_type(path))
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
                self.end_headers()
                return f
            except Exception as e:
                print(f"Range parse error: {e}, falling back to full response")
                f.seek(0)

        # 普通完整响应
        self.send_response(200)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Length", str(file_size))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
        self.end_headers()
        return f

    def log_message(self, format, *args):
        # 简化日志：只显示关键信息
        if "206" in str(args) or "GET" in str(args[0]):
            print(f"[{self.address_string()}] {format % args}")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8888
    serve_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(serve_dir)

    handler = RangeHTTPRequestHandler
    with http.server.ThreadingHTTPServer(("", port), handler) as httpd:
        print(f"支持断点续传的文件服务器已启动")
        print(f"服务目录: {serve_dir}")
        print(f"端口: {port}")
        print(f"访问地址: http://192.168.1.13:{port}/")
        print("Ctrl+C 停止")
        httpd.serve_forever()
