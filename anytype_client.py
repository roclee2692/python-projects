"""
Anytype API 客户端
用于通过 Python 操作本地 Anytype 笔记库
API 文档: https://doc.anytype.io/anytype-api
"""

import requests
import json
from typing import Optional

API_BASE = "http://localhost:31009/v1"
API_KEY = "tdC7Sib+jABmFjYai4QbX0ycO/ICM/qs93SKMygxxRE="
ANYTYPE_VERSION = "2025-11-08"

# 已知空间
SPACES = {
    "essays": "bafyreiafdqvsf73h6s35hwdeazz322grli67344asut6xphh2u6chutvpe.u62ytu6jfj5t",
    "文件库": "bafyreibhkvsak7hbksv6qoujxyfcsf2war5yzyhqwz7qqye6466sfbkmaq.u62ytu6jfj5t",
}

DEFAULT_SPACE = "文件库"


class AnytypeClient:
    def __init__(self, api_key: str = API_KEY, base_url: str = API_BASE):
        self.base_url = base_url
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Anytype-Version": ANYTYPE_VERSION,
        }

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.base_url}{path}"
        resp = requests.request(method, url, headers=self.headers, **kwargs)
        resp.raise_for_status()
        return resp.json()

    # ── Spaces ──────────────────────────────────────────────
    def list_spaces(self) -> list:
        data = self._request("GET", "/spaces")
        return data["data"]

    def get_space(self, space_id: str) -> dict:
        return self._request("GET", f"/spaces/{space_id}")

    # ── Types ───────────────────────────────────────────────
    def list_types(self, space_id: str) -> list:
        data = self._request("GET", f"/spaces/{space_id}/types")
        return data["data"]

    # ── Objects ─────────────────────────────────────────────
    def search_objects(self, space_id: str, query: str, limit: int = 10, offset: int = 0) -> list:
        """搜索对象"""
        payload = {"query": query, "limit": limit, "offset": offset}
        data = self._request("POST", f"/spaces/{space_id}/search", json=payload)
        return data.get("data", [])

    def get_object(self, space_id: str, object_id: str) -> dict:
        """获取对象详情"""
        return self._request("GET", f"/spaces/{space_id}/objects/{object_id}")

    def get_object_content(self, space_id: str, object_id: str) -> str:
        """获取对象的 Markdown 内容"""
        resp = requests.get(
            f"{self.base_url}/spaces/{space_id}/objects/{object_id}",
            headers={**self.headers, "Accept": "text/markdown"},
        )
        resp.raise_for_status()
        return resp.text

    def create_object(
        self,
        space_id: str,
        name: str,
        content_markdown: str = "",
        type_key: str = "page",
        icon_emoji: Optional[str] = None,
    ) -> dict:
        """创建新对象（笔记/页面）"""
        payload = {
            "name": name,
            "type_key": type_key,
            "body": content_markdown,
        }
        if icon_emoji:
            payload["icon"] = {"format": "emoji", "emoji": icon_emoji}
        resp = self._request("POST", f"/spaces/{space_id}/objects", json=payload)
        return resp.get("object", resp)

    def update_object(
        self,
        space_id: str,
        object_id: str,
        name: Optional[str] = None,
        content_markdown: Optional[str] = None,
    ) -> dict:
        """更新对象"""
        payload = {}
        if name is not None:
            payload["name"] = name
        if content_markdown is not None:
            payload["body"] = content_markdown
        return self._request("PATCH", f"/spaces/{space_id}/objects/{object_id}", json=payload)

    def delete_object(self, space_id: str, object_id: str) -> dict:
        """删除对象（移到回收站）"""
        return self._request("DELETE", f"/spaces/{space_id}/objects/{object_id}")

    # ── Lists / Collections ─────────────────────────────────
    def list_objects(self, space_id: str, type_key: str = "page", limit: int = 20, offset: int = 0) -> list:
        """按类型列出对象（通过搜索实现）"""
        payload = {"type_key": type_key, "limit": limit, "offset": offset}
        data = self._request("POST", f"/spaces/{space_id}/search", json=payload)
        return data.get("data", [])


def _resolve_space(space: Optional[str]) -> str:
    """将空间名称或 ID 解析为 space_id"""
    if space is None:
        return SPACES[DEFAULT_SPACE]
    if space in SPACES:
        return SPACES[space]
    return space  # 假设是完整 ID


def main():
    """快速测试 & 演示"""
    import sys
    import io
    # 确保 Windows 终端正确输出中文
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    client = AnytypeClient()

    # 列出空间
    print("=== 空间列表 ===")
    for s in client.list_spaces():
        print(f"  {s['name']}  (id: {s['id'][:30]}...)")

    # 列出文件库中的对象
    space_id = SPACES[DEFAULT_SPACE]
    print(f"\n=== 「{DEFAULT_SPACE}」中的对象 (前5个) ===")
    results = client.search_objects(space_id, "", limit=5)
    for r in results:
        t = r.get('type', {}).get('key', '?')
        name = r.get('name') or '(无标题)'
        snippet = (r.get('snippet', '') or '')[:80]
        print(f"  [{t}] {name}")
        if snippet:
            print(f"    -> {snippet}")
        print(f"    id: {r['id'][:40]}...")


if __name__ == "__main__":
    main()
