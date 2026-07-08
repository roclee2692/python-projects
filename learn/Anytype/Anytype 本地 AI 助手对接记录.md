# Anytype 本地 AI 助手对接记录

> 首次成功通过 Python 脚本 + Claude Code 操作 Anytype 笔记库的完整记录。
> 日期：2026-06-04

---

## 一、环境搭建

### API 连接信息

| 项目 | 值 |
|------|-----|
| API 地址 | `http://localhost:31009/v1` |
| API 版本 | `2025-11-08` |
| 认证方式 | `Authorization: Bearer <API_KEY>` |
| 桌面客户端 | Anytype 桌面版（Windows） |

### 两个空间

| 空间名称 | Space ID |
|----------|----------|
| 文件库 | `bafyreibhkvsak7hbksv6qoujxyfcsf2war5yzyhqwz7qqye6466sfbkmaq.u62ytu6jfj5t` |
| Essays | `bafyreiafdqvsf73h6s35hwdeazz322grli67344asut6xphh2u6chutvpe.u62ytu6jfj5t` |

默认操作空间：**文件库**

---

## 二、Python 客户端脚本

创建了 `anytype_client.py`，封装了 Anytype API 的常用操作：

```python
from anytype_client import AnytypeClient, SPACES

client = AnytypeClient()
space_id = SPACES['文件库']

# 搜索笔记
results = client.search_objects(space_id, '关键词', limit=10)

# 创建笔记
client.create_object(space_id, name='标题', content_markdown='内容', type_key='page')

# 获取笔记内容
content = client.get_object_content(space_id, object_id)

# 更新笔记
client.update_object(space_id, object_id, name='新标题', content_markdown='新内容')
```

### 核心 API 端点

| 操作 | 方法 | 端点 |
|------|------|------|
| 列出空间 | GET | `/spaces` |
| 列出对象 | GET | `/spaces/:id/objects?limit=100&offset=0` |
| 搜索对象 | POST | `/spaces/:id/search` |
| 获取对象 | GET | `/spaces/:id/objects/:obj_id` |
| 创建对象 | POST | `/spaces/:id/objects` |
| 更新对象 | PATCH | `/spaces/:id/objects/:obj_id` |
| 删除对象 | DELETE | `/spaces/:id/objects/:obj_id` |
| 列出标签 | GET | `/spaces/:id/properties/:prop_id/tags` |
| 创建标签 | POST | `/spaces/:id/properties/:prop_id/tags` |

### 关键发现

1. **分页**：`GET /objects` 支持 `offset` + `limit` 参数，`POST /search` 的 offset 无效
2. **标签格式**：更新标签时用 `key` 字段（如 `ai`），不是 `id` 或 `name`
3. **响应结构**：创建对象返回 `{"object": {...}}`，不是 `{"data": {...}}`
4. **标签属性 ID**：`bafyreicxagwmcvkwcsetde6zp6giuzt5vonwdt2d2ndqsngra7zk3c6wsi`（page 和 note 共用）

---

## 三、首次创建笔记

创建了一篇随笔「深夜写给自己」，写入「文件库」空间。

- **Object ID**: `bafyreicgd7oipbfd6isksbboxusau4jtubihsbshwrx66bspettvwomtui`
- **类型**: page
- **内容**: 自我激励随笔，引用芒格名言

---

## 四、笔记分类打标

### 笔记库规模

| 空间 | 对象类型 | 数量 |
|------|---------|------|
| 文件库 | Page | 111 |
| 文件库 | Note | 149 |
| 文件库 | Bookmark | 4 |
| Essays | Page | 31 |
| **合计** | - | **295** |

### 标签体系（15 个标签）

| 标签 Key | 名称 | 覆盖笔记 | 说明 |
|----------|------|---------|------|
| `ying_yu` | 英语 | ~57 | 英语学习、语法、词汇 |
| `gong_ju_bei_wang` | 工具备忘 | ~58 | 密码、配置、软件清单 |
| `ai` | AI | ~38 | AI/ML 项目、实验 |
| `zi_wo_cheng_chang` | 自我成长 | ~34 | 规划、习惯、反思 |
| `life` | life | ~28 | 生活哲学、娱乐 |
| `ren_zhi` | 认知 | ~21 | 哲学、深度思考 |
| `xie_zuo` | 写作 | ~21 | 内容创作、播客 |
| `xue_xi_fang_fa` | 学习方法 | ~17 | 考试策略、学习技巧 |
| `ielts` | IELTS | ~13 | 雅思备考 |
| `bian_cheng` | 编程 | ~9 | 代码、算法竞赛 |
| `tou_zi` | 投资 | ~9 | 投资原则 |
| `de_guo` | 德国🇩🇪 | ~8 | 留学申请、APS |
| `zhi_ye_fa_zhan` | 职业发展 | ~20 | 路径选择、作品集 |
| `jin_rong` | 金融 | ~7 | 财务自由、保险 |
| `shu_ji` | 书籍 | ~5 | 书单整理 |
| `ren_ji_qing_gan` | 人际情感 | ~12 | 亲密关系、社交哲学 |
| `meaning` | Meaning | ~5 | 人生意义 |

### 打标结果

**文件库 264 条对象全部打标完成（100% 覆盖）。**

- 第一批：100 条（搜索 API 返回的前 100 条）
- 第二批：166 条（通过 `GET /objects` 分页获取的剩余对象）
- 补标：6 条（2 个 page + 4 个 bookmark）

### 打标方式

通过 `PATCH /spaces/:id/objects/:obj_id` 更新对象的 `tag` 属性：

```json
{
  "properties": [
    {
      "key": "tag",
      "objects": ["ai", "bian_cheng"]
    }
  ]
}
```

每个笔记分配 1-3 个标签，避免过度标注。

---

## 五、待办事项

- [ ] Essays 空间 31 条雅思写作范文待打标
- [ ] 考虑为 Essays 空间建立专门的写作类标签体系
- [ ] 定期同步新笔记的标签
- [ ] 探索 Anytype MCP Server 集成（`@anyproto/anytype-mcp`）

---

## 六、生成的文件清单

| 文件 | 说明 |
|------|------|
| `anytype_client.py` | Python API 客户端 |
| `notes_inventory.json` | 前 100 条笔记清单 |
| `notes_inventory_full.json` | 全部 266 条笔记清单 |
| `anytype_tagging_scheme.json` | 第一批 100 条的标签分配方案 |
| `notes_tagging_remaining.json` | 第二批 166 条的标签分配方案 |
| `existing_tags.json` | 已有标签列表 |
| `learn/Anytype/first.md` | 本文档 |
