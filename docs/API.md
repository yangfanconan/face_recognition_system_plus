# API 接口文档

## FastAPI 服务

### 基础信息

- **基础 URL**: `http://localhost:8000`
- **API 文档**: `http://localhost:8000/docs`
- **健康检查**: `http://localhost:8000/health`

## 接口列表

### 1. 人脸检测

#### POST /detect

检测图像中的人脸

**请求参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | file | 是 | 上传的图像文件 |
| conf_threshold | float | 否 | 置信度阈值 (默认 0.3) |
| nms_threshold | float | 否 | NMS 阈值 (默认 0.5) |
| do_recognition | bool | 否 | 是否识别 (默认 true) |
| do_search | bool | 否 | 是否检索 (默认 true) |

**请求示例**:
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@test.jpg" \
  -F "conf_threshold=0.5"
```

**响应示例**:
```json
{
  "request_id": "uuid-xxx",
  "faces": [
    {
      "bbox": [100, 100, 200, 200],
      "confidence": 0.95,
      "identity_id": 12345,
      "identity_score": 0.89
    }
  ],
  "latency_ms": {
    "detection": 2.1,
    "recognition": 0.5,
    "search": 5.2,
    "total": 7.8
  },
  "image_shape": [480, 640],
  "status": "success"
}
```

---

#### POST /detect/base64

Base64 图像检测

**请求参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| image_base64 | string | 是 | Base64 编码的图像 |

**请求示例**:
```bash
curl -X POST "http://localhost:8000/detect/base64" \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "iVBORw0KGgoAAAANS..."}'
```

---

### 2. 人脸验证

#### POST /verify

1:1 人脸验证

**请求参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| feature1 | array | 是 | 特征向量 1 |
| feature2 | array | 是 | 特征向量 2 |
| threshold | float | 否 | 验证阈值 (默认 0.5) |

**请求示例**:
```bash
curl -X POST "http://localhost:8000/verify" \
  -H "Content-Type: application/json" \
  -d '{
    "feature1": [0.1, 0.2, ...],
    "feature2": [0.3, 0.4, ...],
    "threshold": 0.5
  }'
```

**响应示例**:
```json
{
  "is_same": true,
  "similarity": 0.85,
  "request_id": "uuid-xxx"
}
```

---

### 3. 人脸检索

#### POST /search

1:N 人脸检索

**请求参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| feature | array | 是 | 查询特征向量 |
| top_k | int | 否 | 返回数量 (默认 10) |
| threshold | float | 否 | 分数阈值 (默认 0.5) |

**请求示例**:
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "feature": [0.1, 0.2, ...],
    "top_k": 5
  }'
```

**响应示例**:
```json
{
  "request_id": "uuid-xxx",
  "results": [
    {
      "face_id": 12345,
      "score": 0.89,
      "distance": 0.45,
      "metadata": {"name": "张三"}
    }
  ],
  "latency_ms": 5.2,
  "status": "success"
}
```

---

### 4. 人脸管理

#### POST /register

注册新人脸

**请求参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | file | 是 | 人脸图像 |
| face_id | string | 否 | 自定义人脸 ID |
| metadata | string | 否 | JSON 格式元数据 |

**请求示例**:
```bash
curl -X POST "http://localhost:8000/register" \
  -F "file=@face.jpg" \
  -F "face_id=user_001" \
  -F 'metadata={"name": "张三", "department": "技术部"}'
```

---

#### DELETE /face/{face_id}

删除人脸

**请求示例**:
```bash
curl -X DELETE "http://localhost:8000/face/12345"
```

---

#### GET /face/{face_id}

获取人脸信息

**响应示例**:
```json
{
  "face_id": 12345,
  "metadata": {
    "name": "张三",
    "department": "技术部"
  },
  "status": "success"
}
```

---

### 5. 批量处理

#### POST /batch/detect

批量人脸检测

**请求参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| files | file[] | 是 | 图像文件列表 |

**请求示例**:
```bash
curl -X POST "http://localhost:8000/batch/detect" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

---

### 6. 监控

#### GET /health

健康检查

**响应示例**:
```json
{
  "status": "healthy",
  "uptime_seconds": 3600.5,
  "request_count": 1000,
  "avg_latency_ms": 7.5
}
```

---

#### GET /stats

获取服务统计

**响应示例**:
```json
{
  "pipeline_stats": {
    "total_inferences": 1000,
    "avg_latency_ms": 7.5,
    "fps": 133.3
  },
  "app_stats": {
    "request_count": 1000,
    "uptime_seconds": 3600.5
  }
}
```

---

### 7. WebSocket 实时处理

#### WS /ws/realtime

实时人脸检测

**连接示例**:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/realtime');

// 发送图像
ws.send(imageData);

// 接收结果
ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  console.log(result);
};
```

---

## 错误码

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 404 | 资源未找到 |
| 500 | 服务器内部错误 |
| 503 | 服务不可用 |

## 限流

- 默认：100 请求/秒
- 批量请求：10 请求/秒

## 认证

当前版本无需认证，生产环境建议添加 API Key 认证。
