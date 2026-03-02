# yolo_android

一个从原始工程中抽离出来的纯净 Android YOLO 实时识别 Demo。

当前工程聚焦于本地视觉推理
## 当前能力

- CameraX 实时摄像头预览
- YOLO 本地目标检测
- 姿态骨架叠加（ML Kit Pose）
- 底部实时显示当前识别结果
- 支持在 `YOLO26` 与旧版 `YOLOv8` 模型间切换
- 支持截图并保存到系统相册

## 当前模型

工程内同时保留了两套模型：

- `app/src/main/assets/yolo26n.onnx`
- `app/src/main/assets/yolov8n_opset21.onnx`

加载逻辑：

1. 默认优先加载 `yolo26n.onnx`
2. 如果缺失或加载失败，回退到 `yolov8n_opset21.onnx`
3. 页面底部会实时显示当前实际加载的模型文件名

右下角按钮可以手动切换：

- `Use YOLOv8`
- `Use YOLO26`

## 截图说明

页面支持截图，保存的是：

- 相机实时画面
- 检测框
- 姿态骨架叠加

不是只保存 UI 外壳。

保存位置：

- Android 10 及以上：通过 `MediaStore` 写入系统相册，目录为 `Pictures/YOLODemo`
- Android 9 及以下：写入公共图片目录 `Pictures/YOLODemo`，并主动刷新媒体库

## 页面交互

- 左上角：返回
- 底部左侧：截图
- 底部右侧：切换模型
- 底部信息栏：显示当前模型、目标数量、识别到的类别、姿态状态

信息栏为固定高度，不会因为文本长短改变预览区大小。

## 项目结构

- `app/src/main/java/com/example/androidvoiceinteractiveapp/YoloDemoActivity.kt`
  - 主页面
  - 摄像头取流
  - 姿态识别
  - 模型切换
  - 截图保存

- `app/src/main/java/com/example/androidvoiceinteractiveapp/YoloV8Detector.kt`
  - ONNX Runtime 推理
  - 同时兼容两种输出格式：
    - 旧版 YOLOv8：`(1, 84, 8400)`
    - YOLO26：`(1, 300, 6)`

- `app/src/main/java/com/example/androidvoiceinteractiveapp/DetectionOverlayView.kt`
  - 绘制检测框
  - 绘制姿态骨架

## 环境要求

- Android Studio（建议较新版本）
- Android SDK 已正确配置
- JDK 17
- 真机优先（摄像头与 ONNX Runtime 更稳定）
- 64 位设备/64 位进程

注意：

- 当前代码在 32 位进程下会直接提示并退出
- 因为 ONNX Runtime 在部分 32 位 ARM 环境存在稳定性问题

## 构建方式

在项目根目录执行：

```bash
./gradlew.bat :app:assembleDebug
```

APK 输出路径：

```text
app/build/outputs/apk/debug/app-debug.apk
```

## 已验证状态

当前工程已经完成独立编译验证：

- `:app:assembleDebug`
- 结果：`BUILD SUCCESSFUL`

## 适合继续扩展的方向

这个 Demo 现在适合作为以下方向的基础壳：

1. 运动姿态计数
   - 深蹲
   - 俯卧撑
   - 姿态纠正

2. 视觉识别工具
   - 实时“看到了什么”
   - 寻物
   - 盘点计数

3. 视觉工具接入 AI
   - 把当前识别结果作为 Tool 输出
   - 把截图路径作为 Tool 输出
   - 接到上层问答系统做“看图说话”

## 当前限制

- 还没有实例分割
- 还没有旋转框（OBB）绘制
- 还没有多目标跟踪
- 还没有把识别结果接到外部 AI/MCP 服务
- 截图当前保存的是“预览画面 + 检测层”，不包含额外业务叠加逻辑

## 备注

这个工程是从原项目抽离出来的“纯净视觉 Demo”，目的是降低耦合，方便单独验证和迭代 YOLO 能力。
