package com.example.androidvoiceinteractiveapp

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.util.PriorityQueue
import kotlin.math.max
import kotlin.math.min

class YoloV8Detector(
    context: Context,
    preferredModelName: String? = null
) : AutoCloseable {

    class ModelLoadException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)

    data class Detection(
        val classId: Int,
        val className: String,
        val score: Float,
        val left: Float,
        val top: Float,
        val right: Float,
        val bottom: Float
    )

    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val sessionLock = Any()
    @Volatile
    private var isClosed = false
    val loadedModelName: String

    //    这一段是 COCO 80 类，对应官方 yolov8n.pt 默认检测类别
    private val labels = arrayOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
        "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    )

    init {
        val candidates = buildList {
            preferredModelName?.let { add(it) }
            add("yolo26n.onnx")
            add("yolov8n_opset21.onnx")
            add("yolov8n.onnx")
        }.distinct()
        var lastError: Throwable? = null
        var selectedName: String? = null
        var selectedSession: OrtSession? = null

        for (name in candidates) {
            try {
                val modelBytes = context.assets.open(name).use { it.readBytes() }
                selectedSession = ortEnv.createSession(modelBytes, OrtSession.SessionOptions())
                selectedName = name
                break
            } catch (e: Throwable) {
                lastError = e
            }
        }

        if (selectedSession == null || selectedName == null) {
            val msg = when (lastError) {
                is OrtException -> "ONNX model is incompatible with runtime. Please use opset <= 21 model file."
                else -> "Failed to load YOLO model from assets."
            }
            throw ModelLoadException(msg, lastError)
        }

        session = selectedSession
        loadedModelName = selectedName
    }

    fun detect(bitmap: Bitmap, confThreshold: Float = 0.35f, iouThreshold: Float = 0.45f, maxDetections: Int = 40): List<Detection> {
        if (isClosed) return emptyList()

        val inputSize = 640
        val srcW = bitmap.width
        val srcH = bitmap.height
        val scaled = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        val input = FloatArray(1 * 3 * inputSize * inputSize)
        val pixels = IntArray(inputSize * inputSize)
        scaled.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        val stride = inputSize * inputSize
        for (y in 0 until inputSize) {
            for (x in 0 until inputSize) {
                val idx = y * inputSize + x
                val p = pixels[idx]
                input[idx] = (p shr 16 and 0xFF) / 255f
                input[stride + idx] = (p shr 8 and 0xFF) / 255f
                input[stride * 2 + idx] = (p and 0xFF) / 255f
            }
        }

        return try {
            synchronized(sessionLock) {
                if (isClosed) return emptyList()
                val inputName = session.inputNames.iterator().next()
                val tensor = OnnxTensor.createTensor(
                    ortEnv,
                    FloatBuffer.wrap(input),
                    longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
                )

                tensor.use {
                    session.run(mapOf(inputName to tensor)).use { outputs ->
                        @Suppress("UNCHECKED_CAST")
                        val out = outputs[0].value as Array<Array<FloatArray>>
                        val preds = out[0]
                        val boxes = if (preds.isNotEmpty() && preds[0].size == 6) {
                            parseEndToEndOutput(preds, srcW, srcH, confThreshold)
                        } else {
                            parseLegacyOutput(preds, srcW, srcH, inputSize, confThreshold)
                        }

                        nms(boxes, iouThreshold, maxDetections)
                    }
                }
            }
        } finally {
            if (scaled !== bitmap && !scaled.isRecycled) {
                scaled.recycle()
            }
        }
    }

    private fun nms(dets: List<Detection>, iouThreshold: Float, maxDetections: Int): List<Detection> {
        if (dets.isEmpty()) return emptyList()

        val kept = ArrayList<Detection>()
        val byClass = dets.groupBy { it.classId }

        for ((_, list) in byClass) {
            val pq = PriorityQueue<Detection>(compareByDescending { it.score })
            pq.addAll(list)
            val classKept = ArrayList<Detection>()

            while (pq.isNotEmpty() && classKept.size < maxDetections) {
                val best = pq.poll() ?: break
                classKept.add(best)

                val remain = ArrayList<Detection>()
                while (pq.isNotEmpty()) {
                    val candidate = pq.poll() ?: continue
                    if (iou(best, candidate) < iouThreshold) {
                        remain.add(candidate)
                    }
                }
                pq.addAll(remain)
            }

            kept.addAll(classKept)
        }

        return kept.sortedByDescending { it.score }.take(maxDetections)
    }

    private fun iou(a: Detection, b: Detection): Float {
        val interLeft = max(a.left, b.left)
        val interTop = max(a.top, b.top)
        val interRight = min(a.right, b.right)
        val interBottom = min(a.bottom, b.bottom)

        val interW = max(0f, interRight - interLeft)
        val interH = max(0f, interBottom - interTop)
        val interArea = interW * interH
        if (interArea <= 0f) return 0f

        val areaA = max(0f, a.right - a.left) * max(0f, a.bottom - a.top)
        val areaB = max(0f, b.right - b.left) * max(0f, b.bottom - b.top)
        return interArea / (areaA + areaB - interArea + 1e-6f)
    }

    private fun parseLegacyOutput(
        preds: Array<FloatArray>,
        srcW: Int,
        srcH: Int,
        inputSize: Int,
        confThreshold: Float
    ): List<Detection> {
        val numAnchors = preds[0].size
        val boxes = ArrayList<Detection>(numAnchors)

        for (i in 0 until numAnchors) {
            var bestClass = -1
            var bestScore = 0f

            for (c in 4 until preds.size) {
                val score = preds[c][i]
                if (score > bestScore) {
                    bestScore = score
                    bestClass = c - 4
                }
            }

            if (bestScore < confThreshold || bestClass < 0) continue

            val cx = preds[0][i]
            val cy = preds[1][i]
            val w = preds[2][i]
            val h = preds[3][i]

            boxes.add(
                Detection(
                    classId = bestClass,
                    className = labels.getOrElse(bestClass) { "cls_$bestClass" },
                    score = bestScore,
                    left = ((cx - w / 2f) / inputSize) * srcW,
                    top = ((cy - h / 2f) / inputSize) * srcH,
                    right = ((cx + w / 2f) / inputSize) * srcW,
                    bottom = ((cy + h / 2f) / inputSize) * srcH
                )
            )
        }

        return boxes
    }

    private fun parseEndToEndOutput(
        preds: Array<FloatArray>,
        srcW: Int,
        srcH: Int,
        confThreshold: Float
    ): List<Detection> {
        val boxes = ArrayList<Detection>(preds.size)

        for (row in preds) {
            if (row.size < 6) continue

            val left = row[0].coerceIn(0f, srcW.toFloat())
            val top = row[1].coerceIn(0f, srcH.toFloat())
            val right = row[2].coerceIn(0f, srcW.toFloat())
            val bottom = row[3].coerceIn(0f, srcH.toFloat())
            val score = row[4]
            val classId = row[5].toInt()

            if (score < confThreshold) continue
            if (right <= left || bottom <= top) continue

            boxes.add(
                Detection(
                    classId = classId,
                    className = labels.getOrElse(classId) { "cls_$classId" },
                    score = score,
                    left = left,
                    top = top,
                    right = right,
                    bottom = bottom
                )
            )
        }

        return boxes
    }

    override fun close() {
        synchronized(sessionLock) {
            if (isClosed) return
            isClosed = true
            session.close()
        }
    }
}
