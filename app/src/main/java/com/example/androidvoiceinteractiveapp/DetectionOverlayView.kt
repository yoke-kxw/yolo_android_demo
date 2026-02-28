package com.example.androidvoiceinteractiveapp

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseLandmark

class DetectionOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var detections: List<YoloV8Detector.Detection> = emptyList()
    private var pose: Pose? = null
    private var sourceWidth: Int = 1
    private var sourceHeight: Int = 1

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#00A8B5")
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }

    private val textBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#AA000000")
        style = Paint.Style.FILL
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 32f
    }

    private val skeletonPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FF7B54")
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }

    private val jointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FF9800")
        style = Paint.Style.FILL
    }

    fun update(detections: List<YoloV8Detector.Detection>, pose: Pose?, sourceWidth: Int, sourceHeight: Int) {
        this.detections = detections
        this.pose = pose
        this.sourceWidth = if (sourceWidth > 0) sourceWidth else 1
        this.sourceHeight = if (sourceHeight > 0) sourceHeight else 1
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val sx = width.toFloat() / sourceWidth
        val sy = height.toFloat() / sourceHeight
        val scale = minOf(sx, sy)
        val dx = (width - sourceWidth * scale) / 2f
        val dy = (height - sourceHeight * scale) / 2f

        for (det in detections) {
            val left = det.left * scale + dx
            val top = det.top * scale + dy
            val right = det.right * scale + dx
            val bottom = det.bottom * scale + dy

            canvas.drawRect(left, top, right, bottom, boxPaint)

            val label = "${det.className} ${(det.score * 100).toInt()}%"
            val textWidth = textPaint.measureText(label)
            val textTop = (top - 38f).coerceAtLeast(0f)
            canvas.drawRect(left, textTop, left + textWidth + 20f, textTop + 40f, textBgPaint)
            canvas.drawText(label, left + 10f, textTop + 30f, textPaint)
        }

        val p = pose ?: return
        for ((a, b) in POSE_CONNECTIONS) {
            val la = p.getPoseLandmark(a)
            val lb = p.getPoseLandmark(b)
            if (la != null && lb != null) {
                val x1 = la.position.x * scale + dx
                val y1 = la.position.y * scale + dy
                val x2 = lb.position.x * scale + dx
                val y2 = lb.position.y * scale + dy
                canvas.drawLine(x1, y1, x2, y2, skeletonPaint)
            }
        }

        for (landmark in p.allPoseLandmarks) {
            val x = landmark.position.x * scale + dx
            val y = landmark.position.y * scale + dy
            canvas.drawCircle(x, y, 6f, jointPaint)
        }
    }

    companion object {
        private val POSE_CONNECTIONS = listOf(
            PoseLandmark.LEFT_SHOULDER to PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.LEFT_SHOULDER to PoseLandmark.LEFT_ELBOW,
            PoseLandmark.LEFT_ELBOW to PoseLandmark.LEFT_WRIST,
            PoseLandmark.RIGHT_SHOULDER to PoseLandmark.RIGHT_ELBOW,
            PoseLandmark.RIGHT_ELBOW to PoseLandmark.RIGHT_WRIST,
            PoseLandmark.LEFT_SHOULDER to PoseLandmark.LEFT_HIP,
            PoseLandmark.RIGHT_SHOULDER to PoseLandmark.RIGHT_HIP,
            PoseLandmark.LEFT_HIP to PoseLandmark.RIGHT_HIP,
            PoseLandmark.LEFT_HIP to PoseLandmark.LEFT_KNEE,
            PoseLandmark.LEFT_KNEE to PoseLandmark.LEFT_ANKLE,
            PoseLandmark.RIGHT_HIP to PoseLandmark.RIGHT_KNEE,
            PoseLandmark.RIGHT_KNEE to PoseLandmark.RIGHT_ANKLE
        )
    }
}
