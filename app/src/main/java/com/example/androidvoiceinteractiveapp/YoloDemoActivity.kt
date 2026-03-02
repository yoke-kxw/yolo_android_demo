package com.example.androidvoiceinteractiveapp

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Process
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.graphics.createBitmap
import com.example.androidvoiceinteractiveapp.databinding.ActivityYoloDemoBinding
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.PoseDetector
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class YoloDemoActivity : AppCompatActivity() {

    companion object {
        private const val MODEL_YOLO26 = "yolo26n.onnx"
        private const val MODEL_YOLOV8 = "yolov8n_opset21.onnx"
    }

    private lateinit var binding: ActivityYoloDemoBinding
    private lateinit var cameraExecutor: ExecutorService
    private var yoloDetector: YoloV8Detector? = null
    private lateinit var poseDetector: PoseDetector
    @Volatile
    private var isShuttingDown = false
    private var selectedModelName = MODEL_YOLO26
    private var lastInfoLine = ""

    private val isProcessing = AtomicBoolean(false)

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera() else {
            Toast.makeText(this, "Camera permission is required", Toast.LENGTH_SHORT).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityYoloDemoBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnBack.setOnClickListener { finish() }
        binding.btnSwitchModel.setOnClickListener { switchModel() }
        binding.btnCapture.setOnClickListener { saveSnapshot() }

        cameraExecutor = Executors.newSingleThreadExecutor()

        // ORT has stability issues on some 32-bit ARM devices/processes.
        // Guard before any ORT call to avoid native crash in libonnxruntime.so.
        if (!Process.is64Bit()) {
            val abi = Build.SUPPORTED_ABIS.joinToString()
            val msg = "Current process is 32-bit ($abi). YOLO ONNX requires 64-bit process."
            Log.e("YoloDemoActivity", msg)
            Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
            binding.tvInfo.text = "YOLO ONNX requires 64-bit process. Please use arm64 device/runtime."
            finish()
            return
        }

        try {
            yoloDetector = YoloV8Detector(this, selectedModelName)
        } catch (t: Throwable) {
            Log.e("YoloDemoActivity", "Load YOLO model failed", t)
            val reason = t.message ?: "Unknown error"
            Toast.makeText(this, "YOLO model load failed: $reason", Toast.LENGTH_LONG).show()
            binding.tvInfo.text = "Model load failed. Put opset<=21 model as assets/yolov8n_opset21.onnx"
            finish()
            return
        }

        updateModelSwitchLabel()
        updateInfoText(getString(R.string.yolo_info_idle))
        poseDetector = PoseDetection.getClient(
            PoseDetectorOptions.Builder()
                .setDetectorMode(PoseDetectorOptions.STREAM_MODE)
                .build()
        )

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val cameraProvider = providerFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                analyzeFrame(imageProxy)
            }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            } catch (e: Exception) {
                Log.e("YoloDemoActivity", "Bind camera failed", e)
                runOnUiThread {
                    Toast.makeText(this, "Failed to start camera", Toast.LENGTH_SHORT).show()
                }
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(imageProxy: ImageProxy) {
        if (isShuttingDown) {
            imageProxy.close()
            return
        }
        if (!isProcessing.compareAndSet(false, true)) {
            imageProxy.close()
            return
        }

        try {
            val bitmap = imageProxy.toBitmapCompat()
            val rotated = bitmap.rotate(imageProxy.imageInfo.rotationDegrees.toFloat())
            if (bitmap !== rotated && !bitmap.isRecycled) {
                bitmap.recycle()
            }

            val detector = yoloDetector
            if (detector == null || isShuttingDown) {
                if (!rotated.isRecycled) {
                    rotated.recycle()
                }
                isProcessing.set(false)
                imageProxy.close()
                return
            }
            val detections = detector.detect(rotated)
            val inputImage = InputImage.fromBitmap(rotated, 0)

            poseDetector.process(inputImage)
                .addOnSuccessListener { pose ->
                    binding.overlayView.update(detections, pose, rotated.width, rotated.height)
                    updateInfoText(buildDetectionSummary(detections, if (pose.allPoseLandmarks.isNotEmpty()) "Detected" else "None"))
                }
                .addOnFailureListener {
                    binding.overlayView.update(detections, null, rotated.width, rotated.height)
                    updateInfoText(buildDetectionSummary(detections, "Error"))
                }
                .addOnCompleteListener {
                    if (!rotated.isRecycled) {
                        rotated.recycle()
                    }
                    isProcessing.set(false)
                    imageProxy.close()
                }
        } catch (e: Exception) {
            Log.e("YoloDemoActivity", "Analyze frame failed", e)
            isProcessing.set(false)
            imageProxy.close()
        }
    }

    private fun switchModel() {
        selectedModelName = if (selectedModelName == MODEL_YOLO26) MODEL_YOLOV8 else MODEL_YOLO26
        val previous = yoloDetector
        val next = try {
            YoloV8Detector(this, selectedModelName)
        } catch (t: Throwable) {
            Log.e("YoloDemoActivity", "Switch YOLO model failed", t)
            val reason = t.message ?: "Unknown error"
            Toast.makeText(this, "Switch model failed: $reason", Toast.LENGTH_LONG).show()
            return
        }

        yoloDetector = next
        previous?.close()
        updateModelSwitchLabel()
        updateInfoText(lastInfoLine.ifBlank { getString(R.string.yolo_info_idle) })
        Toast.makeText(this, "Switched to ${next.loadedModelName}", Toast.LENGTH_SHORT).show()
    }

    private fun updateModelSwitchLabel() {
        val current = yoloDetector?.loadedModelName ?: selectedModelName
        binding.btnSwitchModel.text = if (current == MODEL_YOLO26) {
            getString(R.string.switch_to_yolov8)
        } else {
            getString(R.string.switch_to_yolo26)
        }
    }

    private fun updateInfoText(info: String) {
        lastInfoLine = info
        val modelName = yoloDetector?.loadedModelName ?: selectedModelName
        binding.tvInfo.text = "Model: $modelName\n$info"
    }

    private fun buildDetectionSummary(detections: List<YoloV8Detector.Detection>, poseText: String): String {
        val labels = detections
            .groupingBy { it.className }
            .eachCount()
            .entries
            .sortedByDescending { it.value }
            .take(5)
            .joinToString(", ") { (name, count) ->
                if (count > 1) "$name x$count" else name
            }
            .ifBlank { "None" }
        return "Objects: ${detections.size} | Seen: $labels | Pose: $poseText"
    }

    private fun saveSnapshot() {
        val screenshot = buildSnapshotBitmap()
        cameraExecutor.execute {
            try {
                val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
                val fileName = "yolo_capture_$timestamp.png"
                val savedPath = saveBitmapToGallery(fileName, screenshot)
                runOnUiThread {
                    Toast.makeText(this, "Saved to gallery: $savedPath", Toast.LENGTH_LONG).show()
                }
            } catch (e: Exception) {
                Log.e("YoloDemoActivity", "Save snapshot failed", e)
                runOnUiThread {
                    Toast.makeText(this, "Save snapshot failed: ${e.message ?: "Unknown error"}", Toast.LENGTH_LONG).show()
                }
            } finally {
                if (!screenshot.isRecycled) {
                    screenshot.recycle()
                }
            }
        }
    }

    private fun buildSnapshotBitmap(): Bitmap {
        val cameraBitmap = binding.previewView.bitmap
            ?: throw IllegalStateException("Camera frame is not ready")
        val overlayBitmap = createBitmap(
            binding.overlayView.width.coerceAtLeast(1),
            binding.overlayView.height.coerceAtLeast(1)
        )
        val overlayCanvas = Canvas(overlayBitmap)
        binding.overlayView.draw(overlayCanvas)

        val composed = createBitmap(cameraBitmap.width, cameraBitmap.height)
        val canvas = Canvas(composed)
        canvas.drawBitmap(cameraBitmap, 0f, 0f, null)
        val overlayRect = Rect(0, 0, cameraBitmap.width, cameraBitmap.height)
        canvas.drawBitmap(overlayBitmap, null, overlayRect, null)

        if (!cameraBitmap.isRecycled) {
            cameraBitmap.recycle()
        }
        if (!overlayBitmap.isRecycled) {
            overlayBitmap.recycle()
        }
        return composed
    }

    private fun saveBitmapToGallery(fileName: String, bitmap: Bitmap): String {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            saveBitmapToMediaStore(fileName, bitmap)
        } else {
            saveBitmapToPublicPictures(fileName, bitmap)
        }
    }

    private fun saveBitmapToMediaStore(fileName: String, bitmap: Bitmap): String {
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/YOLODemo")
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val resolver = contentResolver
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            ?: throw IllegalStateException("MediaStore insert failed")
        try {
            resolver.openOutputStream(uri)?.use { stream ->
                writeBitmap(bitmap, stream)
            } ?: throw IllegalStateException("Open output stream failed")
            values.clear()
            values.put(MediaStore.Images.Media.IS_PENDING, 0)
            resolver.update(uri, values, null, null)
            return uri.toString()
        } catch (e: Exception) {
            resolver.delete(uri, null, null)
            throw e
        }
    }

    private fun saveBitmapToPublicPictures(fileName: String, bitmap: Bitmap): String {
        val picturesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
        val outputDir = File(picturesDir, "YOLODemo")
        if (!outputDir.exists()) {
            outputDir.mkdirs()
        }
        val output = File(outputDir, fileName)
        FileOutputStream(output).use { stream ->
            writeBitmap(bitmap, stream)
        }
        MediaScannerConnection.scanFile(
            this,
            arrayOf(output.absolutePath),
            arrayOf("image/png"),
            null
        )
        return output.absolutePath
    }

    private fun writeBitmap(bitmap: Bitmap, stream: OutputStream) {
        if (!bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)) {
            throw IllegalStateException("Bitmap compress failed")
        }
        stream.flush()
    }

    override fun onDestroy() {
        isShuttingDown = true
        super.onDestroy()
        cameraExecutor.shutdownNow()
        try {
            cameraExecutor.awaitTermination(500, TimeUnit.MILLISECONDS)
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
        }
        poseDetector.close()
        yoloDetector?.close()
        yoloDetector = null
    }

    private fun ImageProxy.toBitmapCompat(): Bitmap {
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 90, out)
        val bytes = out.toByteArray()

        return android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            ?: throw IllegalStateException("Decode frame failed")
    }

    private fun Bitmap.rotate(degrees: Float): Bitmap {
        if (degrees == 0f) return this
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    }
}
