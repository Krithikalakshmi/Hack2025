//import android.content.Context
//import android.graphics.*
//import android.os.SystemClock
//import android.util.Log
//import androidx.compose.ui.unit.max
//import kotlinx.coroutines.flow.MutableStateFlow
//import kotlinx.coroutines.flow.StateFlow
//import org.tensorflow.lite.DataType
//import org.tensorflow.lite.Interpreter
//import org.tensorflow.lite.support.common.FileUtil
//import org.tensorflow.lite.support.common.ops.NormalizeOp
//import org.tensorflow.lite.support.image.ImageProcessor
//import org.tensorflow.lite.support.image.TensorImage
//import org.tensorflow.lite.support.image.ops.ResizeOp
//import org.tensorflow.lite.support.image.ops.Rot90Op
//import java.io.IOException
//import java.nio.ByteBuffer
//import java.nio.ByteOrderimport kotlin.math.max
//import kotlin.math.min
//
//// Data classes (as defined above)
////data class Detection(
////    val boundingBox: RectF, // Normalized coordinates (0-1) relative to the original image content
////    val className: String,
////    val confidence: Float
////)
//
////data class DetectionResult(
////    val detections: List<Detection>,
////    val inferenceTimeMs: Long,
////    val imageWidth: Int,
////    val imageHeight: Int
////)
//
//
//class YoloV8Detector(
//    private val context: Context,
//    private val numModelClasses: Int // Pass the number of classes your model was trained on
//) {
//
//    private var interpreter: Interpreter? = null
//    private var modelInputWidth: Int = 0
//    private var modelInputHeight: Int = 0
//    // numClasses will now be numModelClasses, passed in constructor
//
//
//    // Define class names - replace with your actual class names
//    private var classNames: List<String> = emptyList()
//
//
//    companion object {
//        private const val TAG = "YoloV8Detector"
//        private const val MODEL_FILE_NAME = "best_float32.tflite"
//        private const val CONFIDENCE_THRESHOLD = 0.3f
//        private const val IOU_THRESHOLD = 0.5f
//        private const val MAX_DETECTIONS_DISPLAYED = 10 // Max results after NMS
//
//        // Indices in the YOLOv8 output tensor's last dimension (4_bbox + num_classes)
//        const val BOX_X_CENTER_INDEX = 0
//        const val BOX_Y_CENTER_INDEX = 1
//        const val BOX_WIDTH_INDEX = 2
//        const val BOX_HEIGHT_INDEX = 3
//        const val CLASS_SCORES_START_INDEX = 4
//    }
//
//    // MutableStateFlow to emit detection results
//    private val _detectionResultFlow = MutableStateFlow<DetectionResult?>(null)
//    val detectionResultFlow: StateFlow<DetectionResult?> = _detectionResultFlow
//
//
//    init {
//        try {
//            val modelBuffer = FileUtil.loadMappedFile(context, MODEL_FILE_NAME)
//            val options = Interpreter.Options()
//            // options.addDelegate(GpuDelegate()) // Optional
//            interpreter = Interpreter(modelBuffer, options)
//
//            val inputTensor = interpreter!!.getInputTensor(0)
//            modelInputWidth = inputTensor.shape()[2] // Assuming shape [batch, height, width, channels]
//            modelInputHeight = inputTensor.shape()[1]
//
//            val outputTensor = interpreter!!.getOutputTensor(0)
//            val expectedOutputElements = 4 + numModelClasses
//            if (outputTensor.shape()[2] != expectedOutputElements) {
//                Log.e(TAG, "Model output elements (${outputTensor.shape()[2]}) " +
//                        "does not match expected (4 + numClasses = $expectedOutputElements). " +
//                        "Please check numModelClasses constructor argument.")
//                // Potentially throw an exception or disable detector
//            }
//
//            // Load labels (ensure this matches your model's labels.txt)
//            // This is a placeholder, adapt to your actual label loading
//            try {
//                // Assuming labels.txt is in assets and has one class name per line
//                context.assets.open("labels.txt").bufferedReader().useLines { lines ->
//                    classNames = lines.toList()
//                }
//                if (classNames.size != numModelClasses) {
//                    Log.w(TAG, "Number of loaded labels (${classNames.size}) does not match numModelClasses ($numModelClasses).")
//                }
//            } catch (e: IOException) {
//                Log.e(TAG, "Error loading labels.txt from assets", e)
//                // Fallback or error
//                classNames = List(numModelClasses) { "Class $it" }
//            }
//
//
//            Log.i(TAG, "Model loaded. Input: [$modelInputHeight, $modelInputWidth], Output: [${outputTensor.shape().joinToString()}], Classes: $numModelClasses, Labels: ${classNames.size}")
//
//        } catch (e: IOException) {
//            Log.e(TAG, "Error loading TFLite model: ${e.message}", e)
//        }
//    }
//
//    /**
//     * Main detection function. Processes the bitmap and emits DetectionResult via a Flow.
//     */
//    suspend fun detect(bitmap: Bitmap, rotationDegrees: Int) {
//        if (interpreter == null) {
//            Log.e(TAG, "Interpreter not initialized.")
//            _detectionResultFlow.emit(DetectionResult(emptyList(), 0, bitmap.width, bitmap.height))
//            return
//        }
//        if (classNames.isEmpty()) {
//            Log.e(TAG, "Class names not loaded.")
//            _detectionResultFlow.emit(DetectionResult(emptyList(), 0, bitmap.width, bitmap.height))
//            return
//        }
//
//
//        val originalImageWidth = bitmap.width
//        val originalImageHeight = bitmap.height
//        val startTime = SystemClock.uptimeMillis()
//
//        // 1. Image Preparation
//        val tensorImage = createTensorImageForYolo(bitmap, rotationDegrees, modelInputWidth, modelInputHeight)
//        val inputBuffer = tensorImage.buffer
//
//        // 2. Inference
//        val outputTensor = interpreter!!.getOutputTensor(0)
//        val outputShape = outputTensor.shape()
//        val numPredictions = outputShape[1]
//        val elementsPerPrediction = outputShape[2] // Should be 4 + numModelClasses
//
//        val outputBuffer = ByteBuffer.allocateDirect(1 * numPredictions * elementsPerPrediction * 4) // FLOAT32
//        outputBuffer.order(ByteOrder.nativeOrder())
//        outputBuffer.rewind()
//
//        interpreter!!.run(inputBuffer, outputBuffer)
//        outputBuffer.rewind()
//
//        // 3. Output Parsing (Intermediate Predictions)
//        val parsedPredictions = parseYOLOv8Output(outputBuffer, numPredictions, elementsPerPrediction)
//
//        // 4. Apply Non-Maximum Suppression (NMS)
//        val nmsResults = applyNMS(parsedPredictions, IOU_THRESHOLD)
//
//        val inferenceTimeMs = SystemClock.uptimeMillis() - startTime
//
//        // 5. Transform Bounding Boxes and Create Final Detections
//        val finalDetections = nmsResults.mapNotNull { pred ->
//            // The boundingBox in 'pred' is normalized to modelInput (e.g., 640x640)
//            // Transform it to be normalized to the original image's content area
//            val transformedBox = transformToOriginalImageCoordinates(
//                modelRelativeBox = pred.boundingBox, // Normalized to model input
//                originalImageWidth = originalImageWidth,
//                originalImageHeight = originalImageHeight,
//                modelInputWidth = modelInputWidth,
//                modelInputHeight = modelInputHeight
//            )
//
//            if (transformedBox.width() > 0 && transformedBox.height() > 0) {
//                Detection(
//                    boundingBox = transformedBox, // Now normalized to original image content
//                    className = pred.className,
//                    confidence = pred.confidence
//                )
//            } else {
//                null // Skip if the transformed box is invalid
//            }
//        }
//
//        _detectionResultFlow.emit(
//            DetectionResult(
//                detections = finalDetections.take(MAX_DETECTIONS_DISPLAYED),
//                inferenceTimeMs = inferenceTimeMs,
//                imageWidth = originalImageWidth,
//                imageHeight = originalImageHeight
//            )
//        )
//    }
//
//
//    private fun createTensorImageForYolo(
//        bitmap: Bitmap,
//        rotationDegrees: Int,
//        targetWidth: Int,
//        targetHeight: Int
//    ): TensorImage {
//        val imageProcessorBuilder = ImageProcessor.Builder()
//        if (rotationDegrees != 0) {
//            imageProcessorBuilder.add(Rot90Op(-rotationDegrees / 90))
//        }
//        imageProcessorBuilder
//            .add(ResizeOp(targetHeight, targetWidth, ResizeOp.ResizeMethod.BILINEAR))
//            .add(NormalizeOp(0.0f, 255.0f)) // Common for YOLO: [0,1]
//
//        val imageProcessor = imageProcessorBuilder.build()
//        var processedTensorImage = TensorImage(DataType.FLOAT32)
//        // Important: Load the original bitmap first, then process.
//        // The processor handles the rotation and then resizing.
//        processedTensorImage.load(bitmap)
//        return imageProcessor.process(processedTensorImage)
//    }
//
//    /**
//     * Parses the raw YOLOv8 output buffer.
//     */
//    private fun parseYOLOv8Output(
//        outputBuffer: ByteBuffer, // Changed to ByteBuffer
//        numPredictions: Int,
//        elementsPerPrediction: Int
//    ): List<IntermediatePrediction> {
//        val predictions = mutableListOf<IntermediatePrediction>()
//
//        for (i in 0 until numPredictions) {
//            val predictionData = FloatArray(elementsPerPrediction)
//            for (j in 0 until elementsPerPrediction) {
//                predictionData[j] = outputBuffer.float // Read directly from ByteBuffer
//            }
//
//            val xCenterNorm = predictionData[BOX_X_CENTER_INDEX]
//            val yCenterNorm = predictionData[BOX_Y_CENTER_INDEX]
//            val widthNorm = predictionData[BOX_WIDTH_INDEX]
//            val heightNorm = predictionData[BOX_HEIGHT_INDEX]
//
//            var maxScore = 0.0f
//            var bestClassId = -1
//            for (classIdx in 0 until numModelClasses) {
//                val score = predictionData[CLASS_SCORES_START_INDEX + classIdx]
//                if (score > maxScore) {
//                    maxScore = score
//                    bestClassId = classIdx
//                }
//            }
//
//            if (maxScore >= CONFIDENCE_THRESHOLD) {
//                val xMin = xCenterNorm - widthNorm / 2f
//                val yMin = yCenterNorm - heightNorm / 2f
//                val xMax = xCenterNorm + widthNorm / 2f
//                val yMax = yCenterNorm + heightNorm / 2f
//
//                predictions.add(
//                    IntermediatePrediction(
//                        boundingBox = RectF(
//                            xMin.coerceIn(0f, 1f),
//                            yMin.coerceIn(0f, 1f),
//                            xMax.coerceIn(0f, 1f),
//                            yMax.coerceIn(0f, 1f)
//                        ), // Still normalized to model input
//                        confidence = maxScore,
//                        classId = bestClassId,
//                        className = classNames.getOrElse(bestClassId) { "Class $bestClassId" }
//                    )
//                )
//            }
//        }
//        Log.d(TAG, "Parsed ${predictions.size} intermediate predictions with score >= threshold.")
//        return predictions
//    }
//
//    /**
//     * Intermediate prediction data class (before NMS and coordinate transformation).
//     * BoundingBox is normalized relative to model input (e.g., 640x640).
//     */
//    private data class IntermediatePrediction(
//        val boundingBox: RectF,
//        val confidence: Float,
//        val classId: Int,
//        val className: String
//    )
//
//    private fun applyNMS(
//        predictions: List<IntermediatePrediction>,
//        iouThreshold: Float
//    ): List<IntermediatePrediction> {
//        val sortedPredictions = predictions.sortedByDescending { it.confidence }
//        val selectedPredictions = mutableListOf<IntermediatePrediction>()
//        val active = BooleanArray(sortedPredictions.size) { true }
//
//        for (i in sortedPredictions.indices) {
//            if (active[i]) {
//                val p1 = sortedPredictions[i]
//                selectedPredictions.add(p1)
//                // Limit by MAX_DETECTIONS_DISPLAYED early if desired, or after NMS completely
//                // if (selectedPredictions.size >= MAX_DETECTIONS_DISPLAYED) break
//
//                for (j in (i + 1) until sortedPredictions.size) {
//                    if (active[j]) {
//                        val p2 = sortedPredictions[j]
//                        if (calculateIoU(p1.boundingBox, p2.boundingBox) > iouThreshold) {
//                            active[j] = false
//                        }
//                    }
//                }
//            }
//        }
//        Log.d(TAG, "NMS selected ${selectedPredictions.size} predictions.")
//        return selectedPredictions // Could also apply MAX_DETECTIONS_DISPLAYED limit here
//    }
//
//    private fun calculateIoU(box1: RectF, box2: RectF): Float {
//        // Standard IoU calculation (same as before)
//        val xA = max(box1.left, box2.left)
//        val yA = max(box1.top, box2.top)
//        val xB = min(box1.right, box2.right)
//        val yB = min(box1.bottom, box2.bottom)
//        val intersectionArea = max(0f, xB - xA) * max(0f, yB - yA)
//        if (intersectionArea == 0f) return 0f
//        val box1Area = (box1.right - box1.left) * (box1.bottom - box1.top)
//        val box2Area = (box2.right - box2.left) * (box2.bottom - box2.top)
//        return intersectionArea / (box1Area + box2Area - intersectionArea)
//    }
//
//    /**
//     * Transforms a bounding box from model input space (normalized, possibly padded)
//     * to original image content space (normalized 0-1).
//     */
//    private fun transformToOriginalImageCoordinates(
//        modelRelativeBox: RectF, // Normalized 0-1 box relative to modelInputWidth/Height
//        originalImageWidth: Int,
//        originalImageHeight: Int,
//        modelInputWidth: Int,
//        modelInputHeight: Int
//    ): RectF {
//        val originalAspectRatio = originalImageWidth.toFloat() / originalImageHeight.toFloat()
//        val modelInputAspectRatio = modelInputWidth.toFloat() / modelInputHeight.toFloat()
//
//        var scaledContentWidth: Float
//        var scaledContentHeight: Float
//        var paddingLeft = 0f
//        var paddingTop = 0f
//
//        // Calculate how the original image was scaled to fit into modelInput dimensions
//        // This logic assumes letterboxing/pillarboxing (maintaining aspect ratio)
//        if (originalAspectRatio > modelInputAspectRatio) {
//            // Original image is wider than model input aspect ratio (letterboxed vertically)
//            scaledContentWidth = modelInputWidth.toFloat()
//            scaledContentHeight = modelInputWidth / originalAspectRatio
//            paddingTop = (modelInputHeight - scaledContentHeight) / 2f
//        } else {
//            // Original image is taller or same aspect ratio (pillarboxed horizontally)
//            scaledContentHeight = modelInputHeight.toFloat()
//            scaledContentWidth = modelInputHeight * originalAspectRatio
//            paddingLeft = (modelInputWidth - scaledContentWidth) / 2f
//        }
//
//        // Bounding box coordinates from model are relative to modelInputWidth/Height
//        // Denormalize them first to pixel values within the modelInput space
//        val boxLeftModelPx = modelRelativeBox.left * modelInputWidth
//        val boxTopModelPx = modelRelativeBox.top * modelInputHeight
//        val boxRightModelPx = modelRelativeBox.right * modelInputWidth
//        val boxBottomModelPx = modelRelativeBox.bottom * modelInputHeight
//
//        // Remove padding to get coordinates relative to the scaled original content
//        val boxOnScaledContentLeft = (boxLeftModelPx - paddingLeft)
//        val boxOnScaledContentTop = (boxTopModelPx - paddingTop)
//        val boxOnScaledContentRight = (boxRightModelPx - paddingLeft)
//        val boxOnScaledContentBottom = (boxBottomModelPx - paddingTop)
//
//        // Normalize these coordinates relative to the scaledContent dimensions
//        // to get final 0-1 coordinates relative to the original image content area.
//        val finalLeft = (boxOnScaledContentLeft / scaledContentWidth).coerceIn(0f, 1f)
//        val finalTop = (boxOnScaledContentTop / scaledContentHeight).coerceIn(0f, 1f)
//        val finalRight = (boxOnScaledContentRight / scaledContentWidth).coerceIn(0f, 1f)
//        val finalBottom = (boxOnScaledContentBottom / scaledContentHeight).coerceIn(0f, 1f)
//
//        return RectF(finalLeft, finalTop, finalRight, finalBottom)
//    }
//
//    fun close() {
//        interpreter?.close()
//        interpreter = null
//    }
//}
