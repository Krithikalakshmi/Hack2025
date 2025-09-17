/*
 * Copyright 2024 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.aiedge.examples.object_detection.objectdetector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageProxy
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

class ObjectDetectorHelper(
    val context: Context,
    var delegate: Delegate = Delegate.CPU,
    var model: Model = MODEL_DEFAULT,
) {

    private var interpreter: Interpreter? = null
    private var labels: List<String> = listOf("closed valve", "opened valve")
    private var modelInputWidth: Int = 640
    private var modelInputHeight: Int = 640
    private var numModelClasses: Int = 2

    private val _detectionResult = MutableSharedFlow<DetectionResult>()
    val detectionResult: SharedFlow<DetectionResult> = _detectionResult

    private val _error = MutableSharedFlow<Throwable>()
    val error: SharedFlow<Throwable> = _error

    private fun convertToARGB8888(image: Bitmap): Bitmap {
        if (image.config == Bitmap.Config.ARGB_8888) {
            return image
        }
        Log.d(TAG, "Converting bitmap from ${image.config} to ARGB_8888")
        return image.copy(Bitmap.Config.ARGB_8888, true)
    }

    suspend fun setupObjectDetector() {
        try {
            val modelBuffer = FileUtil.loadMappedFile(context, model.fileName)
            val interpreterOptions = Interpreter.Options().apply {
                numThreads = Runtime.getRuntime().availableProcessors()
                useNNAPI = delegate == Delegate.NNAPI
            }
            interpreter = Interpreter(modelBuffer, interpreterOptions)

            val inputTensor = interpreter!!.getInputTensor(0)
            modelInputWidth = inputTensor.shape()[2] 
            modelInputHeight = inputTensor.shape()[1]

            val outputTensor = interpreter!!.getOutputTensor(0)
            val outputTensorShape = outputTensor.shape() 
            Log.i(TAG, "Setup: Output tensor shape: ${outputTensorShape.joinToString(", ")}") // Expected [1, 6, 8400]

            // For output [1, NumAttributes, NumProposals] where NumAttributes = NumClasses + 4 box_coords
            // e.g., [1, 6, 8400] for 2 classes.
            if (outputTensorShape.size == 3 && outputTensorShape[0] == 1) {
                val numAttributes = outputTensorShape[1]
                if (numAttributes < BOX_COORDINATES_SIZE) {
                    Log.e(TAG, "SETUP ERROR: Output tensor has too few attributes (${numAttributes}) to contain box coordinates.")
                    _error.emit(IllegalStateException("Output tensor attributes (${numAttributes}) < ${BOX_COORDINATES_SIZE}"))
                    return
                }
                numModelClasses = numAttributes - BOX_COORDINATES_SIZE
                Log.i(TAG, "Setup: numModelClasses derived: $numModelClasses (from ${numAttributes} attributes - ${BOX_COORDINATES_SIZE} box coords)")
            } else {
                Log.e(TAG, "SETUP ERROR: Unexpected output tensor shape. Expected 3D [1, NumAttributes, NumProposals]. Got ${outputTensorShape.joinToString(", ")}")
                _error.emit(IllegalStateException("Unexpected output tensor shape: ${outputTensorShape.joinToString(", ")}"))
                return
            }
            
            if (numModelClasses != labels.size) {
                Log.e(TAG, "MODEL CONFIG ERROR: Derived numModelClasses ($numModelClasses) != labels.size (${labels.size}). Check model and labels list.")
                _error.emit(IllegalStateException("Model class count ($numModelClasses) != labels list size (${labels.size})."))
                return
            }

            Log.i(TAG, "Setup Successful - Model: ${model.fileName}, Input: [${modelInputHeight}x$modelInputWidth], Output Classes (model): $numModelClasses, Labels (code): ${labels.joinToString()}")
        } catch (e: Exception) {
            _error.emit(e)
            Log.e(TAG, "TFLite setup failed: " + e.message, e)
        }
    }

    suspend fun detect(imageProxy: ImageProxy) {
        val originalBitmap = imageProxy.toBitmap()
        if (originalBitmap == null) {
            Log.e(TAG, "Failed to convert ImageProxy to Bitmap.")
            _detectionResult.emit(DetectionResult(emptyList(), 0, 0, 0))
            return
        }
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        detect(originalBitmap, rotationDegrees)
    }

    suspend fun detect(tempBitmap: Bitmap, rotationDegrees: Int) {
        val bitmap = convertToARGB8888(tempBitmap)
        if (interpreter == null || numModelClasses < 0) { // numModelClasses can be 0 if model has only boxes
            Log.e(TAG, "Interpreter not ready or numModelClasses not set properly ($numModelClasses). Skipping.")
            _detectionResult.emit(DetectionResult(emptyList(), 0, bitmap.width, bitmap.height))
            return
        }

        val originalImageWidth = bitmap.width
        val originalImageHeight = bitmap.height
        val startTime = SystemClock.uptimeMillis()

        val tensorImage = createTensorImageForYolo(bitmap, rotationDegrees, modelInputWidth, modelInputHeight)
        val inputBuffer = tensorImage.buffer

        val outputTensorShape = interpreter!!.getOutputTensor(0).shape()
        // val numAttributes = outputTensorShape[1] // Should be 6 (4 box_coords + 2 class_scores)
        val numProposals = outputTensorShape[2] // Should be 8400
        val outputElementCount = 1 * outputTensorShape[1] * numProposals // 1 * 6 * 8400 = 50400

        val outputBuffer = ByteBuffer.allocateDirect(outputElementCount * 4) // 4 bytes per float
        outputBuffer.order(ByteOrder.nativeOrder())
        val outputMap = mapOf(0 to outputBuffer)

        interpreter!!.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)
        outputBuffer.rewind()
        val outputArray = FloatArray(outputElementCount)
        outputBuffer.asFloatBuffer().get(outputArray)

        val parsedPredictions = parseYOLOv8Output(outputArray, numProposals, numModelClasses)
        val nmsResults = applyNMS(parsedPredictions, IOU_THRESHOLD)
        val inferenceTimeMs = SystemClock.uptimeMillis() - startTime

        val finalDetections = nmsResults.mapNotNull { pred ->
            val transformedBox = convertModelBoxToOriginalBitmapCoordinates(
                modelRelativeBox = pred.boundingBox,
                originalBitmapWidth = originalImageWidth,
                originalBitmapHeight = originalImageHeight,
                paramModelInputWidth = this.modelInputWidth,
                paramModelInputHeight = this.modelInputHeight,
                imageRotationDegrees = rotationDegrees
            )
            if (transformedBox.width() > 0 && transformedBox.height() > 0) {
                Detection(boundingBox = transformedBox, label = pred.className, score = pred.confidence)
            } else {
                Log.w(TAG, "Skipping detection with invalid transformed box: Label=${pred.className}, OrigBox=${pred.boundingBox}, TransformedBox=$transformedBox")
                null
            }
        }
        
        finalDetections.take(MAX_DETECTIONS_DISPLAYED).forEachIndexed { index, detection ->
            Log.d(TAG, "Final Detection $index: Label=${detection.label}, Score=${String.format("%.2f", detection.score)}, BoxNorm=[L:${String.format("%.3f", detection.boundingBox.left)}, T:${String.format("%.3f", detection.boundingBox.top)}, R:${String.format("%.3f", detection.boundingBox.right)}, B:${String.format("%.3f", detection.boundingBox.bottom)}]")
        }
        Log.i(TAG, "Detection took $inferenceTimeMs ms. Raw Proposals: $numProposals, Parsed (Pre-NMS): ${parsedPredictions.size}, NMS: ${nmsResults.size}, Final (Displayed): ${finalDetections.size}")

        _detectionResult.emit(DetectionResult(finalDetections.take(MAX_DETECTIONS_DISPLAYED), inferenceTimeMs, originalImageWidth, originalImageHeight))
    }

    private fun createTensorImageForYolo(bitmap: Bitmap, rotationDegrees: Int, targetWidth: Int, targetHeight: Int): TensorImage {
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeWithCropOrPadOp(targetHeight, targetWidth))
            .add(Rot90Op(rotationDegrees / 90))
            .add(NormalizeOp(0.0f, 255.0f))
            .build()
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        return imageProcessor.process(tensorImage)
    }
    
    // Corrected parsing based on Python code: output is (1, NumAttributes, NumProposals)
    // Python transposes to (NumProposals, NumAttributes) then processes.
    // Flat Kotlin array index = attribute_index * numProposals + proposal_index
    private fun parseYOLOv8Output(
        outputArray: FloatArray, // Flattened from (1, NumAttributes, NumProposals) e.g. (1,6,8400)
        numProposals: Int,       // e.g., 8400
        numClassesFromModel: Int // e.g., 2
    ): List<IntermediatePrediction> {
        val predictions = mutableListOf<IntermediatePrediction>()
        
        val numAttributes = BOX_COORDINATES_SIZE + numClassesFromModel // Should be 6 for your model
        if (outputArray.size != numAttributes * numProposals) {
            Log.e(TAG, "PARSE ERROR: outputArray size (${outputArray.size}) != numAttributes (${numAttributes}) * numProposals (${numProposals}). Incorrect tensor flattening or shape assumption.")
            return emptyList()
        }

        Log.d(TAG, "Starting YOLOv8 output parsing (Python-transposed logic). numProposals=$numProposals, numAttributesExpected=$numAttributes, numClassesFromModel=$numClassesFromModel")

        for (proposalIndex in 0 until numProposals) {
            // Accessing data based on attribute_index * numProposals + proposal_index
            val xc = outputArray[XC_ATTRIBUTE_INDEX * numProposals + proposalIndex]
            val yc = outputArray[YC_ATTRIBUTE_INDEX * numProposals + proposalIndex]
            val w = outputArray[W_ATTRIBUTE_INDEX * numProposals + proposalIndex]
            val h = outputArray[H_ATTRIBUTE_INDEX * numProposals + proposalIndex]

            var maxScore = -1f
            var bestClassId = -1
            val currentProposalScores = mutableListOf<Float>()

            for (classIdx in 0 until numClassesFromModel) {
                val score = outputArray[(CLASS_SCORES_START_ATTRIBUTE_INDEX + classIdx) * numProposals + proposalIndex]
                currentProposalScores.add(score)
                if (score > maxScore) {
                    maxScore = score
                    bestClassId = classIdx
                }
            }
            
            // Log first 3 proposals' raw data and parsed values for debugging
            if (proposalIndex < 3) {
                val rawBoxData = "xc=${String.format("%.4f",xc)}, yc=${String.format("%.4f",yc)}, w=${String.format("%.4f",w)}, h=${String.format("%.4f",h)}"
                val rawScoreData = (0 until numClassesFromModel).map {
                    outputArray[(CLASS_SCORES_START_ATTRIBUTE_INDEX + it) * numProposals + proposalIndex]
                }.joinToString(", ") { String.format("%.4f", it) }
                Log.d(TAG, "Raw Data Proposal $proposalIndex: Box=[$rawBoxData], Scores=[$rawScoreData]")
                Log.d(TAG, "Parsed Proposal $proposalIndex: xc=${String.format("%.3f",xc)}, yc=${String.format("%.3f",yc)}, w=${String.format("%.3f",w)}, h=${String.format("%.3f",h)}, scores=[${currentProposalScores.joinToString { String.format("%.3f", it) }}], bestClassId=$bestClassId, maxScore=${String.format("%.3f",maxScore)}")
            }
            
            if (maxScore >= CONFIDENCE_THRESHOLD) {
                val xMin = (xc - w / 2f).coerceIn(0f, 1f)
                val yMin = (yc - h / 2f).coerceIn(0f, 1f)
                val xMax = (xc + w / 2f).coerceIn(0f, 1f)
                val yMax = (yc + h / 2f).coerceIn(0f, 1f)

                if (xMin >= xMax || yMin >= yMax) {
                    continue
                }
                predictions.add(IntermediatePrediction(RectF(xMin, yMin, xMax, yMax), maxScore, bestClassId, labels.getOrElse(bestClassId) { "Class $bestClassId" }))
            }
        }
        Log.d(TAG, "Parsed ${predictions.size} intermediate predictions with score >= $CONFIDENCE_THRESHOLD.")
        return predictions
    }

    private fun applyNMS(predictions: List<IntermediatePrediction>, iouThreshold: Float): List<IntermediatePrediction> {
        val sortedPredictions = predictions.sortedByDescending { it.confidence }
        val selectedPredictions = mutableListOf<IntermediatePrediction>()
        val active = BooleanArray(sortedPredictions.size) { true }

        for (i in sortedPredictions.indices) {
            if (active[i]) {
                val p1 = sortedPredictions[i]
                selectedPredictions.add(p1)
                if (selectedPredictions.size >= MAX_DETECTIONS_DISPLAYED_NMS_INTERNAL_LIMIT) break
                for (j in (i + 1) until sortedPredictions.size) {
                    if (active[j]) {
                        val p2 = sortedPredictions[j]
                        if (calculateIoU(p1.boundingBox, p2.boundingBox) > iouThreshold) {
                            active[j] = false
                        }
                    }
                }
            }
        }
        Log.d(TAG, "NMS input: ${predictions.size}, output: ${selectedPredictions.size}")
        return selectedPredictions
    }

    private fun calculateIoU(box1: RectF, box2: RectF): Float {
        val xA = max(box1.left, box2.left)
        val yA = max(box1.top, box2.top)
        val xB = min(box1.right, box2.right)
        val yB = min(box1.bottom, box2.bottom)
        val intersectionArea = max(0f, xB - xA) * max(0f, yB - yA)
        if (intersectionArea == 0f) return 0f
        val box1Area = (box1.right - box1.left) * (box1.bottom - box1.top)
        val box2Area = (box2.right - box2.left) * (box2.bottom - box2.top)
        val iou = intersectionArea / (box1Area + box2Area - intersectionArea)
        return if (iou.isNaN()) 0f else iou
    }

    private fun convertModelBoxToOriginalBitmapCoordinates(modelRelativeBox: RectF, originalBitmapWidth: Int, originalBitmapHeight: Int, paramModelInputWidth: Int, paramModelInputHeight: Int, imageRotationDegrees: Int): RectF {
        val preRotationPaddedCanvasWidth: Float
        val preRotationPaddedCanvasHeight: Float
        if (imageRotationDegrees == 90 || imageRotationDegrees == 270) {
            preRotationPaddedCanvasWidth = paramModelInputHeight.toFloat()
            preRotationPaddedCanvasHeight = paramModelInputWidth.toFloat()
        } else {
            preRotationPaddedCanvasWidth = paramModelInputWidth.toFloat()
            preRotationPaddedCanvasHeight = paramModelInputHeight.toFloat()
        }
        val unrotatedBox = when (imageRotationDegrees) {
            90 -> RectF(modelRelativeBox.top, 1.0f - modelRelativeBox.right, modelRelativeBox.bottom, 1.0f - modelRelativeBox.left)
            180 -> RectF(1.0f - modelRelativeBox.right, 1.0f - modelRelativeBox.bottom, 1.0f - modelRelativeBox.left, 1.0f - modelRelativeBox.top)
            270 -> RectF(1.0f - modelRelativeBox.bottom, modelRelativeBox.left, 1.0f - modelRelativeBox.top, modelRelativeBox.right)
            else -> modelRelativeBox
        }
        val originalAspectRatio = originalBitmapWidth.toFloat() / originalBitmapHeight.toFloat()
        val paddedCanvasAspectRatio = preRotationPaddedCanvasWidth / preRotationPaddedCanvasHeight
        var paddingNormX = 0f
        var paddingNormY = 0f
        val contentWidthOnCanvasNorm: Float
        val contentHeightOnCanvasNorm: Float
        if (originalAspectRatio > paddedCanvasAspectRatio) {
            val scale = preRotationPaddedCanvasWidth / originalBitmapWidth.toFloat()
            contentWidthOnCanvasNorm = 1.0f
            contentHeightOnCanvasNorm = (originalBitmapHeight.toFloat() * scale) / preRotationPaddedCanvasHeight
            paddingNormY = (1.0f - contentHeightOnCanvasNorm) / 2.0f
        } else {
            val scale = preRotationPaddedCanvasHeight / originalBitmapHeight.toFloat()
            contentHeightOnCanvasNorm = 1.0f
            contentWidthOnCanvasNorm = (originalBitmapWidth.toFloat() * scale) / preRotationPaddedCanvasWidth
            paddingNormX = (1.0f - contentWidthOnCanvasNorm) / 2.0f
        }
        val finalLeft = (unrotatedBox.left - paddingNormX) / contentWidthOnCanvasNorm
        val finalTop = (unrotatedBox.top - paddingNormY) / contentHeightOnCanvasNorm
        val finalRight = (unrotatedBox.right - paddingNormX) / contentWidthOnCanvasNorm
        val finalBottom = (unrotatedBox.bottom - paddingNormY) / contentHeightOnCanvasNorm
        return RectF(finalLeft.coerceIn(0f, 1f), finalTop.coerceIn(0f, 1f), finalRight.coerceIn(0f, 1f), finalBottom.coerceIn(0f, 1f))
    }

    companion object {
        val MODEL_DEFAULT = Model.CustomModel
        private const val CONFIDENCE_THRESHOLD = 0.3f
        private const val IOU_THRESHOLD = 0.45f
        private const val MAX_DETECTIONS_DISPLAYED = 10
        private const val MAX_DETECTIONS_DISPLAYED_NMS_INTERNAL_LIMIT = 50
        
        // Attribute indices if the original TFLite output is (1, NumAttributes, NumProposals)
        // and the Python code effectively transposes it before processing.
        const val XC_ATTRIBUTE_INDEX = 0
        const val YC_ATTRIBUTE_INDEX = 1
        const val W_ATTRIBUTE_INDEX = 2
        const val H_ATTRIBUTE_INDEX = 3
        const val BOX_COORDINATES_SIZE = 4 // Number of box coordinates
        const val CLASS_SCORES_START_ATTRIBUTE_INDEX = 4 // Scores for class 0 start at attribute index 4

        const val TAG = "ObjectDetectorHelper"
    }
    enum class Model(val fileName: String) { CustomModel("metadata1.tflite") }
    enum class Delegate(val value: Int) { CPU(0), NNAPI(1) }
    data class DetectionResult(val detections: List<Detection>, val inferenceTime: Long, val inputImageHeight: Int, val inputImageWidth: Int)
    data class Detection(val label: String, val boundingBox: RectF, val score: Float)
    private data class IntermediatePrediction(val boundingBox: RectF, val confidence: Float, val classId: Int, val className: String)
}
