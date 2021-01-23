from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2


def load_labels(path):
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}


def prepare_interpreter():
    # interpreter = Interpreter('models/mobilenet_v1_1.0_224_quant.tflite')
    interpreter = Interpreter('models/inception_v4_299_quant.tflite')
    interpreter.allocate_tensors()
    labels = load_labels('models/labels_mobilenet_quant_v1_224.txt')
    return interpreter, labels


def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    _, h, w, _ = interpreter.get_input_details()[0]['shape']
    ml_image = cv2.resize(image, (h, w))
    input_tensor[:, :] = ml_image


def classify_image(interpreter, image, top_k=1):
    """Returns a sorted array of classification results."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    # If the model is quantized (uint8 data), then dequantize the results
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)

    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]

