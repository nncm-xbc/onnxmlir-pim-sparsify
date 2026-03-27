# tester.py

###########################################################################################################################################################

import os
import sys
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

from Compiler.compiler import Program
from Compiler.network import Network

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def main():

    args         = sys.argv
    input_network = os.path.abspath(args[1])
    x_test_file   = os.path.abspath(args[2])
    y_test_file   = os.path.abspath(args[3])

    x_test = np.genfromtxt(x_test_file, delimiter = ',', max_rows = 1000)
    y_test = np.genfromtxt(y_test_file, delimiter = ',', max_rows = 1000)

    x_single = x_test[0]
    y_single = y_test[0]

    if input_network.endswith('.pt'):

        print("Loading PyTorch model")
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        y_test = y_test.argmax(dim=1)

        model = torch.load(input_network, weights_only=False)
        model.eval()
        out = model(x_test)

        _, predicted = torch.max(out.data, 1)
        correct = (predicted == y_test).sum().item()
        total = y_test.size(0)
        accuracy = 100 * correct / total
        print(f'Test accuracy: {accuracy:.2f}%')

    elif input_network.endswith('.onnx'):
        onnx_model = onnx.load(input_network)
        onnx.checker.check_model(onnx_model)

        ort_session = ort.InferenceSession(input_network)
        input_name = ort_session.get_inputs()[0].name

        # onnx_inputs = np.random.randn(1, x_test.shape[1]).astype(np.float32)
        results = []
        for i in range(x_test.shape[0]):
            onnx_inputs = x_test[i:i+1].astype(np.float32)

            onnxruntime_inputs = {input_name: onnx_inputs}
            onnxruntime_outputs = ort_session.run(None, onnxruntime_inputs)

            results.append(onnxruntime_outputs)

        predicted = np.stack(results)
        predicted = predicted.reshape(predicted.shape[0], -1)

        predicted_classes = np.argmax(predicted, axis=1)
        true_classes = np.argmax(y_test, axis=1)

        correct = (predicted_classes == true_classes).sum()
        total = len(true_classes)
        accuracy = 100 * correct / total
        print(f'Test accuracy: {accuracy:.2f}%')


    else:

        #TODO: emulate arm architecture and run the assembly code of the network
        print("Loading OG network")
        results = []
        model = Program(input_network)
        for i in range(x_test.shape[0]):
            out = model.run(x_test[i])
            results.append(out)
        predicted = np.stack(results)

        predicted_classes = np.argmax(predicted, axis=1)
        true_classes = np.argmax(y_test, axis=1)

        correct = (predicted_classes == true_classes).sum()
        total = len(true_classes)
        accuracy = 100 * correct / total
        print(f'Test accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    main()
