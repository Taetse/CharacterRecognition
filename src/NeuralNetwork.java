import org.omg.PortableInterceptor.SYSTEM_EXCEPTION;

import java.util.ArrayList;

public class NeuralNetwork {
    private ArrayList<InputVector> inputVectors;
    private int hiddenUnitCount;
    private int outputUnitCount;
    private int inputUnitCount;
    private double hiddenLayerOutputs[];
    private double outputLayerOutputs[];
    private int actualOutput[];
    private double hiddenLayerWeights[][];
    private double outputLayerWeights[][];
    private double learningRate;
    private double momentum;
    private double trainingError;
    private int epochCount;
    private int maxEpoch;
    private double desiredTrainingAccuracy;


    NeuralNetwork(int inputUnitCount, int hiddenUnitCount, int outputUniCount) {
        this.hiddenUnitCount = hiddenUnitCount;
        this.outputUnitCount = outputUniCount;
        this.inputUnitCount = inputUnitCount;

        hiddenLayerOutputs = new double[hiddenUnitCount];
        outputLayerOutputs = new double[outputUniCount];
        actualOutput = new int[outputUniCount];

        hiddenLayerWeights = new double[hiddenUnitCount][inputUnitCount + 1]; //hidden units x input units + bias unit
        outputLayerWeights = new double[outputUnitCount][hiddenUnitCount + 1]; //hidden units x input units + bias unit

        for (int y = 0; y < hiddenLayerWeights.length; y++) {
            for (int x = 0; x < hiddenLayerWeights[y].length; x++) {
                hiddenLayerWeights[y][x] = getRandomValue(-(1/inputUnitCount), (1/inputUnitCount));
            }
        }

        for (int y = 0; y < outputLayerWeights.length; y++) {
            for (int x = 0; x < outputLayerWeights[y].length; x++) {
                hiddenLayerWeights[y][x] = getRandomValue(-(1/hiddenUnitCount), (1/hiddenUnitCount));
            }
        }
    }

    public void initControlVariables(double learningRate, double momentum, double trainingError, int epochCount, int maxEpoch, double desiredTrainingAccuracy) {
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.trainingError = trainingError;
        this.epochCount = epochCount;
        this.maxEpoch = maxEpoch;
        this.desiredTrainingAccuracy = desiredTrainingAccuracy;
    }

    public void setInputVectors(ArrayList<InputVector> inputVectors) {
        this.inputVectors = inputVectors;
    }

    public void train() {
        boolean criteriaMet = false;
        while (!criteriaMet) {
            trainingError = 0;
            epochCount++;

            for (InputVector inputVector : inputVectors) {
                boolean missClassification = false;
                //calculate hidden layer outputs
                double inputs[] = augmentInputArray(inputVector.vector);
                for (int j = 0; j < hiddenUnitCount; j++) {
                    hiddenLayerOutputs[j] = activation(NetYJ(j, inputs));
                }

                //calculate output layer outputs
                double hiddenOutputs[] = augmentInputArray(hiddenLayerOutputs);
                for (int k = 0; k < outputUnitCount; k++) {
                    outputLayerOutputs[k] = activation(NetOK(k, hiddenOutputs));
                    if (outputLayerOutputs[k] >= 0.7)
                        actualOutput[k] = 1;
                    else if (outputLayerOutputs[k] <= 0.3)
                        actualOutput[k] = 0;
                    else
                        missClassification = true;
                }

                //training error
                missClassification = missClassification || missClassification(actualOutput, inputVector.target);
                if (missClassification)
                    trainingError++;

                //Calculate the error signal for each output
                double outputErrorSignal[] = getOutputErrorSignal(inputVector.target, outputLayerOutputs);

                //Calculate the new weight values for the hidden-to-output weights

            }
        }
    }

    private double[] getOutputErrorSignal(int target[], double output[]) {
        double errorSignal[] = new double[target.length];
        for (int a = 0; a < target.length; a++) {
            errorSignal[a] = -(target[a] - output[a]) * (1 - output[a]) * output[a];
        }
        return errorSignal;
    }

    private boolean missClassification(int output[], int target[]) {
        for (int a = 0; a < output.length; a++)
            if (output[a] == target[a])
                return true;
        return false;
    }

    private double NetYJ(int j, double inputs[]) {
        double net = 0;
        for (int x = 0; x < inputUnitCount + 1; x++) {
            net += hiddenLayerWeights[j][x] * inputs[x];
        }
        return net;
    }

    private double NetOK(int k, double inputs[]) {
        double net = 0;
        for (int x = 0; x < hiddenUnitCount + 1; x++) {
            net += outputLayerWeights[k][x] * inputs[x];
        }
        return net;
    }

    private double[] augmentInputArray(double inputs[]) {
        double augmented[] = new double[inputs.length + 1];
        System.arraycopy(inputs, 0, augmented, 0, inputs.length);
        augmented[inputs.length] = -1;
        return augmented;
    }

    private double activation(double net) {
        return 1/(1 + Math.pow(Math.E, -1*(net)));
    }

    private double getRandomValue(double low, double top) {
        double value = Math.random() * (top - low);
        return value + low;
    }

}
