import java.util.ArrayList;

public class NeuralNetwork {
    private ArrayList<InputVector> inputVectors;
    private int hiddenUnitCount;
    private int outputUnitCount;
    private int inputUnitCount;
    private double inputLayerOutputs[];
    private double hiddenLayerOutputs[];
    private double outputLayerOutputs[];
    private double hiddenLayerWeights[][];
    private double outputLayerWeights[][];


    NeuralNetwork(ArrayList<InputVector> inputVectors, int inputUnitCount, int hiddenUnitCount, int outputUniCount) {
        this.inputVectors = inputVectors;
        this.hiddenUnitCount = hiddenUnitCount;
        this.outputUnitCount = outputUniCount;
        this.inputUnitCount = inputUnitCount;

        inputLayerOutputs = new double[inputUnitCount + 1]; //input units + bias unit
        hiddenLayerOutputs = new double[hiddenUnitCount + 1]; //input units + bias unit
        outputLayerOutputs = new double[outputUniCount + 1]; //input units + bias unit

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

    private double getRandomValue(double low, double top) {
        double value = Math.random() * (top - low);
        return value + low;
    }

}
