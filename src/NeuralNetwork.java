import java.util.ArrayList;

public class NeuralNetwork {
    private InputVector[] inputVectors;
    private int J;
    private int K;
    private int I;
    private double z[];
    private double y[];
    private double o[];
    private int a[];
    private double learningRate;
    private double momentum;
    private double trainingError;
    private int maxEpoch;
    private double desiredTrainingAccuracy;
    private EpochManager epochManager = new EpochManager();

    private class EpochManager {
        EpochManager() {
            prev = new Epoch();
            current = new Epoch();
            next = new Epoch();
        }

        public int count = 0;
        public Epoch prev = null;
        public Epoch current = null;
        public Epoch next = null;

        public void nextEpoch() {
            count++;
            prev = current;
            current = next;
            next = new Epoch();
        }
    }

    private class Epoch {
        public double v[][];
        public double w[][];
        public double deltav[][];
        public double deltaw[][];
    }

    NeuralNetwork(int inputUnitCount, int hiddenUnitCount, int outputUniCount) {
        I = inputUnitCount + 1;
        J = hiddenUnitCount + 1;
        K = outputUniCount;

        z = new double[I];
        y = new double[J];
        o = new double[K];
        a = new int[K];

        z[I - 2] = -1;
        y[J - 2] = -1;

        double v[][] = new double[J][I]; //hidden units x input units + bias unit
        double w[][] = new double[K][J]; //hidden units x input units + bias unit

        for (int j = 0; j < J; j++) {
            for (int i = 0; i < I; i++)
                v[j][i] = getRandomValue(-(1.0/I), (1.0/I));
        }

        for (int k = 0; k < K; k++) {
            for (int j = 0; j < J; j++)
                w[k][j] = getRandomValue(-(1.0/J), (1.0/J));
        }
        epochManager.current.v = v;
        epochManager.current.w = w;
        epochManager.prev.deltav = new double[J][I];
        epochManager.prev.deltaw = new double[K][J];
    }

    public void initControlVariables(double learningRate, double momentum, int maxEpoch, double desiredTrainingAccuracy) {
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.maxEpoch = maxEpoch;
        this.desiredTrainingAccuracy = desiredTrainingAccuracy;
    }

    public void setInputVectors(InputVector[] inputVectors) {
        this.inputVectors = inputVectors;
    }

    public double train() {
        boolean criteriaMet = false;
        double trainingAccuracy = 0;
        while (!criteriaMet) {
            trainingError = 0;

            for (InputVector inputVector : inputVectors) {
                boolean missClassification = false;
                System.arraycopy(inputVector.vector, 0, z, 0, I - 1);
                for (int j = 0; j < J - 1; j++)
                    y[j] = activation(NetYJ(j, z));

                for (int k = 0; k < K; k++) {
                    o[k] = activation(NetOK(k, y));
                    if (o[k] >= 0.7)
                        a[k] = 1;
                    else if (o[k] <= 0.3)
                        a[k] = 0;
                }

                //training error
                missClassification = !missClassification(a, inputVector.t);
                if (missClassification)
                    trainingError++;

                //Calculate the error signal for each output
                double errorO[] = getOutputErrorSignal(inputVector.t, o);

                //Calculate the new weight values for the hidden-to-output weights
                epochManager.current.deltaw = calculateDeltaW(errorO);
                epochManager.next.w = calculateWNextEpoch();

                //calculate the error signal for each hidden unit
                double errorY[] = getHiddenErrorSignal(errorO);

                //Calculate the new weight values for the weights between hidden neuron j and input neuron i
                epochManager.current.deltav = calculateDeltaV(errorY);
                epochManager.next.v = calculateVNextEpoch();

                trainingAccuracy = (trainingError / inputVectors.length) * 100;
            }

            criteriaMet = (epochManager.count >= maxEpoch || desiredTrainingAccuracy < trainingAccuracy);
            epochManager.nextEpoch();
            System.out.println("Epoch " + epochManager.count + " done. Accuracy: " + trainingAccuracy);
        }
        return trainingAccuracy;
    }

    private double[][] calculateVNextEpoch() {
        double[][] v = new double[J][I];
        for (int j = 0; j < J; j++) {
            for (int i = 0; i < I; i++)
                v[j][i] = epochManager.current.v[j][i] * epochManager.current.deltav[j][i] + momentum * epochManager.prev.deltav[j][i];
        }
        return v;
    }

    private double[][] calculateDeltaV(double errorY[]) {
        double[][] weightValues = new double[J][I];
        for (int j = 0; j < J; j++) {
            for (int i = 0; i < I; i++)
                weightValues[j][i] = -learningRate * errorY[j] * z[i];
        }
        return weightValues;
    }

    private double[] getHiddenErrorSignal(double errorO[]) {
        double error[] = new double[J];
        for (int j = 0; j < J; j++) {
            for (int k = 0; k < K; k++)
                error[j] = errorO[k] * epochManager.current.w[k][j] * (1 - y[j]) * y[j];
        }
        return error;
    }

    private double[][] calculateWNextEpoch() {
        double[][] w = new double[K][J];
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < J; j++)
                w[k][j] = epochManager.current.w[k][j] + epochManager.current.deltaw[k][j] + momentum * epochManager.prev.deltaw[k][j];
        }
        return w;
    }

    private double[][] calculateDeltaW(double errorO[]) {
        double[][] weightValues = new double[K][J];
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < J; j++)
                weightValues[k][j] = -learningRate * errorO[k] * y[j];
        }
        return weightValues;
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
            if (output[a] != target[a])
                return true;
        return false;
    }

    private double NetYJ(int j, double inputs[]) {
        double net = 0;
        for (int x = 0; x < I; x++)
            net += (epochManager.current.v[j][x] * inputs[x]);
        return net;
    }

    private double NetOK(int k, double inputs[]) {
        double net = 0;
        for (int x = 0; x < J; x++)
            net += (epochManager.current.w[k][x] * inputs[x]);
        return net;
    }

    private double activation(double net) {
        return 1/(1 + Math.pow(Math.E, -1*(net)));
    }

    private double getRandomValue(double low, double top) {
        double value = Math.random() * (top - low);
        return value + low;
    }

}
