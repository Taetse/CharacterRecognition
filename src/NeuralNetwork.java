import java.io.*;
import java.util.Arrays;
import java.util.Collections;

public class NeuralNetwork {
    private int J, K, I;
    private double[] z, y, o, errorO, errorY;
    private int a[];
    private double learningRate, momentum, correctlyClassified, desiredTrainingAccuracy;
    private int maxEpoch;
    private double[][] v, w, deltav, prevDeltav, deltaw, prevDeltaw;

    NeuralNetwork(int inputUnitCount, int hiddenUnitCount, int outputUnitCount) {
        I = inputUnitCount + 1; //+1 for bias
        J = hiddenUnitCount + 1; //+1 for bias
        K = outputUnitCount;

        z = new double[I];
        y = new double[J];
        o = new double[K];
        a = new int[K];

        z[I - 1] = y[J - 1] = -1; //bias inputs

        v = new double[J][I];
        w = new double[K][J];

        errorO = new double[K];
        errorY = new double[J];

        prevDeltav = new double[J][I];
        deltav = new double[J][I];
        prevDeltaw = new double[K][J];
        deltaw = new double[K][J];

        for (int j = 0; j < J; j++)
            for (int i = 0; i < I; i++)
                v[j][i] = getRandomValue(-(1.0/Math.sqrt(I)), (1.0/Math.sqrt(I)));

        for (int k = 0; k < K; k++)
            for (int j = 0; j < J; j++)
                w[k][j] = getRandomValue(-(1.0/Math.sqrt(J)), (1.0/Math.sqrt(J)));
    }

    public void initControlVariables(double learningRate, double momentum, int maxEpoch, double desiredTrainingAccuracy) {
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.maxEpoch = maxEpoch;
        this.desiredTrainingAccuracy = desiredTrainingAccuracy;
        writeOutputLine("NeuralNetwork setup. LearningRate:" + learningRate + ", Momentum:" + momentum + ", HiddenNeurons:" + (J - 1) + ", OutputNeurons:" + K);
    }

    public double validate(InputVector[] validationSet) {
        correctlyClassified = 0;
        int counter = 0;
        for (InputVector pattern : validationSet) {
            boolean isCC = feedForward(pattern.vector, pattern.t);
            if (isCC)
                correctlyClassified++;
            System.out.println("Pattern #" + ++counter + ": Char:" + pattern.classChar + ", Correct:" + isCC);
            for (int k = 0; k < K; k++)
                System.out.println(" o[" + k + "] Output:" + String.format("%.3f", o[k]) + ", Actual:" + a[k] + ", Target:" + pattern.t[k]);
        }
        writeOutputLine("Dv. Accuracy:" + String.format("%.3f", (correctlyClassified / validationSet.length) * 100));
        return  (correctlyClassified / validationSet.length) * 100;
    }

    public double train(InputVector[] Dt, String setDescription) {
        int epochCount = 0;
        boolean criteriaMet = false;
        double trainingAccuracy = 0, E = 0;
        System.out.println("#, Accuracy");
        while (!criteriaMet) {
            E = correctlyClassified = 0;
            randomizeSet(Dt);

            for (InputVector pattern : Dt) {
                if (feedForward(pattern.vector, pattern.t))
                    correctlyClassified++;
                backPropagate(pattern);
                E += calculateE(pattern.t);
            }
            E /= (Dt.length * K * 2);
            trainingAccuracy = (correctlyClassified / Dt.length) * 100;
            epochCount++;

            criteriaMet = (epochCount >= maxEpoch || desiredTrainingAccuracy < trainingAccuracy);
//            System.out.println("Epoch " + epochCount + ". Accuracy: " + String.format("%.3f", trainingAccuracy));
//            System.out.println(setDescription + " Epoch:" + epochCount + ". E:" + String.format("%.3f", E) + ", Accuracy:" + String.format("%.3f", trainingAccuracy));
            System.out.println(epochCount + ", " + String.format("%.3f", trainingAccuracy));
        }
        writeOutputLine(setDescription + ". Accuracy:" + String.format("%.3f", trainingAccuracy));
//        writeOutputLine(setDescription + ". E:" + String.format("%.3f", E) + ", Accuracy:" + String.format("%.3f", trainingAccuracy));
        return trainingAccuracy;
    }

    private boolean feedForward(double in[], int t[]) {
        calculateZ(in); //calculate input layer outputs
        calculateY(); //calculate hidden layer outputs
        calculateO(); //calculate output layer outputs
        calculateA(); //calculate actual values
        return !missClassification(t);
    }

    private void backPropagate(InputVector pattern) {
        calculateErrorO(pattern.t); //calculate output error
        calculateErrorY(); //calculate hidden error

        calculateDeltaW(); //hidden - output weight update steps
        calculateDeltaV(); //input - hidden weights update steps

        calculateW(); //update hidden - output weights
        calculateV(); //update input - hidden weights

        //swap of deltas (only for efficiency reasons)
        double temp[][] = prevDeltaw;
        prevDeltaw = deltaw;
        deltaw = temp;
        temp = prevDeltav;
        prevDeltav = deltav;
        deltav = temp;
    }

    private void calculateZ(double in[]) {
        System.arraycopy(in, 0, z, 0, I - 1);
    }

    private void calculateY() {
        for (int j = 0; j < J - 1; j++)
            y[j] = Fan(NetYJ(j));
    }

    private void calculateO() {
        for (int k = 0; k < K; k++)
            o[k] = Fan(NetOK(k));
    }

    private void calculateA() {
        for (int k = 0; k < K; k++) {
            if (o[k] >= 0.7)
                a[k] = 1;
            else if (o[k] <= 0.3)
                a[k] = 0;
            else
                a[k] = -1;
        }
    }

    private void calculateV() {
        for (int j = 0; j < J; j++)
            for (int i = 0; i < I; i++)
                v[j][i] += deltav[j][i] + (momentum * prevDeltav[j][i]);
    }

    private void calculateDeltaV() {
        for (int j = 0; j < J; j++)
            for (int i = 0; i < I; i++)
                deltav[j][i] = -learningRate * errorY[j] * z[i];
    }

    private void calculateErrorY() {
        for (int j = 0; j < J; j++) {
            errorY[j] = 0;
            for (int k = 0; k < K; k++)
                errorY[j] += errorO[k] * w[k][j] * (1 - y[j]) * y[j];
        }
    }

    private void calculateW() {
        for (int k = 0; k < K; k++)
            for (int j = 0; j < J; j++)
                w[k][j] += deltaw[k][j] + (momentum * prevDeltaw[k][j]);
    }

    private void calculateDeltaW() {
        for (int k = 0; k < K; k++)
            for (int j = 0; j < J; j++)
                deltaw[k][j] = -learningRate * errorO[k] * y[j];
    }

    private void calculateErrorO(int t[]) {
        for (int k = 0; k < K; k++)
            errorO[k] = -(t[k] - o[k]) * (1 - o[k]) * o[k];
    }

    private boolean missClassification(int t[]) {
        for (int k = 0; k < K; k++)
            if (a[k] != t[k])
                return true;
        return false;
    }

    private double calculateE(int t[]) {
        double e = 0;
        for (int k = 0; k < K; k++)
            e += Math.pow((t[k] - o[k]), 2);
        return e;
    }

    private double NetYJ(int j) {
        double net = 0;
        for (int i = 0; i < I; i++)
            net += (v[j][i] * z[i]);
        return net;
    }

    private double NetOK(int k) {
        double net = 0;
        for (int j = 0; j < J; j++)
            net += (w[k][j] * y[j]);
        return net;
    }

    private void writeOutputLine(String line) {
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter("output.txt", true));

            bw.write(line);
            bw.newLine();

            bw.close();
        } catch (FileNotFoundException ex) {
            ex.printStackTrace();
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    private double Fan(double net) {
        return 1/(1 + Math.pow(Math.E, -1 * (net)));
    }

    private double getRandomValue(double low, double top) {
        return (Math.random() * (top - low)) + low;
    }

    protected void randomizeSet(InputVector[] set) {
        Collections.shuffle(Arrays.asList(set));
    }
}
