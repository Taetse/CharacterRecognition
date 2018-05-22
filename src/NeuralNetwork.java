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
    private int epochCount = 0;
    private int maxEpoch;
    private double desiredTrainingAccuracy;
    private double v[][];
    private double nextv[][];
    private double w[][];
    private double nextw[][];
    private double deltav[][];
    private double prevDeltav[][];
    private double deltaw[][];
    private double prevDeltaw[][];

    NeuralNetwork(int inputUnitCount, int hiddenUnitCount, int outputUniCount) {
        I = inputUnitCount + 1;
        J = hiddenUnitCount + 1;
        K = outputUniCount;

        z = new double[I];
        y = new double[J];
        o = new double[K];
        a = new int[K];

        z[I - 1] = y[J - 1] = -1;

        v = new double[J][I]; //hidden units x input units + bias unit
        w = new double[K][J]; //hidden units x input units + bias unit

        for (int j = 0; j < J; j++) {
            for (int i = 0; i < I; i++)
//                v[j][i] = 0.2;
                v[j][i] = getRandomValue(-(1.0/Math.sqrt(I)), (1.0/Math.sqrt(I)));
        }

        for (int k = 0; k < K; k++) {
            for (int j = 0; j < J; j++)
//                w[k][j] = 0.5;
                w[k][j] = getRandomValue(-(1.0/Math.sqrt(J)), (1.0/Math.sqrt(J)));
        }
        prevDeltav = new double[J][I];
        prevDeltaw = new double[K][J];
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
        epochCount = 0;
        printNetwork();
        boolean criteriaMet = false;
        double trainingAccuracy = 0;
        while (!criteriaMet) {
            trainingError = 0;

            for (InputVector inputVector : inputVectors) {
                boolean correctlyClassified = true;
                calculateZ(inputVector);
                calculateY();
                calculateO();
                correctlyClassified = !calculateA();

                //training error
                correctlyClassified = correctlyClassified && !missClassification(inputVector.t);
                if (correctlyClassified)
                    trainingError++;

                //Calculate the error signal for each output
                double errorO[] = calculateOutputErrorSignal(inputVector.t);

                //Calculate the new weight values for the hidden-to-output weights
                deltaw = calculateDeltaW(errorO);
                nextw = calculateW();

                //calculate the error signal for each hidden unit
                double errorY[] = calculateHiddenErrorSignal(errorO);

                //Calculate the new weight values for the weights between hidden neuron j and input neuron i
                deltav = calculateDeltaV(errorY);
                nextv = calculateV();

                trainingAccuracy = (trainingError / inputVectors.length) * 100;

                prevDeltaw = deltaw;
                prevDeltav = deltav;
                w = nextw;
                v = nextv;

//                printV();
//                printW();
//                System.out.println("done");
            }

            criteriaMet = (epochCount >= maxEpoch || desiredTrainingAccuracy < trainingAccuracy);
//            printNetwork();
            epochCount++;
            System.out.println("Epoch " + epochCount + " done. Accuracy: " + trainingAccuracy);
        }
        return trainingAccuracy;
    }

    private void calculateZ(InputVector inputVector) {
        System.arraycopy(inputVector.vector, 0, z, 0, I - 1);
    }

    private void calculateY() {
        for (int j = 0; j < J - 1; j++)
            y[j] = Fan(NetYJ(j));
    }

    private void calculateO() {
        for (int k = 0; k < K; k++)
            o[k] = Fan(NetOK(k));
    }

    private boolean calculateA() {
        boolean classEr = false;
        for (int k = 0; k < K; k++) {
            if (o[k] >= 0.7)
                a[k] = 1;
            else if (o[k] <= 0.3)
                a[k] = 0;
            else
                classEr = true;
        }
        return classEr;
    }

    private double[][] calculateV() {
        double[][] v = new double[J][I];
        for (int j = 0; j < J; j++) {
            for (int i = 0; i < I; i++)
                v[j][i] = v[j][i] * deltav[j][i] + (momentum * prevDeltav[j][i]);
        }
        return v;
    }

    private double[][] calculateDeltaV(double errorY[]) {
        double[][] deltaV = new double[J][I];
        for (int j = 0; j < J; j++) {
            for (int i = 0; i < I; i++)
                deltaV[j][i] = -learningRate * errorY[j] * z[i];
        }
        return deltaV;
    }

    private double[] calculateHiddenErrorSignal(double errorO[]) {
        double errorY[] = new double[J];
        for (int j = 0; j < J; j++) {
            for (int k = 0; k < K; k++)
                errorY[j] = errorO[k] * w[k][j] * (1 - y[j]) * y[j];
        }
        return errorY;
    }

    private double[][] calculateW() {
        double[][] w = new double[K][J];
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < J; j++)
                w[k][j] = w[k][j] + deltaw[k][j] + momentum * prevDeltaw[k][j];
        }
        return w;
    }

    private double[][] calculateDeltaW(double errorO[]) {
        double[][] deltaW = new double[K][J];
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < J; j++)
                deltaW[k][j] = -learningRate * errorO[k] * y[j];
        }
        return deltaW;
    }

    private double[] calculateOutputErrorSignal(int t[]) {
        double errorO[] = new double[K];
        for (int k = 0; k < K; k++)
            errorO[k] = -(t[k] - o[k]) * (1 - o[k]) * o[k];
        return errorO;
    }

    private boolean missClassification(int t[]) {
        for (int a = 0; a < K; a++)
            if (o[a] != t[a])
                return true;
        return false;
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

    private double Fan(double net) {
        return 1/(1 + Math.pow(Math.E, -1*(net)));
    }

    private double getRandomValue(double low, double top) {
        double value = Math.random() * (top - low);
        return value + low;
    }

    private void printNetwork() {
        System.out.println("Network");
        System.out.println("Z:");
        printZ();

        System.out.println("Y:");
        printY();

        System.out.println("O:");
        printO();

        System.out.println("V:");
        printV();

        System.out.println("W:");
        printW();
    }

    private void printW() {
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < J; j++) {
                System.out.print(w[k][j] + "|");
                System.out.print(prevDeltaw[k][j] + " ");
            }
            System.out.println("");
        }
    }

    private void printV() {
        for (int j = 0; j < J; j++) {
            for (int i = 0; i < I; i++) {
                System.out.print(v[j][i] + "|");
                System.out.print(prevDeltav[j][i] + " ");
            }
            System.out.println("");
        }
    }

    private void printZ() {
        for (int i = 0; i < I; i++)
            System.out.print(z[i] + " ");
        System.out.println("");
    }

    private void printY() {
        for (int j = 0; j < J; j++)
            System.out.print(y[j] + " ");
        System.out.println("");
    }

    private void printO() {
        for (int k = 0; k < K; k++)
            System.out.print(o[k] + " ");
        System.out.println("");
    }
}
