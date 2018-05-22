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
    private double w[][];
    private double deltav[][];
    private double prevDeltav[][];
    private double deltaw[][];
    private double prevDeltaw[][];

    NeuralNetwork(int inputUnitCount, int hiddenUnitCount, int outputUnitCount) {
        I = inputUnitCount + 1;
        J = hiddenUnitCount + 1;
        K = outputUnitCount;

        z = new double[I];
        y = new double[J];
        o = new double[K];
        a = new int[K];

        z[I - 1] = y[J - 1] = -1;

        v = new double[J][I]; //hidden units x input units + bias unit
        w = new double[K][J]; //hidden units x input units + bias unit

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
                calculateZ(inputVector);
                calculateY();
                calculateO();
                boolean correctlyClassified = !calculateA();
                if (correctlyClassified && !missClassification(inputVector.t))
                    trainingError++;

                double errorO[] = calculateErrorO(inputVector.t);
                double errorY[] = calculateErrorY(errorO);

                calculateDeltaW(errorO);
                calculateDeltaV(errorY);

                calculateW();
                calculateV();

                double temp[][] = prevDeltaw;
                prevDeltaw = deltaw;
                deltaw = temp;
                temp = prevDeltav;
                prevDeltav = deltav;
                deltav = temp;
            }
            trainingAccuracy = (trainingError / inputVectors.length) * 100;

            criteriaMet = (epochCount >= maxEpoch || desiredTrainingAccuracy < trainingAccuracy);
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

    private void calculateV() {
        for (int j = 0; j < J; j++)
            for (int i = 0; i < I; i++)
                v[j][i] += + deltav[j][i] + (momentum * prevDeltav[j][i]);
    }

    private void calculateDeltaV(double errorY[]) {
        for (int j = 0; j < J; j++)
            for (int i = 0; i < I; i++)
                deltav[j][i] = -learningRate * errorY[j] * z[i];
    }

    private double[] calculateErrorY(double errorO[]) {
        double errorY[] = new double[J];
        for (int j = 0; j < J; j++)
            for (int k = 0; k < K; k++)
                errorY[j] += errorO[k] * w[k][j] * (1 - y[j]) * y[j];
        return errorY;
    }

    private void calculateW() {
        for (int k = 0; k < K; k++)
            for (int j = 0; j < J; j++)
                w[k][j] += deltaw[k][j] + momentum * prevDeltaw[k][j];
    }

    private void calculateDeltaW(double errorO[]) {
        for (int k = 0; k < K; k++)
            for (int j = 0; j < J; j++)
                deltaw[k][j] = -learningRate * errorO[k] * y[j];
    }

    private double[] calculateErrorO(int t[]) {
        double errorO[] = new double[K];
        for (int k = 0; k < K; k++)
            errorO[k] = -(t[k] - o[k]) * (1 - o[k]) * o[k];
        return errorO;
    }

    private boolean missClassification(int t[]) {
        for (int k = 0; k < K; k++)
            if (a[k] != t[k])
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
