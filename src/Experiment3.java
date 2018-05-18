public class Experiment3 extends NeuralNetwork {
    public Experiment3(InputVector[] inputVectors) {
        super(16, 100, 26);
        super.initControlVariables(0.2, 0.2, 5000, 90);
        super.setInputVectors(inputVectors);
    }

    public double start() {
        return train();
    }
}
