public abstract class Experiment {
    protected InputVector[] inputVectors;
    protected NeuralNetwork neuralNetwork;
    public Experiment(InputVector[] inputVectors) {
        this.inputVectors = inputVectors;
        generateTargetSets();
    }

    abstract double start();
    abstract void generateTargetSets();
}
