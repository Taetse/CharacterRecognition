public class Experiment2 extends Experiment {
    public Experiment2(InputVector[] inputVectors) {
        super(inputVectors);
        neuralNetwork = new NeuralNetwork(16, 10, 1);
        neuralNetwork.initControlVariables(0.2, 0.05, 10000, 98);
        neuralNetwork.setInputVectors(inputVectors);
    }

    public double start() {
        return neuralNetwork.train();
    }

    public void generateTargetSets() {
        for (InputVector inputVector : inputVectors) {
            inputVector.t = new int[] {("AEIOU".contains(String.valueOf(inputVector.classChar))? 1 : 0)};
        }
    }
}
