public class Experiment2 extends Experiment {
    public Experiment2(InputVector[] inputVectors) {
        super(inputVectors);
        neuralNetwork = new NeuralNetwork(16, 5, 1);
        neuralNetwork.initControlVariables(0.1, 0.2, 100, 90);
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
