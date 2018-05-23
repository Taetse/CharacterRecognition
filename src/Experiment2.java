public class Experiment2 extends Experiment {
    public Experiment2(InputVector[] inputVectors) {
        super(inputVectors);
        neuralNetwork = new NeuralNetwork(16, 30, 1);
        neuralNetwork.initControlVariables(0.2, 0.1, 5000, 90);
    }

    public void generateTargetSets() {
        for (InputVector inputVector : inputVectors) {
            inputVector.t = new int[] {("AEIOU".contains(String.valueOf(inputVector.classChar))? 0 : 1)};
        }
    }
}
