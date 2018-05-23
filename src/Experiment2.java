public class Experiment2 extends Experiment {
    public Experiment2(InputVector[] inputVectors) {
        super(inputVectors, "AEIOU");
        neuralNetwork = new NeuralNetwork(16, 30, 1);
        neuralNetwork.initControlVariables(0.2, 0.1, 5000, 90);
    }

    public void generateTargetSets(String pattern) {
        for (InputVector inputVector : inputVectors) {
            inputVector.t = new int[] {(pattern.contains(String.valueOf(inputVector.classChar))? 1 : 0)};
        }
    }
}
