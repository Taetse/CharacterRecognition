public class Experiment1 extends Experiment {
    public Experiment1(InputVector[] inputVectors, String classChar) {
        super(inputVectors, String.valueOf(classChar));
        neuralNetwork = new NeuralNetwork(16, 30, 1);
        neuralNetwork.initControlVariables(0.2, 0.1, 5000, 99.5);
    }

    public void generateTargetSets(String pattern) {
        for (InputVector inputVector : inputVectors) {
            inputVector.t = new int[] {(inputVector.classChar == Character.toUpperCase(pattern.charAt(0))? 1 : 0)};
        }
    }
}
