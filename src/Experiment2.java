public class Experiment2 extends Experiment {
    public Experiment2(InputVector[] inputVectors) {
        super(inputVectors);
        neuralNetwork = new NeuralNetwork(16, 60, 1);
        neuralNetwork.initControlVariables(0.5, 0.1, 10000, 99);
        neuralNetwork.setDataSets(Dt, Dg, Dv);
    }

    public double start() {
        return neuralNetwork.train();
    }

    public void generateTargetSets() {
        for (InputVector inputVector : inputVectors) {
            inputVector.t = new int[] {("AEIOU".contains(String.valueOf(inputVector.classChar))? 0 : 1)};
        }
    }
}
