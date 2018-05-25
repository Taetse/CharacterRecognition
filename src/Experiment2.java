public class Experiment2 extends Experiment {
    public static void main(String strings[]) {
        InputVector[] inputVectors = Utils.getInputVectors("letter-recognition.data");
        Utils.normalizeVectors(inputVectors);

        System.out.println("Initializing Experiment 2...");
        Experiment experiment = new Experiment2(inputVectors);

        System.out.println("Training...");
        (experiment).train();
        (experiment).validate();
    }

    public Experiment2(InputVector[] inputVectors) {
        super(inputVectors, "AEIOU");
        neuralNetwork = new NeuralNetwork(16 /*input neuron count*/, 50 /*hidden neuron count*/, 1 /*output neuron count*/);
        neuralNetwork.initControlVariables(0.5 /*learning rate*/, 0.2 /*momentum*/, 1000 /*max epoch*/, 99 /*desired accuracy*/);
    }

    public void generateTargetSets(String pattern) {
        for (InputVector inputVector : inputVectors) {
            inputVector.t = new int[] {(pattern.contains(String.valueOf(inputVector.classChar))? 1 : 0)};
        }
    }
}
