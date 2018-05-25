import java.util.Scanner;

public class Experiment1 extends Experiment {
    public static void main(String strings[]) {
        InputVector[] inputVectors = Utils.getInputVectors("letter-recognition.data");
        Utils.normalizeVectors(inputVectors);

        System.out.println("Initializing Experiment 1...");
        System.out.print("Enter a character to distinguish: ");
        Scanner s = new Scanner(System.in);
        String str = s.nextLine();
        Experiment experiment = new Experiment1(inputVectors, str);

        System.out.println("Training...");
        (experiment).train();
        (experiment).validate();
    }

    public Experiment1(InputVector[] inputVectors, String classChar) {
        super(inputVectors, String.valueOf(classChar));
        neuralNetwork = new NeuralNetwork(16 /*input neuron count*/, 30 /*hidden neuron count*/, 1 /*output neuron count*/);
        neuralNetwork.initControlVariables(0.2 /*learning rate*/, 0.2 /*momentum*/, 1000 /*max epoch*/, 99.8 /*desired accuracy*/);
    }

    public void generateTargetSets(String pattern) {
        for (InputVector inputVector : inputVectors) {
            inputVector.t = new int[] {(inputVector.classChar == Character.toUpperCase(pattern.charAt(0))? 1 : 0)};
        }
    }
}
