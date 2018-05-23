import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class Main {
    public static void main(String strings[]) {
        InputVector[] inputVectors = getInputVectors("letter-recognition.data");
//        InputVector[] inputVectors = getInputVectors("test.data");
        normalizeVectors(inputVectors);

        Experiment neuralNetwork = new Experiment3(inputVectors);
//        Experiment neuralNetwork = new Experiment2(inputVectors);
        System.out.println("Accuracy: ");
        (neuralNetwork).train();
        (neuralNetwork).validate();
    }

    private static void normalizeVectors(InputVector[] vectors) {
        for (int a = 0; a < vectors[0].vector.length; a++) {
            double biggest = Double.NEGATIVE_INFINITY;
            for (int b = 0; b < vectors.length; b++) {
                if (Math.abs(vectors[b].vector[a]) > biggest)
                    biggest = Math.abs(vectors[b].vector[a]);
            }

            if (biggest > Math.sqrt(3)) {
                double scale = (Math.sqrt(3) / biggest);

                for (int b = 0; b < vectors.length; b++)
                    vectors[b].vector[a] = (vectors[b].vector[a] * scale);
            }
        }
    }

    private static InputVector[] getInputVectors(String fileName) {
        ArrayList<String> fileLines = getFileLines(fileName);
        InputVector[] inputVectors = new InputVector[fileLines.size()];
        int index = 0;
        for (String line : fileLines) {
            if (line.length() == 0)
                continue;
            String split[] = line.split(",");
            InputVector inputVector = new InputVector();
            double attributes[] = new double[16];
            for (int a = 1; a < split.length; a++)
                attributes[a - 1] = Integer.parseInt(split[a]);
            inputVector.vector = attributes;
            inputVector.classChar = split[0].charAt(0);
            inputVectors[index++] = inputVector;
        }
        return inputVectors;
    }

    private static ArrayList<String> getFileLines(String fileName) {
        ArrayList<String> lines = new ArrayList<>();
        try {
            File file = new File(fileName);
            FileReader fileReader = new FileReader(file);
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                if (line.length() != 0)
                    lines.add(line);
            }
            fileReader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return lines;
    }
}
