import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class Main {
    public static void main(String strings[]) {
        ArrayList<InputVector> inputVectors = getInputVectors("letter-recognition.data");

    }

    private static ArrayList<InputVector> getInputVectors(String fileName) {
        ArrayList<InputVector> inputVectors = new ArrayList<>();
        ArrayList<String> fileLines = getFileLines(fileName);
        for (String line : fileLines) {
            String split[] = line.split(",");
            InputVector inputVector = new InputVector();
            int attributes[] = new int[16];
            for (int a = 1; a < split.length; a++)
                attributes[a] = Integer.parseInt(split[a]);
            inputVector.vector = attributes;
            inputVector.classChar = split[0].charAt(0);
            inputVectors.add(inputVector);
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
            while ((line = bufferedReader.readLine()) != null)
                lines.add(line);
            fileReader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return lines;
    }
}
