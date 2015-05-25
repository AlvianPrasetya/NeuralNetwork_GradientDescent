import java.util.*;

public class testRun {
	
	public static void main(String[] args){		
		// start debug mode
		
		System.out.println("Starting debug mode...");
		
		Scanner sc = new Scanner(System.in);
		
		String cmd;
		int layerId;
		double tempDouble;
		
		// initialize neural network
		NeuralNetwork testNetwork;
		System.out.print("Enter depth of neural network: ");
		int depth = sc.nextInt();
		int[] initArray = new int[depth];
		
		for (int i=0; i<depth; i++){
			initArray[i] = sc.nextInt();
		}
		
		testNetwork = new NeuralNetwork(initArray);
		
		// neural network debug
		do {
			cmd = sc.next();
			
			if (cmd.equals("weight")){
				layerId = sc.nextInt();
				System.out.println(testNetwork.getWeightMatrixById(layerId).toString());
			}
			
			else if (cmd.equals("bias")){
				layerId = sc.nextInt();
				System.out.println(testNetwork.getBiasMatrixById(layerId).toString());
			}
			
			else if (cmd.equals("input")){
				layerId = sc.nextInt();
				System.out.println(testNetwork.getInputActivationMatrixById(layerId).toString());
			}
			
			else if (cmd.equals("weighted")){
				layerId = sc.nextInt();
				System.out.println(testNetwork.getWeightedSumMatrixById(layerId).toString());
			}
			
			else if (cmd.equals("output")){
				layerId = sc.nextInt();
				System.out.println(testNetwork.getOutputActivationMatrixById(layerId).toString());
			}
			
			else if (cmd.equals("evaluate")){
				cmd = sc.next();
				
				if (cmd.equals("random")){
					int numTrainingSample = sc.nextInt();
					
					testNetwork.evaluateTrainingBatch(new TrainingBatch(numTrainingSample, testNetwork.getNeuralLayerById(0).getNumNeuron(), testNetwork.getNeuralLayerById(testNetwork.getNumLayer() - 1).getNumNeuron()));
				}
				else if (cmd.equals("batch")){
					String fileName = sc.next();
					
					testNetwork.evaluateTrainingBatch(new TrainingBatch(fileName));
				}
			}
			
			else if (cmd.equals("feed")){
				int numInputNeuron = testNetwork.getNeuralLayerById(0).getNumNeuron();
				
				Matrix inputActivation = new Matrix(numInputNeuron, 1);
				
				for (int i=0; i<numInputNeuron; i++){
					tempDouble = sc.nextDouble();
					inputActivation.setNextGrid(tempDouble);
				}
				
				System.out.println(testNetwork.feedForward(inputActivation).toString());
			}
			
			else if (cmd.equals("esc")){
				sc.close();
				System.out.println("Escaping debug mode...");
			}
			
			else {
				System.out.println("Command not recognized");
			}
			
		} while (!cmd.equals("esc"));
		
		System.out.println("End of program");
	}
	
}
