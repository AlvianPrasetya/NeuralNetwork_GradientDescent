import java.io.*;

public class TrainingBatch {

	private String _sourceFileName;
	private String _batchName;
	private int _numTrainingSample;
	private int _batchPtr;
	private TrainingSample[] _trainingSample;
	
	// default constructor (construct from text file)
	public TrainingBatch(String sourceFileName){
		_sourceFileName = sourceFileName;
		
		try (BufferedReader br = new BufferedReader(new FileReader(_sourceFileName))){
			_batchName = br.readLine();
			_numTrainingSample = Integer.parseInt(br.readLine());
			_batchPtr = 0;
			_trainingSample = new TrainingSample[_numTrainingSample];
			
			String firstLine;
			String secondLine;
			
			firstLine = br.readLine();
			while (firstLine != null){
				secondLine = br.readLine();
				
				setNextTrainingSample(new TrainingSample(stringToMatrix(firstLine), stringToMatrix(secondLine)));
				
				firstLine = br.readLine();
			}
		} catch (Exception e){
			System.err.println(e.getMessage());
		}
	}
	
	// secondary constructor (randomize and evaluate samples)
	public TrainingBatch(int numTrainingSample, int numInput, int numDesiredOutput){
		_batchName = "training_batch_random";
		_numTrainingSample = numTrainingSample;
		_batchPtr = 0;
		
		// initialize training samples
		_trainingSample = new TrainingSample[_numTrainingSample];
		for (int i=0; i<_numTrainingSample; i++){
			_trainingSample[i] = new TrainingSample(numInput, numDesiredOutput);
		}
	}
	
	// accessors
	
	public String getBatchName(){
		return _batchName;
	}
	
	public int getNumTrainingSample(){
		return _numTrainingSample;
	}
	
	public int getBatchPtr(){
		return _batchPtr;
	}
	
	public TrainingSample getNextTrainingSample(){
		TrainingSample tempTrainingSample = _trainingSample[_batchPtr];
		
		// update batch pointer
		_batchPtr++;
		_batchPtr %= _numTrainingSample;
		
		return tempTrainingSample;
	}
	
	public TrainingSample getTrainingSampleById(int trainingSampleId){
		return _trainingSample[trainingSampleId];
	}
	
	public String getSourceFileName(){
		return _sourceFileName;
	}
	
	// mutators
	
	public void setBatchName(String newBatchName){
		_batchName = newBatchName;
	}
	
	public void setBatchPtr(int newBatchPtr){
		_batchPtr = newBatchPtr;
	}
	
	public void setNextTrainingSample(TrainingSample newTrainingSample){
		_trainingSample[_batchPtr] = newTrainingSample;
		
		// update batch pointer
		_batchPtr++;
		_batchPtr %= _numTrainingSample;
	}
	
	public void setTrainingSampleById(int trainingSampleId, TrainingSample newTrainingSample){
		_trainingSample[trainingSampleId] = newTrainingSample;
	}
	
	// task-specific functions
	
	public static Matrix stringToMatrix(String str){
		String[] strToken = str.split(" ");
		
		// initialize resultMatrix
		Matrix resultMatrix = new Matrix(strToken.length, 1);
		for (int i=0; i<strToken.length; i++){
			resultMatrix.setNextGrid(Double.parseDouble(strToken[i]));
		}
		
		return resultMatrix;
	}
	
}
