// hyper-parameter(s):
// evaluateDesiredOutput() -> method to calculate the desired output of random training samples

public class TrainingSample {

	private int _numInput;
	private int _numDesiredOutput;
	private Matrix _inputMatrix;
	private Matrix _desiredOutputMatrix;
	
	// default constructor
	public TrainingSample(Matrix inputMatrix, Matrix desiredOutputMatrix){
		_numInput = inputMatrix.getNumRows();
		_numDesiredOutput = desiredOutputMatrix.getNumRows();
		_inputMatrix = inputMatrix;
		_desiredOutputMatrix = desiredOutputMatrix;
	}
	
	// secondary constructor (randomize and evaluate output)
	public TrainingSample(int numInput, int numDesiredOutput){
		_numInput = numInput;
		_numDesiredOutput = numDesiredOutput;
		_inputMatrix = Matrix.randomize(numInput, 1);
		_desiredOutputMatrix = evaluateDesiredOutput();
	}
	
	// accessors
	
	public int getNumInput(){
		return _numInput;
	}
	
	public int getNumDesiredOutput(){
		return _numDesiredOutput;
	}
	
	public Matrix getInputMatrix(){
		return _inputMatrix;
	}
	
	public Matrix getDesiredOutputMatrix(){
		return _desiredOutputMatrix;
	}
	
	// mutators
	
	public void setInputMatrix(Matrix newInputMatrix){
		_inputMatrix = newInputMatrix;
	}
	
	public void setDesiredOutputMatrix(Matrix newDesiredOutputMatrix){
		_desiredOutputMatrix = newDesiredOutputMatrix;
	}
	
	// mathematical functions
	
	public Matrix evaluateDesiredOutput(){
		Matrix desiredOutputMatrix = new Matrix(_numDesiredOutput, 1);
		
		double one = _inputMatrix.getNextGrid();
		double two = _inputMatrix.getNextGrid();
		// double three = _inputMatrix.getNextGrid();
		// double four = _inputMatrix.getNextGrid();
		
		if (one < 0.5 || two < 0.5){
			desiredOutputMatrix.setNextGrid(0.0);
		}
		else {
			desiredOutputMatrix.setNextGrid(1.0);
		}
		
		return desiredOutputMatrix;
	}
	
}
