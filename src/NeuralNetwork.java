// hyper-parameter(s):
// LEARNING_RATE -> learning constant
// TARGET_ACCURACY -> accuracy to achieve in each training batch (1.00 -> 100%)

public class NeuralNetwork {
	
	// constants
	public static final double LEARNING_RATE = 0.30;
	public static final double TARGET_ACCURACY = 0.995;
	
	private int _numLayer;
	private NeuralLayer[] _neuralLayer;
	private Matrix[] _weightMatrix;
	private Matrix[] _biasMatrix;
	private Matrix[] _inputActivationMatrix;
	private Matrix[] _weightedSumMatrix;
	private Matrix[] _outputActivationMatrix;
	private Matrix[] _errorMatrix;
	
	// default constructor
	public NeuralNetwork(int[] numNeuronAtLayer){
		_numLayer = numNeuronAtLayer.length;
		
		_weightMatrix = new Matrix[_numLayer];
		_biasMatrix = new Matrix[_numLayer];
		_inputActivationMatrix = new Matrix[_numLayer];
		_weightedSumMatrix = new Matrix[_numLayer];
		_outputActivationMatrix = new Matrix[_numLayer];
		_errorMatrix = new Matrix[_numLayer];
		
		// initialize neural layers
		_neuralLayer = new NeuralLayer[_numLayer];
		for (int i=0; i<_numLayer; i++){
			_neuralLayer[i] = new NeuralLayer(i, numNeuronAtLayer[i]);
			
			if (i != 0){
				// set previous neural layer's _outputLayer
				_neuralLayer[i-1].setOutputLayer(_neuralLayer[i]);
				
				// set current neural layer's _inputLayer
				_neuralLayer[i].setInputLayer(_neuralLayer[i-1]);
				
				// initialize current neural layer's _weightMatrix
				_neuralLayer[i].initializeWeightMatrix();
			}
		}
		
		// bottom up synchronization
		updateWeightBottomUp();
		updateBiasBottomUp();
		
		System.out.println("> Initialized neural network of " + _numLayer + " layer(s)");
	}
	
	// accessors
	
	public int getNumLayer(){
		return _numLayer;
	}
	
	public NeuralLayer[] getNeuralLayerList(){
		return _neuralLayer;
	}
	
	public NeuralLayer getNeuralLayerById(int layerId){
		return _neuralLayer[layerId];
	}
	
	public Matrix getWeightMatrixById(int layerId){
		return _weightMatrix[layerId];
	}
	
	public Matrix getBiasMatrixById(int layerId){
		return _biasMatrix[layerId];
	}
	
	public Matrix getInputActivationMatrixById(int layerId){
		return _inputActivationMatrix[layerId];
	}
	
	public Matrix getWeightedSumMatrixById(int layerId){
		return _weightedSumMatrix[layerId];
	}
	
	public Matrix getOutputActivationMatrixById(int layerId){
		return _outputActivationMatrix[layerId];
	}
	
	// mutators

	// update weight based on value in neurons
	public void updateWeightBottomUp(){
		for (int i=1; i<_numLayer; i++){
			_neuralLayer[i].collateWeight();
		}
		
		collateWeight();
	}

	// update weight based on value in neural network
	public void updateWeightTopDown(){
		disperseWeight();
		
		for (int i=1; i<_numLayer; i++){
			_neuralLayer[i].disperseWeight();
		}
	}

	// update bias based on value in neurons
	public void updateBiasBottomUp(){
		for (int i=1; i<_numLayer; i++){
			_neuralLayer[i].collateBias();
		}
		
		collateBias();
	}

	// update bias based on value in neural network
	public void updateBiasTopDown(){
		disperseBias();
		
		for (int i=1; i<_numLayer; i++){
			_neuralLayer[i].disperseBias();
		}
	}

	// update input activation based on value in neurons
	public void updateInputActivationBottomUp(){
		for (int i=0; i<_numLayer; i++){
			_neuralLayer[i].collateInputActivation();
		}
		
		collateInputActivation();
	}

	// update input activation based on value in neural network
	public void updateInputActivationTopDown(){
		disperseInputActivation();
		
		for (int i=0; i<_numLayer; i++){
			_neuralLayer[i].disperseInputActivation();
		}
	}

	// update weighted sum based on value in neurons
	public void updateWeightedSumBottomUp(){
		for (int i=1; i<_numLayer; i++){
			_neuralLayer[i].collateWeightedSum();
		}
		
		collateWeightedSum();
	}

	// update weighted sum based on value in neural network
	public void updateWeightedSumTopDown(){
		disperseWeightedSum();
		
		for (int i=1; i<_numLayer; i++){
			_neuralLayer[i].disperseWeightedSum();
		}
	}

	// update output activation based on value in neurons
	public void updateOutputActivationBottomUp(){
		for (int i=0; i<_numLayer; i++){
			_neuralLayer[i].collateOutputActivation();
		}
		
		collateOutputActivation();
	}

	// update output activation based on value in neural network
	public void updateOutputActivationTopDown(){
		disperseOutputActivation();
		
		for (int i=0; i<_numLayer; i++){
			_neuralLayer[i].disperseOutputActivation();
		}
	}
	
	// update output error based on value in neurons
	public void updateErrorBottomUp(){
		for (int i=1; i<_numLayer; i++){
			_neuralLayer[i].collateError();
		}
		
		collateError();
	}
	
	// update output error based on value in neural network
	public void updateErrorTopDown(){
		disperseError();
		
		for (int i=1; i<_numLayer; i++){
			_neuralLayer[i].disperseError();
		}
	}
	
	// synchronize functions
	
	public void collateWeight(){
		for (int i=1; i<_numLayer; i++){
			_weightMatrix[i] = _neuralLayer[i].getWeightMatrix();
		}
	}
	
	public void disperseWeight(){
		for (int i=1; i<_numLayer; i++){
			_neuralLayer[i].setWeightMatrix(_weightMatrix[i]);
		}
	}
	
	public void collateBias(){
		for (int i=1; i<_numLayer; i++){
			_biasMatrix[i] = _neuralLayer[i].getBiasMatrix();
		}
	}
	
	public void disperseBias(){
		for (int i=1; i<_numLayer; i++){
			_neuralLayer[i].setBiasMatrix(_biasMatrix[i]);
		}
	}
	
	public void collateInputActivation(){
		for (int i=0; i<_numLayer; i++){
			_inputActivationMatrix[i] = _neuralLayer[i].getInputActivationMatrix();
		}
	}
	
	public void disperseInputActivation(){
		for (int i=0; i<_numLayer; i++){
			_neuralLayer[i].setInputActivationMatrix(_inputActivationMatrix[i]);
		}
	}
	
	public void collateWeightedSum(){
		for (int i=1; i<_numLayer; i++){
			_weightedSumMatrix[i] = _neuralLayer[i].getWeightedSumMatrix();
		}
	}
	
	public void disperseWeightedSum(){
		for (int i=1; i<_numLayer; i++){
			_neuralLayer[i].setWeightedSumMatrix(_weightedSumMatrix[i]);
		}
	}
	
	public void collateOutputActivation(){
		for (int i=0; i<_numLayer; i++){
			_outputActivationMatrix[i] = _neuralLayer[i].getOutputActivationMatrix();
		}
	}
	
	public void disperseOutputActivation(){
		for (int i=0; i<_numLayer; i++){
			_neuralLayer[i].setOutputActivationMatrix(_outputActivationMatrix[i]);
		}
	}
	
	public void collateError(){
		for (int i=1; i<_numLayer; i++){
			_errorMatrix[i] = _neuralLayer[i].getErrorMatrix();
		}
	}
	
	public void disperseError(){
		for (int i=1; i<_numLayer; i++){
			_neuralLayer[i].setErrorMatrix(_errorMatrix[i]);
		}
	}
	
	// functionalities
	
	public void evaluateTrainingBatch(TrainingBatch trainingBatch){
		
		int numIteration = 0;
		int numCorrectSample = 0;
		
		Matrix[] deltaWeightMatrix = new Matrix[_numLayer];
		Matrix[] deltaBiasMatrix = new Matrix[_numLayer];
		
		do {
			// update
			numIteration++;
			
			// initialize
			numCorrectSample = 0;
			
			// calculate delta
			for (int i=0; i<trainingBatch.getNumTrainingSample(); i++){
				numCorrectSample += evaluateTrainingSample(trainingBatch.getNextTrainingSample());
				
				// computation of deltaWeightMatrix[] and deltaBiasMatrix[]
				
				if (i == 0){ // first training sample
					for (int j=1; j<_numLayer; j++){
						deltaWeightMatrix[j] = _errorMatrix[j].matrixMultiply(_inputActivationMatrix[j].transpose());
						deltaBiasMatrix[j] = _errorMatrix[j];
					}
				}
				else { // remaining training sample
					for (int j=1; j<_numLayer; j++){
						deltaWeightMatrix[j] = deltaWeightMatrix[j].matrixAdd(_errorMatrix[j].matrixMultiply(_inputActivationMatrix[j].transpose()));
						deltaBiasMatrix[j] = deltaBiasMatrix[j].matrixAdd(_errorMatrix[j]);
					}
				}
			}
			
			// apply delta
			for (int i=1; i<_numLayer; i++){
				deltaWeightMatrix[i].scalarMultiply(LEARNING_RATE / trainingBatch.getNumTrainingSample());
				deltaBiasMatrix[i].scalarMultiply(LEARNING_RATE / trainingBatch.getNumTrainingSample());
				
				// update weight and bias of each neural layer
				_weightMatrix[i] = _weightMatrix[i].matrixSubtract(deltaWeightMatrix[i]);
				_biasMatrix[i] = _biasMatrix[i].matrixSubtract(deltaBiasMatrix[i]);
			}
			
			updateWeightTopDown();
			updateBiasTopDown();
			
			// iteration summary
			System.out.println("Iteration " + numIteration + " completed with " + numCorrectSample 
								+ "/" + trainingBatch.getNumTrainingSample() + " correct sample(s)");
			
		} while (numCorrectSample < TARGET_ACCURACY * trainingBatch.getNumTrainingSample());
	}
	
	public int evaluateTrainingSample(TrainingSample trainingSample){
		Matrix trainingOutput = feedForward(trainingSample.getInputMatrix());
		Matrix trainingError = computeError(trainingSample.getDesiredOutputMatrix());
		backpropagateError(trainingError);

		if ((trainingOutput.matrixNormalize()).equals(trainingSample.getDesiredOutputMatrix())){
			return 1;
		}
		else return 0;
	}
	
	public Matrix feedForward(Matrix inputActivation){
		// nothing happens in the first/input layer
		_inputActivationMatrix[0] = inputActivation;
		_outputActivationMatrix[0] = inputActivation;
		
		_inputActivationMatrix[1] = inputActivation;
		
		for (int i=1; i<_numLayer; i++){
			// out = sigmoid(w.in + b)
			_weightedSumMatrix[i] = (_weightMatrix[i].matrixMultiply(_inputActivationMatrix[i])).matrixAdd(_biasMatrix[i]);
			_outputActivationMatrix[i] = _weightedSumMatrix[i].sigmoid_vec();
			
			if (i != _numLayer-1){ // if not last/output layer
				_inputActivationMatrix[i+1] = _outputActivationMatrix[i];
			}
		}
		
		// top down synchronization
		updateInputActivationTopDown();
		updateWeightedSumTopDown();
		updateOutputActivationTopDown();
		
		return _outputActivationMatrix[_numLayer-1];
	}
	
	public Matrix computeError(Matrix desiredOutputMatrix){
		Matrix networkOutputMatrix = _outputActivationMatrix[_numLayer-1];
		Matrix derivativeCostMatrix = networkOutputMatrix.matrixSubtract(desiredOutputMatrix);
		
		Matrix derivativeActivationMatrix = _weightedSumMatrix[_numLayer-1].sigmoid_derivative_vec();
		
		_errorMatrix[_numLayer-1] = derivativeCostMatrix.matrixHadamard(derivativeActivationMatrix);
		
		return _errorMatrix[_numLayer-1];
	}
	
	public void backpropagateError(Matrix outputErrorMatrix){
		// temporary objects
		Matrix tempErrorMatrix;
		Matrix tempDerivativeActivationMatrix;
		
		for (int i=_numLayer-2; i>0; i--){
			
			tempErrorMatrix = (_weightMatrix[i+1].transpose()).matrixMultiply(_errorMatrix[i+1]);
			tempDerivativeActivationMatrix = _weightedSumMatrix[i].sigmoid_derivative_vec();
			
			_errorMatrix[i] = tempErrorMatrix.matrixHadamard(tempDerivativeActivationMatrix);
		}
		
		updateErrorTopDown();
	}
	
	// mathematical functions
	
	public static double calculateCost(Matrix outputMatrix, Matrix desiredOutputMatrix){
		double resultCost = 0.5 * Math.pow((desiredOutputMatrix.matrixSubtract(outputMatrix)).magnitude(), 2.0);
		
		return resultCost;
	}
	
}
