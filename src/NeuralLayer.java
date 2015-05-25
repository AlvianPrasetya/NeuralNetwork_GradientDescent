public class NeuralLayer {

	private int _layerId; // layer id w.r.t. the parent network
	private int _numNeuron;
	private NeuralLayer _inputLayer;
	private NeuralLayer _outputLayer;
	private Neuron[] _neuron;
	private Matrix _weightMatrix;
	private Matrix _biasMatrix;
	private Matrix _inputActivationMatrix;
	private Matrix _weightedSumMatrix;
	private Matrix _outputActivationMatrix;
	private Matrix _errorMatrix;
	
	// default constructor
	public NeuralLayer(int layerId, int numNeuron){
		_layerId = layerId;
		_numNeuron = numNeuron;
		
		if (_layerId != 0){
			_biasMatrix = new Matrix(_numNeuron, 1);
			_weightedSumMatrix = new Matrix(_numNeuron, 1);
			_errorMatrix = new Matrix(_numNeuron, 1);
		}
		
		_inputActivationMatrix = new Matrix(_numNeuron, 1);
		_outputActivationMatrix = new Matrix(_numNeuron, 1);
		
		// initialize neurons
		_neuron = new Neuron[_numNeuron];
		for (int i=0; i<_numNeuron; i++){
			_neuron[i] = new Neuron(i);
		}
		
		System.out.println(" > Initialized neural layer of " + _numNeuron + " neuron(s)");
	}
	
	// accessors
	
	public int getLayerId(){
		return _layerId;
	}
	
	public int getNumNeuron(){
		return _numNeuron;
	}
	
	public NeuralLayer getInputLayer(){
		return _inputLayer;
	}
	
	public NeuralLayer getOutputLayer(){
		return _outputLayer;
	}
	
	public Neuron[] getNeuronList(){
		return _neuron;
	}
	
	public Neuron getNeuronById(int neuronId){
		return _neuron[neuronId];
	}
	
	public Matrix getWeightMatrix(){
		return _weightMatrix;
	}
	
	public Matrix getBiasMatrix(){
		return _biasMatrix;
	}
	
	public Matrix getWeightedSumMatrix(){
		return _weightedSumMatrix;
	}
	
	public Matrix getInputActivationMatrix(){
		return _inputActivationMatrix;
	}
	
	public Matrix getOutputActivationMatrix(){
		return _outputActivationMatrix;
	}
	
	public Matrix getErrorMatrix(){
		return _errorMatrix;
	}
	
	// mutators
	
	public void setInputLayer(NeuralLayer inputLayer){
		_inputLayer = inputLayer;
		
		if (_layerId != 0){ // if not first/input layer
			// initialize _thisLayer Neuron' _inputNeuron
			for (int i=0; i<_numNeuron; i++){
				_neuron[i].setInputNeuron(_inputLayer.getNeuronList());
			}
			
			// initialize _inputLayer Neuron' _outputNeuron
			for (int i=0; i<_inputLayer.getNumNeuron(); i++){
				_inputLayer.getNeuronById(i).setOutputNeuron(_neuron);
			}
			
			// initialize _thisLayer Neuron' _weightMatrix
			for (int i=0; i<_numNeuron; i++){
				_neuron[i].initializeWeightMatrix(_inputLayer.getNumNeuron());
			}
			
			// initialize _thisLayer Neuron's _inputActivationMatrix
			for (int i=0; i<_numNeuron; i++){
				_neuron[i].initializeInputActivationMatrix(_inputLayer.getNumNeuron());
			}
		}
	}
	
	public void setOutputLayer(NeuralLayer outputLayer){
		_outputLayer = outputLayer;
	}
	
	public void initializeWeightMatrix(){
		_weightMatrix = new Matrix(_numNeuron, _inputLayer.getNumNeuron());
	}

	public void setWeightMatrix(Matrix newWeightMatrix){
		_weightMatrix = newWeightMatrix;
	}
	
	public void setBiasMatrix(Matrix newBiasMatrix){
		_biasMatrix = newBiasMatrix;
	}
	
	public void setInputActivationMatrix(Matrix newInputActivationMatrix){
		_inputActivationMatrix = newInputActivationMatrix;
	}
	
	public void setWeightedSumMatrix(Matrix newWeightedSumMatrix){
		_weightedSumMatrix = newWeightedSumMatrix;
	}
	
	public void setOutputActivationMatrix(Matrix newOutputActivationMatrix){
		_outputActivationMatrix = newOutputActivationMatrix;
	}
	
	public void setErrorMatrix(Matrix newErrorMatrix){
		_errorMatrix = newErrorMatrix;
	}
	
	// synchronize functions
	
	public void collateWeight(){
		Matrix tempWeightMatrix = new Matrix(_neuron[0].getWeightMatrix());
		for (int i=1; i<_numNeuron; i++){
			tempWeightMatrix.verticalMerge(_neuron[i].getWeightMatrix());
		}
		
		_weightMatrix = tempWeightMatrix;
	}
	
	public void disperseWeight(){
		for (int i=0; i<_numNeuron; i++){
			_neuron[i].setWeightMatrix(_weightMatrix.horizontalSlice(0));
		}
	}
	
	public void collateBias(){
		Matrix tempBiasMatrix = new Matrix(_numNeuron, 1);
		for (int i=0; i<_numNeuron; i++){
			tempBiasMatrix.setNextGrid(_neuron[i].getBias());
		}
		
		_biasMatrix = tempBiasMatrix;
	}
	
	public void disperseBias(){
		for (int i=0; i<_numNeuron; i++){
			_neuron[i].setBias(_biasMatrix.getNextGrid());
		}
	}
	
	public void collateInputActivation(){
		_inputActivationMatrix = _neuron[0].getInputActivationMatrix();
	}
	
	public void disperseInputActivation(){
		for (int i=0; i<_numNeuron; i++){
			_neuron[i].setInputActivationMatrix(_inputActivationMatrix);
		}
	}
	
	public void collateWeightedSum(){
		Matrix tempWeightedSumMatrix = new Matrix(_numNeuron, 1);
		for (int i=0; i<_numNeuron; i++){
			tempWeightedSumMatrix.setNextGrid(_neuron[i].getWeightedSum());
		}
		
		_weightedSumMatrix = tempWeightedSumMatrix;
	}
	
	public void disperseWeightedSum(){
		for (int i=0; i<_numNeuron; i++){
			_neuron[i].setWeightedSum(_weightedSumMatrix.getNextGrid());
		}
	}
	
	public void collateOutputActivation(){
		Matrix tempOutputActivationMatrix = new Matrix(_numNeuron, 1);
		for (int i=0; i<_numNeuron; i++){
			tempOutputActivationMatrix.setNextGrid(_neuron[i].getOutputActivation());
		}
		
		_outputActivationMatrix = tempOutputActivationMatrix;
	}
	
	public void disperseOutputActivation(){
		for (int i=0; i<_numNeuron; i++){
			_neuron[i].setOutputActivation(_outputActivationMatrix.getNextGrid());
		}
	}
	
	public void collateError(){
		Matrix tempErrorMatrix = new Matrix(_numNeuron, 1);
		for (int i=0; i<_numNeuron; i++){
			tempErrorMatrix.setNextGrid(_neuron[i].getError());
		}
		
		_errorMatrix = tempErrorMatrix;
	}
	
	public void disperseError(){
		for (int i=0; i<_numNeuron; i++){
			_neuron[i].setError(_errorMatrix.getNextGrid());
		}
	}
	
}
