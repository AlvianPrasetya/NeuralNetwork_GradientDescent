import java.util.*;

public class Neuron {

	private int _neuronId; // neuron id w.r.t. the parent layer
	private Neuron[] _inputNeuron;
	private Neuron[] _outputNeuron;
	private Matrix _weightMatrix;
	private double _bias;
	private Matrix _inputActivationMatrix;
	private double _weightedSum;
	private double _outputActivation;
	private double _error;
	
	// default constructor
	public Neuron(int neuronId){
		_neuronId = neuronId;

		Random rand = new Random();
		
		// initialize bias
		_bias = rand.nextGaussian();
		
		System.out.println("  > Initialized neuron of id " + neuronId);
	}
	
	// accessors
	
	public int getNeuronId(){
		return _neuronId;
	}
	
	public Neuron[] getInputNeuron(){
		return _inputNeuron;
	}
	
	public Neuron getInputNeuronById(int neuronId){
		return _inputNeuron[neuronId];
	}
	
	public Neuron[] getOutputNeuron(){
		return _outputNeuron;
	}
	
	public Neuron getOutputNeuronById(int neuronId){
		return _outputNeuron[neuronId];
	}
	
	public Matrix getWeightMatrix(){
		return _weightMatrix;
	}
	
	public double getBias(){
		return _bias;
	}
	
	public double getWeightedSum(){
		return _weightedSum;
	}
	
	public Matrix getInputActivationMatrix(){
		return _inputActivationMatrix;
	}
	
	public double getOutputActivation(){
		return _outputActivation;
	}
	
	public double getError(){
		return _error;
	}
	
	// mutators
	
	public void setInputNeuron(Neuron[] inputNeuron){
		_inputNeuron = inputNeuron;
	}
	
	public void setOutputNeuron(Neuron[] outputNeuron){
		_outputNeuron = outputNeuron;
	}
	
	public void initializeWeightMatrix(int numInputNeurons){
		_weightMatrix = new Matrix(1, numInputNeurons);
		
		Random rand = new Random();
		for (int i=0; i<_weightMatrix.getNumCols(); i++){
			_weightMatrix.setNextGrid(rand.nextGaussian());
		}
	}
	
	public void setWeightMatrix(Matrix newWeightMatrix){
		_weightMatrix = newWeightMatrix;
	}
	
	public void setBias(double newBias){
		_bias = newBias;
	}
	
	public void initializeInputActivationMatrix(int numInputNeurons){
		_inputActivationMatrix = new Matrix(numInputNeurons, 1);
	}
	
	public void setInputActivationMatrix(Matrix newInputActivationMatrix){
		_inputActivationMatrix = newInputActivationMatrix;
	}
	
	public void setWeightedSum(double newWeightedSum){
		_weightedSum = newWeightedSum;
	}
	
	public void setOutputActivation(double newOutputActivation){
		_outputActivation = newOutputActivation;
	}
	
	public void setError(double newError){
		_error = newError;
	}
	
}
