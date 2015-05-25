import java.util.*;
import java.text.*;

public class Matrix {
	
	public static final double DOUBLE_EPSILON = 1E-6;

	private int _numRow;
	private int _numCol;
	private double[][] _grid;
	private int _ptrRow; // used to point to the row of next empty grid
	private int _ptrCol; // used to point to the column of the next empty grid
	
	// default constructor
	public Matrix(int numRow, int numCol){
		_numRow = numRow;
		_numCol = numCol;
		_ptrRow = 0;
		_ptrCol = 0;
		_grid = new double[_numRow][_numCol];
	}
	
	// clone constructor
	public Matrix(Matrix oldMatrix){
		_numRow = oldMatrix.getNumRows();
		_numCol = oldMatrix.getNumCols();
		_grid = oldMatrix.getGrid();
		_ptrRow = oldMatrix.getPtrRow();
		_ptrCol = oldMatrix.getPtrCol();
	}
	
	// accessors
	
	public int getNumRows(){
		return _numRow;
	}
	
	public int getNumCols(){
		return _numCol;
	}
	
	public double[][] getGrid(){
		return _grid;
	}
	
	public double getNextGrid(){
		double tempReturn = _grid[_ptrRow][_ptrCol];
		
		_ptrCol++;
		if (_ptrCol == _numCol){
			_ptrCol = 0;
			_ptrRow++;
			if (_ptrRow == _numRow){
				_ptrRow = 0;
			}
		}
		
		return tempReturn;
	}
	
	public double getGridByPos(int row, int col){
		return _grid[row][col];
	}
	
	public int getPtrRow(){
		return _ptrRow;
	}
	
	public int getPtrCol(){
		return _ptrCol;
	}
	
	// mutators
	
	public void setNumRows(int newNumRows){
		_numRow = newNumRows;
	}
	
	public void setNumCols(int newNumCols){
		_numCol = newNumCols;
	}
	
	public void setNextGrid(double inputValue){
		_grid[_ptrRow][_ptrCol] = inputValue;
		
		_ptrCol++;
		if (_ptrCol == _numCol){
			_ptrCol = 0;
			_ptrRow++;
			if (_ptrRow == _numRow){
				_ptrRow = 0;
			}
		}
	}
	
	public void setGridByPos(int row, int col, double inputValue){
		_grid[row][col] = inputValue;
	}
	
	public void setPtrRow(int newRow){
		_ptrRow = newRow;
	}
	
	public void setPtrCol(int newCol){
		_ptrCol = newCol;
	}
	
	public void scalarMultiply(double multFactor){
		for (int i=0; i<_numRow; i++){
			for (int j=0; j<_numCol; j++){
				_grid[i][j] *= multFactor;
			}
		}
	}
	
	public void verticalMerge(Matrix mergeMatrix){
		if (_numCol != mergeMatrix.getNumCols()){
			return;
		}
		
		double[][] newGrid = new double[_numRow + mergeMatrix.getNumRows()][_numCol];
		
		for (int i=0; i<_numRow + mergeMatrix.getNumRows(); i++){
			for (int j=0; j<_numCol; j++){
				if (i < _numRow){
					newGrid[i][j] = _grid[i][j];
				}
				else {
					newGrid[i][j] = mergeMatrix.getGridByPos(i - _numRow, j);
				}
			}
		}
		
		_numRow += mergeMatrix.getNumRows();
		_grid = newGrid;
	}
	
	// evaluator
	
	public Matrix sigmoid_vec(){
		if (_numCol != 1) return null;
		
		Matrix tempMatrix = new Matrix(_numRow, 1);
		
		for (int i=0; i<_numRow; i++){
			tempMatrix._grid[i][0] = sigmoid(_grid[i][0]);
		}
		
		return tempMatrix;
	}
	
	public Matrix sigmoid_derivative_vec(){
		if (_numCol != 1) return null;
		
		Matrix tempMatrix = new Matrix(_numRow, 1);
		
		for (int i=0; i<_numRow; i++){
			tempMatrix._grid[i][0] = sigmoid_derivative(_grid[i][0]);
		}
		
		return tempMatrix;
	}
	
	public Matrix matrixAdd(Matrix addMatrix){
		if (_numRow != addMatrix.getNumRows() || _numCol != addMatrix.getNumCols()){
			return null;
		}
		
		Matrix resultMatrix = new Matrix(_numRow, _numCol);
		
		for (int i=0; i<_numRow; i++){
			for (int j=0; j<_numCol; j++){
				resultMatrix.setGridByPos(i, j, _grid[i][j] + addMatrix.getGridByPos(i, j));
			}
		}
		
		return resultMatrix;
	}
	
	public Matrix matrixSubtract(Matrix subtractMatrix){
		if (_numRow != subtractMatrix.getNumRows() || _numCol != subtractMatrix.getNumCols()){
			return null;
		}
		
		Matrix resultMatrix = new Matrix(_numRow, _numCol);
		
		for (int i=0; i<_numRow; i++){
			for (int j=0; j<_numCol; j++){
				resultMatrix.setGridByPos(i, j, _grid[i][j] - subtractMatrix.getGridByPos(i, j));
			}
		}
		
		return resultMatrix;
	}
	
	public Matrix matrixMultiply(Matrix multiplierMatrix){
		if (_numCol != multiplierMatrix.getNumRows()){
			return null;
		}
		
		Matrix resultMatrix = new Matrix(_numRow, multiplierMatrix.getNumCols());
		
		for (int i=0; i<resultMatrix.getNumRows(); i++){
			for (int j=0; j<resultMatrix.getNumCols(); j++){
				for (int k=0; k<_numCol; k++){
					resultMatrix.setGridByPos(i, j, resultMatrix.getGridByPos(i, j) + 
											  _grid[i][k] * multiplierMatrix.getGridByPos(k, j));
				}
			}
		}
		
		return resultMatrix;
	}
	
	public Matrix matrixHadamard(Matrix hadamardMatrix){
		if (_numCol != 1 || hadamardMatrix.getNumCols() != 1 || _numRow != hadamardMatrix.getNumRows()){
			return null;
		}
		
		Matrix resultMatrix = new Matrix(_numRow, 1);
		
		for (int i=0; i<_numRow; i++){
			resultMatrix.setGridByPos(i, 0, _grid[i][0] * hadamardMatrix.getGridByPos(i, 0));
		}
		
		return resultMatrix;
	}
	
	public Matrix matrixNormalize(){
		Matrix resultMatrix = new Matrix(_numRow, _numCol);
		
		for (int i=0; i<_numRow; i++){
			for (int j=0; j<_numCol; j++){
				if (_grid[i][j] < 0.5){
					resultMatrix.setNextGrid(0.0);
				}
				else {
					resultMatrix.setNextGrid(1.0);
				}
			}
		}
		
		return resultMatrix;
	}
	
	public boolean equals(Matrix compareMatrix){
		for (int i=0; i<_numRow; i++){
			for (int j=0; j<_numCol; j++){
				if (Math.abs(_grid[i][j] - compareMatrix.getNextGrid()) > DOUBLE_EPSILON){
					return false;
				}
			}
		}
		
		return true;
	}
	
	public Matrix horizontalSlice(int rowNum){
		Matrix resultMatrix = new Matrix(1, _numCol);
		
		for (int i=0; i<_numCol; i++){
			resultMatrix.setGridByPos(0, i, _grid[rowNum][i]);
		}
		
		return resultMatrix;
	}
	
	public Matrix transpose(){
		Matrix resultMatrix = new Matrix(_numCol, _numRow);
		
		for (int i=0; i<_numRow; i++){
			for (int j=0; j<_numCol; j++){
				resultMatrix.setGridByPos(j, i, _grid[i][j]);
			}
		}
		
		return resultMatrix;
	}
	
	public double magnitude(){
		if (_numCol != 1) return Double.POSITIVE_INFINITY;
		
		double resultMagnitude = 0.0;
		
		for (int i=0; i<_numRow; i++){
			resultMagnitude += _grid[i][0] * _grid[i][0];
		}
		
		resultMagnitude = Math.sqrt(resultMagnitude);
		
		return resultMagnitude;
	}
	
	// functionalities
	
	@Override
	public String toString(){
		DecimalFormat df = new DecimalFormat("#.###");
		
		String resultString = new String("\n");
		
		for (int i=0; i<_numRow; i++){
			resultString += "|";
			
			for (int j=0; j<_numCol; j++){
				resultString += " " + df.format(_grid[i][j]);
			}
			
			resultString += " |\n";
		}
		
		return resultString;
	}
	
	public static Matrix randomize(int numRows, int numCols){
		Matrix resultMatrix = new Matrix(numRows, numCols);
		Random rand = new Random();
		
		for (int i=0; i<numRows; i++){
			for (int j=0; j<numCols; j++){
				resultMatrix.setNextGrid(rand.nextDouble());
			}
		}
		
		return resultMatrix;
	}
	
	// mathematical functions
	
	public static double sigmoid(double x){
		return 1.0/(1.0 + Math.exp(-x));
	}
	
	public static double sigmoid_derivative(double x){
		return sigmoid(x) * (1.0 - sigmoid(x));
	}
	
}
