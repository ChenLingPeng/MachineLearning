package chen.bupt.datastructure;

import chen.bupt.utils.MathUtils;

public class Matrix {
  public int row;
  public int column;
  public double[][] weights;


  public Matrix(int row, int column, boolean initRandom) {
    this.row = row;
    this.column = column;
    weights = new double[row][column];
    if (initRandom) {
      init();
    }
  }

  public Matrix(int row, int column) {
    this(row, column, true);
  }

  private void init() {
    for (int i = 0; i < row; i++) {
      for (int j = 0; j < column; j++) {
        this.weights[i][j] = MathUtils.gaussian();
      }
    }
  }

}
