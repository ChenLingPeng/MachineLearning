package chen.bupt.deeplearning;


import chen.bupt.utils.MathUtils;

public class LinearRegressionLayer {
  // for mini-batch learning
  public int N;

  // hidden_size * visual_size
  public double[][] W;
  public double[] bias;

  public int feature_size;

  public int label_size;


  public LinearRegressionLayer(int n_feature, int n_label, int n, double[][] w, double[] bias) {
    feature_size = n_feature;
    label_size = n_label;
    N = n;
    if (w != null) {
      W = w;
    } else {
      W = new double[n_label][n_feature];
      initWeight();
    }
    if (bias != null) {
      this.bias = bias;
    } else {
      this.bias = new double[n_label];
      for (int i = 0; i < n_label; i++) {
        this.bias[i] = 0.0;
      }
    }
  }

  public LinearRegressionLayer(int feature_size, int label_size, int n) {
    this(feature_size, label_size, n, null, null);
  }

  private void initWeight() {
    for (int i = 0; i < label_size; i++) {
      for (int j = 0; j < feature_size; j++) {
        W[i][j] = MathUtils.gaussian();
      }
    }
  }

  public double train(int[] input, double learningRate, int[] label) {
    double[] p_label = new double[this.label_size];
    for (int i = 0; i < this.label_size; i++) {
      p_label[i] = 0;
      for (int j = 0; j < this.feature_size; j++) {
        p_label[i] += W[i][j] * input[j];
      }
      p_label[i] += bias[i];
    }
    MathUtils.softmax(p_label);
    // 根据
    // http://deeplearning.stanford.edu/wiki/index.php/Softmax%E5%9B%9E%E5%BD%92
    // 进行更新
    for (int i = 0; i < this.label_size; i++) {
      double delta = label[i] - p_label[i];
      for (int j = 0; j < this.feature_size; j++) {
        W[i][j] += learningRate * input[j] * delta;
      }
      bias[i] += learningRate * delta;
    }
    return MathUtils.reconstructError(label, p_label);
  }

  public double predictError(double[] prob, int label) {
    double sum = 0.0;
    for (int i = 0; i < prob.length; i++) {
      if (i == label) {
        sum += (1 - prob[i]) * (1 - prob[i]);
      } else {
        sum += prob[i] * prob[i];
      }
    }
    return sum;
  }

  /**
   * @param args
   */
  public static void main(String[] args) {
    int[][] train_X = {{1, 1, 1, 0, 0, 0}, {1, 0, 1, 0, 0, 0},
        {1, 1, 1, 0, 0, 0}, {0, 0, 1, 1, 1, 0},
        {0, 0, 1, 1, 0, 0}, {0, 0, 1, 1, 1, 0}};

    int[][] train_Y = {{1, 0}, {1, 0}, {1, 0}, {0, 1}, {0, 1},
        {0, 1}};
    LinearRegressionLayer lr = new LinearRegressionLayer(6, 2, 1, null, null);
    for (int i = 0; i < 1000; i++) {
      double error = 0.0;
      for (int j = 0; j < train_X.length; j++) {
        error += lr.train(train_X[j], 0.1, train_Y[j]);
      }
      System.out.println("iter " + i + " error: " + error);
    }
  }

}
