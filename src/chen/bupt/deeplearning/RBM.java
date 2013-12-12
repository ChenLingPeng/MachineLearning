package chen.bupt.deeplearning;

import chen.bupt.utils.MathUtils;

public class RBM {
  // for mini-batch learning
  public int N;

  // hidden_size * visual_size
  public double[][] W;
  public double[] bias_hidden;
  public double[] bias_visual;
  // 下一层单元个数
  public int hidden_size;
  // 输入层单元个数
  public int visual_size;
  // 其他参数还可以设置labmda进行l2的惩罚项

  public RBM(int n, int n_hidden, int n_visual, double[][] w,
             double[] h_bias, double[] v_bias) {
    N = n;
    hidden_size = n_hidden;
    visual_size = n_visual;
    if (w != null) {
      assert w.length == n_hidden && w[0].length == n_visual : "size conflict";
      W = w;
    } else {
      W = new double[n_hidden][n_visual];
      initWeight();
    }
    if (h_bias != null) {
      assert h_bias.length == n_hidden : "size conflict";
      bias_hidden = h_bias;
    } else {
      bias_hidden = new double[n_hidden];
      for (int i = 0; i < n_hidden; i++) {
        bias_hidden[i] = 0.0;
      }
    }

    if (v_bias != null) {
      assert v_bias.length == n_visual : "size conflict";
      bias_visual = v_bias;
    } else {
      bias_visual = new double[n_visual];
      for (int i = 0; i < n_visual; i++) {
        bias_visual[i] = 0.0;
      }
    }
  }

  private void initWeight() {
    for (int i = 0; i < hidden_size; i++) {
      for (int j = 0; j < visual_size; j++) {
        W[i][j] = MathUtils.gaussian();
//                W[i][j] = MathUtils.uniform(0, 1);
      }
    }
  }

  /**
   * 训练技巧参见<A Practical Guide to Training Restricted Boltzmann Machines>
   *
   * @param input        输入样本
   * @param learningRate 学习率
   * @param k            CD-k
   */
  public double contrastiveDivergence(int[] input, double learningRate) {
    assert input != null && input.length == visual_size : "invalid train data";
    double[] p_h0 = new double[this.hidden_size];
    double[] p_h1 = new double[this.hidden_size];
    double[] p_v1 = new double[this.visual_size];
    int[] b_h0 = new int[this.hidden_size];
    int[] b_h1 = new int[this.visual_size];
    int[] b_v1 = new int[this.visual_size];
    sample_h_given_v(input, p_h0, b_h0);
    sample_v_given_h(b_h0, p_v1, b_v1);
    sample_h_given_v(p_v1, p_h1, b_h1); // 用概率而不是状态

    // 根据bengio的 <Greedy Layer-Wise Training of Deep NetWorks>
    for (int j = 0; j < hidden_size; j++) {
      for (int i = 0; i < visual_size; i++) {
        // 这个公式有点疑问
        W[j][i] += learningRate
            * (p_h0[j] * input[i] - p_h1[j] * p_v1[i]);
      }
    }
    for (int j = 0; j < hidden_size; j++) {
      bias_hidden[j] += learningRate * (p_h0[j] - p_h1[j]);
    }
    for (int i = 0; i < visual_size; i++) {
      bias_visual[i] += learningRate * (input[i] - p_v1[i]);
    }
    return MathUtils.reconstructError(input, p_v1);
  }

  public void train(int[][] inputs, int iter, double learningRate) {
    for (int i = 0; i < iter; i++) {
      double error = 0.0;
      for (int j = 0; j < inputs.length; j++) {
        error += this.contrastiveDivergence(inputs[j], learningRate);
      }
      System.out.println("round " + i + " with error " + error);
    }
  }

  public void predict(int[] input) {

  }

  private void sample_h_given_v(int[] v_sample, double[] h_prop,
                                int[] h_sample) {
    for (int i = 0; i < this.hidden_size; i++) {
      h_prop[i] = propup(v_sample, i, bias_hidden[i]);
      h_sample[i] = MathUtils.binomial(1, h_prop[i]);
    }
  }

  private void sample_h_given_v(double[] v_sample, double[] h_prop,
                                int[] h_sample) {
    for (int i = 0; i < this.hidden_size; i++) {
      h_prop[i] = propup(v_sample, i, bias_hidden[i]);
      h_sample[i] = MathUtils.binomial(1, h_prop[i]);
    }
  }

  private void sample_v_given_h(int[] h_sample, double[] v_prop,
                                int[] v_sample) {
    for (int i = 0; i < this.visual_size; i++) {
      v_prop[i] = propdown(h_sample, i, bias_visual[i]);
      v_sample[i] = MathUtils.binomial(1, v_prop[i]);
    }
  }

  private double propup(int[] v_sample, int n, double bias) {
    double sum = 0.0;
    for (int i = 0; i < visual_size; i++) {
      sum += W[n][i] * v_sample[i];
    }
    sum += bias;
    return MathUtils.sigmod(sum);
  }

  private double propup(double[] v_sample, int n, double bias) {
    double sum = 0.0;
    for (int i = 0; i < visual_size; i++) {
      sum += W[n][i] * v_sample[i];
    }
    sum += bias;
    return MathUtils.sigmod(sum);
  }

  private double propdown(int[] h_sample, int n, double bias) {
    double sum = 0.0;
    for (int i = 0; i < hidden_size; i++) {
      sum += W[i][n] * h_sample[i];
    }
    sum += bias;
    return MathUtils.sigmod(sum);
  }

  public void printW() {
    System.out.println("[");
    for (int i = 0; i < W.length; i++) {
      System.out.print("[");
      for (int j = 0; j < W[0].length; j++) {
        System.out.print(W[i][j] + ", ");
      }
      System.out.println("]");
    }
    System.out.print("]");
  }

  /**
   * @param args
   */
  public static void main(String[] args) {
    // TODO Auto-generated method stub
    int[][] inputs = {{1, 1, 1, 0, 0, 0}, {1, 0, 1, 0, 0, 0}, {1, 1, 1, 0, 0, 0}, {0, 0, 1, 1, 1, 0}, {0, 0, 1, 1, 0, 0}, {0, 0, 1, 1, 1, 0}};
    RBM rbm = new RBM(6, 2, 6, null, null, null);
    rbm.printW();
    rbm.train(inputs, 5000, 0.1);
    rbm.printW();
  }

}
