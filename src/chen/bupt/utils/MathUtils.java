package chen.bupt.utils;


import java.util.Date;
import java.util.Random;

public class MathUtils {
  private static final Random r = new Random(new Date().getTime());

  /*
   * 生成一个2项分布，返回表示“成功”了几次
   */
  public static int binomial(int n, double p) {
    int c = 0;
    for (int i = 0; i < n; i++) {
      if (r.nextDouble() < p)
        c++;
    }
    return c;
  }

  public static double sigmod(double x) {
    return 1 / (1 + Math.exp(-x));
  }


  public static void softmax(double[] input) {
    assert input != null : "input in softmax is null";
    double sum = 0.0;
    for (int i = 0; i < input.length; i++) {
      input[i] = Math.exp(input[i]);
      sum += input[i];
    }
    for (int i = 0; i < input.length; i++) {
      input[i] = input[i] / sum;
    }
  }

  public static double gaussian() {
    return r.nextGaussian();
  }

  public static double reconstructError(int[] v0, double[] v1_p) {
    double sum = 0.0;
    for (int i = 0; i < v0.length; i++) {
      sum += (v0[i] - v1_p[i]) * (v0[i] - v1_p[i]);
    }
    return sum;
  }

  public static double uniform(double min, double max) {
    return r.nextDouble() * (max - min) + min;
  }


  /**
   * @param args
   */
  public static void main(String[] args) {
  }

}
