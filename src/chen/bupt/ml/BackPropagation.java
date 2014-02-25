package chen.bupt.ml;

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import chen.bupt.datastructure.Matrix;
import chen.bupt.utils.MathUtils;
import chen.bupt.utils.TypeTransform;

public class BackPropagation {
  private int layer; //隐藏层+1
  private int[] layersSize; // 各层神经元数目, size=layer+1
  private List<Matrix> layers;

  public BackPropagation(int layer, int[] layersSize) {
    this.layer = layer;
    this.layersSize = layersSize;
    assert layersSize.length == layer + 1 : "layer size not right";
    layers = new ArrayList<Matrix>();
    for (int i = 0; i < layer; i++) {
      Matrix matrix = new Matrix(layersSize[i + 1], layersSize[i] + 1); // add 1 for bias
      layers.add(matrix);
    }
  }


  /**
   * @param input
   * @param output
   * @param learningRate 学习率
   * @param lambda       l2惩罚因子
   * @param numT         mini-batch learning
   * @return
   */
  public double train(double[] input, double[] output, double learningRate, double lambda, int numT) {

    assert input.length == layersSize[0];
    assert output.length == layersSize[layer];
    double[] a_pre = input;
    // 存储feed-forward阶段的各个层上单元的输出
    List<Vector<Double>> X = new ArrayList<Vector<Double>>();
    X.add(TypeTransform.array2vec(input));
    // feed-forward
    for (int i = 0; i < layer; i++) {
      double[] tmp = new double[layersSize[i + 1]];
      for (int j = 0; j < layersSize[i + 1]; j++) {
        double sum = 0.0;
        assert a_pre.length == layersSize[i];
        for (int k = 0; k < layersSize[i]; k++) {
          sum += a_pre[k] * layers.get(i).weights[j][k];
        }
        sum += layers.get(i).weights[j][layersSize[i]];
        tmp[j] = MathUtils.sigmod(sum);
      }
      a_pre = tmp;
      X.add(TypeTransform.array2vec(a_pre));
    }
//    System.out.println(numT+" origin: "+output[0]+","+output[1]);
//    System.out.println(numT+" output: "+a_pre[0]+","+a_pre[1]);
    assert X.size() == layer + 1 : "X not right";

    //back-propagation
    // for last layer
    double[] delta_post = new double[layersSize[layer]];
    for (int i = 0; i < layersSize[layer]; i++) {
      delta_post[i] = a_pre[i] * (1 - a_pre[i]) * (output[i] - a_pre[i]); //省略负号，更新时用+
    }
    for (int i = 0; i < layersSize[layer]; i++) {
      for (int j = 0; j < layersSize[layer - 1]; j++) {
        layers.get(layer - 1).weights[i][j] += learningRate * X.get(layer - 1).get(j) * delta_post[i];
      }
      layers.get(layer - 1).weights[i][layersSize[layer - 1]] += learningRate * delta_post[i];
    }
    // for hidden layer
    for (int i = layer - 1; i > 0; i--) {
      double[] delta_now = new double[layersSize[i]];
      for (int j = 0; j < layersSize[i]; j++) {
        for (int k = 0; k < layersSize[i + 1]; k++) {
          delta_now[j] = delta_post[k];
          delta_now[j] += layers.get(i).weights[k][j] * delta_post[k];
        }
        delta_now[j] *= X.get(i).get(j) * (1 - X.get(i).get(j));
      }
      for (int j = 0; j < layersSize[i]; j++) {
        for (int k = 0; k < layersSize[i - 1]; k++) {
          layers.get(i - 1).weights[j][k] += learningRate * X.get(i - 1).get(k) * delta_now[j];
        }
        layers.get(i - 1).weights[j][layersSize[i - 1]] += learningRate * delta_now[j];
      }
      delta_post = delta_now;
    }
    return (a_pre[0] - output[0]) * (a_pre[0] - output[0]) + (a_pre[1] - output[1]) * (a_pre[1] - output[1]);
  }

  public void predict(double[] input, double[] exp) {
    // TODO
    assert input.length == layersSize[0];
    double[] a_pre = input;
    // 存储feed-forward阶段的各个层上单元的输出
    // feed-forward
    for (int i = 0; i < layer; i++) {
      double[] tmp = new double[layers.get(i).row];
      for (int j = 0; j < layers.get(i).row; j++) {
        double sum = 0.0;
        assert a_pre.length == layers.get(i).column;
        for (int k = 0; k < layers.get(i).column - 1; k++) {
          sum += a_pre[k] * layers.get(i).weights[j][k];
        }
        sum += layers.get(i).weights[j][layers.get(i).column - 1];
        tmp[j] = MathUtils.sigmod(sum);
      }
      a_pre = tmp;
    }
    System.out.println("\nresult");
    for (int i = 0; i < layersSize[layer]; i++) {
      System.out.printf("%.2f\t", a_pre[i]);
    }
    System.out.println("\nexp");
    for (int i = 0; i < layersSize[layer]; i++) {
      System.out.print(exp[i] + "\t");
    }
  }

  /**
   * @param args
   */
  public static void main(String[] args) {
    double[][] inputs = {{0.1, 0.1}, {0.1, 0.9}, {0.9, 0.1}, {0.9, 0.9}};
//    double[][] inputs = {{0.1, 0.1,0.1}};
//    int[] flags = {1, -1, -1, 1};
    double[][] outputs = {{0.9, 0.1}, {0.1, 0.9}, {0.1, 0.9}, {0.9, 0.1}};
//    double[][] outputs={{0.9},{0.1},{0.1},{0.9}};
//    double[] test = {0.1, 0.1};
    double threshlod = 0.01;
    int maxIter = 1000;
    int[] layerSize = {2, 6, 2};
    BackPropagation bp = new BackPropagation(layerSize.length - 1, layerSize);
    // 迭代次数貌似跟weights的初始化关系比较大。。。
    for (int i = 0; i < maxIter; i++) {
      double error = 0;
      for (int j = 0; j < inputs.length; j++) {
        error += bp.train(inputs[j], outputs[j], 0.9, 0.001, j);
      }
      System.out.println(i + ", " + error);
      if (error < threshlod) break;
    }
    for (int i = 0; i < inputs.length; i++) {
      bp.predict(inputs[i], outputs[i]);
    }
  }

}
