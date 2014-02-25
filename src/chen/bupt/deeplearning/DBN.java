package chen.bupt.deeplearning;

import java.util.ArrayList;
import java.util.List;

public class DBN {
  // for mini-batch learning?
  private int N;
  private int epoch;
  private int[] layerSize;
  // 输入维度
  private int featureSize;
  // 输出维度
  private int labelSize;
  private LinearRegressionLayer lr;
  private List<RBM> rbms;

  public DBN(int n, int[] layerSize, int featureSize, int labelSize, int epoch) {
    assert n > 0 && layerSize != null && featureSize > 0 && labelSize > 0 : "illegal init";
    this.N = n;
    this.layerSize = layerSize;
    this.featureSize = featureSize;
    this.labelSize = labelSize;
    this.lr = new LinearRegressionLayer(layerSize[layerSize.length - 1], labelSize, n);
    rbms = new ArrayList<RBM>(layerSize.length);
    rbms.add(new RBM(n, layerSize[0], featureSize));
    for (int i = 1; i < layerSize.length; i++) {
      rbms.add(new RBM(n, layerSize[i], layerSize[i - 1]));
    }
    this.epoch = epoch;
  }

  public void pretrain(int[][] trainSamples, int[] trainLabels) {
    assert trainSamples[0].length == this.featureSize : "feature size conflict";
    int[][] pre_rbm_data = null;
    for (int layer = 0; layer < layerSize.length; layer++) {
      if (layer == 0) {
        pre_rbm_data = trainSamples;
      }
      this.rbms.get(layer).train(pre_rbm_data, this.epoch, 0.1);
      int[][] rbm_output = new int[trainSamples.length][this.layerSize[layer]];
      for (int i = 0; i < trainSamples.length; i++) {
        this.rbms.get(layer).active_hidden(pre_rbm_data[i], rbm_output[i]);
      }
    }
  }


  /**
   * @param args
   */
  public static void main(String[] args) {
    // TODO Auto-generated method stub

  }

}
