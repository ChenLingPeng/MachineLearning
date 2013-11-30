package ml;

import chen.bupt.utils.MathUtils;

/**
 * ���Է�����
 * 
 * @author chenlingpeng
 * 
 */
public class PerceptronLearningAlgorithm {

    private int featureSize;
    private double bias;
    private double[] weights;

    public PerceptronLearningAlgorithm(int featureSize) {
        this.featureSize = featureSize;
        init();
    }

    private void init() {
        weights = new double[this.featureSize];
        bias = MathUtils.gaussian();
        for (int i = 0; i < this.featureSize; i++) {
            weights[i] = MathUtils.gaussian();
        }
    }

    /**
     * 
     * @param input
     *            ��������
     * @param flag
     *            �������
     */
    public boolean train(double[] input, int flag, double learningRate, double lambda) {
        double tmp = 0.0;
        for (int i = 0; i < this.featureSize; i++) {
            tmp += weights[i] * input[i];
        }
        tmp += bias;
        if (tmp * flag > 0)
            return false;
        for (int i = 0; i < this.featureSize; i++) {
            weights[i] += learningRate * (flag - tmp) * input[i] - lambda*weights[i];
        }
        bias += learningRate * (flag - tmp);
        return true;
    }

    public void printModel(){
        System.out.println("weight: ");
        for(int i=0;i<this.featureSize;i++){
            System.out.println(this.weights[i]+"\t ");
        }
        System.out.println("bias: "+bias);
    }
    
    
    /**
     * @param args
     */
    public static void main(String[] args) {
        PerceptronLearningAlgorithm pla = new PerceptronLearningAlgorithm(2);
        double[][] inputs = { { 0.1, 0.1 }, { 0.1, 0.9 }, { 0.9, 0.1 }, { 0.9, 0.9 } };
        // if 1,-1,-1,1, ���ܽ���ѧϰ����Ϊxor�������Կɷֵ�, ʹ��2������Ե�BP�������ѧϰ
        int[] flags = { -1, -1, -1, 1 };
        int iter = 0;
        boolean flag = true;
        while (flag) {
            flag = false;
            for (int i = 0; i < inputs.length; i++) {
                flag = flag || pla.train(inputs[i], flags[i], 1, 0.001);
            }
            iter++;
        }
        System.out.println("after iter: "+iter+1+" training");
        pla.printModel();
    }

}
