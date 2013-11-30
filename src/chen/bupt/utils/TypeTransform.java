package chen.bupt.utils;

import java.util.Vector;

/**
 * Created with IntelliJ IDEA.
 * User: chenlingpeng
 * Date: 13-11-30
 * Time: 下午6:01
 * To change this template use File | Settings | File Templates.
 */
public class TypeTransform {
  public static Vector<Double> array2vec(double[] a){
    Vector<Double> res = new Vector<Double>();
    for(int i=0;i<a.length;i++){
      res.add(a[i]);
    }
    return res;
  }
}
