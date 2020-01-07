package hex.gam.MatrixFrameUtils;

import hex.Model;
import hex.gam.GAMModel.GAMParameters;
import hex.glm.GLMModel.GLMParameters;
import jsr166y.ForkJoinTask;
import jsr166y.RecursiveAction;
import water.DKV;
import water.Key;
import water.MemoryManager;
import water.Scope;
import water.fvec.Frame;
import water.fvec.Vec;
import water.util.ArrayUtils;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.List;

public class GamUtils {
  /***
   * Allocate 3D array to store various info.
   * @param num2DArrays
   * @param parms
   * @param fileMode: 0: allocate for transpose(Z), 1: allocate for S, 2: allocate for t(Z)*S*Z
   * @return
   */
  public static double[][][] allocate3DArray(int num2DArrays, GAMParameters parms, int fileMode) {
    double[][][] array3D = new double[num2DArrays][][];
    for (int frameIdx = 0; frameIdx < num2DArrays; frameIdx++) {
      int numKnots = parms._k[frameIdx];
      switch (fileMode) {
        case 0: array3D[frameIdx] = MemoryManager.malloc8d(numKnots-1, numKnots); break;
        case 1: array3D[frameIdx] = MemoryManager.malloc8d(numKnots, numKnots); break;
        case 2: array3D[frameIdx] = MemoryManager.malloc8d(numKnots-1, numKnots-1); break;
        default: throw new IllegalArgumentException("fileMode can only be 0, 1 or 2.");
      }
    }
    return array3D;
  }

  public static boolean equalColNames(String[] name1, String[] standardN, String response_column) {
    boolean equalNames = ArrayUtils.contains(name1, response_column)?name1.length==standardN.length:
            (name1.length+1)==standardN.length;
    if (equalNames) { // number of columns are correct but with the same column names and column types?
      for (String name : name1) {
        if (!ArrayUtils.contains(standardN, name))
          return false;
      }
      return true;
    } else
      return equalNames;
  }
  
  public static void copy2DArray(double[][] src_array, double[][] dest_array) {
    int numRows = src_array.length;
    for (int colIdx = 0; colIdx < numRows; colIdx++) { // save zMatrix for debugging purposes or later scoring on training dataset
      System.arraycopy(src_array[colIdx], 0, dest_array[colIdx], 0,
              src_array[colIdx].length);
    }
  }

  /***
   * from Stack overflow by dfa/user1079877
   * @param fields
   * @param type
   * @return
   */
  public static List<Field> getAllFields(List<Field> fields, Class<?> type) {
    fields.addAll(Arrays.asList(type.getDeclaredFields()));

    if (type.getSuperclass() != null) {
      getAllFields(fields, type.getSuperclass());
    }

    return fields;
  }

  public static GLMParameters copyGAMParams2GLMParams(GAMParameters parms, Frame trainData) {
    GLMParameters glmParam = new GLMParameters();
    Field[] field1 = GAMParameters.class.getDeclaredFields();
    setParamField(parms, glmParam, false, field1);
    Field[] field2 = Model.Parameters.class.getDeclaredFields();
    setParamField(parms, glmParam, true, field2);
    glmParam._train = trainData._key;
    return glmParam;
  }
  
  public static void setParamField(GAMParameters parms, GLMParameters glmParam, boolean superClassParams, Field[] gamFields) {
    // assign relevant GAMParameter fields to GLMParameter fields
    List<String> gamOnlyList = Arrays.asList(new String[]{"_k", "_gam_X", "_bs", "_scale", "_train", "_saveZMatrix", 
            "_saveGamCols", "_savePenaltyMat", "_ignored_columns"});
    for (Field oneField : gamFields) {
      if (oneField.getName().equals("_repsonse_column"))
        System.out.println("whoe");
      try {
        if (!gamOnlyList.contains(oneField.getName())) {
          Field glmField = superClassParams?glmParam.getClass().getSuperclass().getDeclaredField(oneField.getName())
                  :glmParam.getClass().getDeclaredField(oneField.getName());
          glmField.set(glmParam, oneField.get(parms));
        }
      } catch (IllegalAccessException e) { // suppress error printing
        ;
      } catch (NoSuchFieldException e) {
        ;
      }
    }
  }

  public static int locateBin(double xval, double[] knots) {
    if (xval <= knots[0])  //small short cut
      return 0;
    int highIndex = knots.length-1;
    if (xval >= knots[highIndex]) // small short cut
      return (highIndex-1);

    int binIndex = 0;
    int count = 0;
    int numBins = knots.length;
    int lowIndex = 0;

    while (count < numBins) {
      int tryBin = (int) Math.floor((highIndex+lowIndex)*0.5);
      if ((xval >= knots[tryBin]) && (xval < knots[tryBin+1]))
        return tryBin;
      else if (xval > knots[tryBin])
        lowIndex = tryBin;
      else if (xval < knots[tryBin])
        highIndex = tryBin;

      count++;
    }
    return binIndex;
  }

  public static Frame addGAM2Train(GAMParameters parms, Frame orig, Frame _train, double[][][] zTranspose, 
                                   double[][][] penalty_mat, String[][] gamColnames, String[][] gamColnamesDecenter, 
                                   boolean modelBuilding, boolean centerGAM) {
    int numGamFrame = parms._gam_X.length;
    Key<Frame>[] gamFramesKey = new Key[numGamFrame];  // store the Frame keys of generated GAM column
    RecursiveAction[] generateGamColumn = new RecursiveAction[numGamFrame];
    for (int index = 0; index < numGamFrame; index++) {
      final Frame predictVec = new Frame(new String[]{parms._gam_X[index]}, new Vec[]{orig.vec(parms._gam_X[index])});  // extract the vector to work on
      final int numKnots = parms._k[index];  // grab number of knots to generate
      final double scale = parms._scale == null ? 1.0 : parms._scale[index];
      final GAMParameters.BSType splineType = parms._bs[index];
      final int frameIndex = index;
      final String[] newColNames = new String[numKnots];
      for (int colIndex = 0; colIndex < numKnots; colIndex++) {
        newColNames[colIndex] = parms._gam_X[index] + "_" + splineType.toString() + "_" + colIndex;
      }
      gamColnames[frameIndex] = new String[numKnots];
      System.arraycopy(newColNames, 0, gamColnames[frameIndex], 0, numKnots);
      generateGamColumn[frameIndex] = new RecursiveAction() {
        @Override
        protected void compute() {
          GenerateGamMatrixOneColumn genOneGamCol = new GenerateGamMatrixOneColumn(splineType, numKnots, null, predictVec,
                  parms._standardize, centerGAM, scale).doAll(numKnots, Vec.T_NUM, predictVec);
          Frame oneAugmentedColumn = genOneGamCol.outputFrame(Key.make(), newColNames,
                  null);
          if (modelBuilding) {  // z and penalty matrices are only needed during model building
            if (centerGAM) {  // calculate z transpose
              oneAugmentedColumn = genOneGamCol.de_centralize_frame(oneAugmentedColumn,
                      predictVec.name(0) + "_" + splineType.toString() + "_decenter_", parms);
              GamUtils.copy2DArray(genOneGamCol._ZTransp, zTranspose[frameIndex]); // copy transpose(Z)
              double[][] transformedPenalty = ArrayUtils.multArrArr(ArrayUtils.multArrArr(genOneGamCol._ZTransp,
                      genOneGamCol._penaltyMat), ArrayUtils.transpose(genOneGamCol._ZTransp));  // transform penalty as zt*S*z
              GamUtils.copy2DArray(transformedPenalty, penalty_mat[frameIndex]);
            } else
              GamUtils.copy2DArray(genOneGamCol._penaltyMat, penalty_mat[frameIndex]); // copy penalty matrix
          }
          gamFramesKey[frameIndex] = oneAugmentedColumn._key;
          DKV.put(oneAugmentedColumn);
        }
      };
    }
    ForkJoinTask.invokeAll(generateGamColumn);
    for (int frameInd = 0; frameInd < numGamFrame; frameInd++) {  // append the augmented columns to _train
      Frame gamFrame = gamFramesKey[frameInd].get();
      if (centerGAM) {
        gamColnamesDecenter[frameInd] = new String[gamFrame.names().length];
        System.arraycopy(gamFrame.names(), 0, gamColnamesDecenter[frameInd], 0, gamFrame.names().length);
      }
      _train.add(gamFrame.names(), gamFrame.removeAll());
      Scope.track(gamFrame);
    }
    return _train;
  }
}
