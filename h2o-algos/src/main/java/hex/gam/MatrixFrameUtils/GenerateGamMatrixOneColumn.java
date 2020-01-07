package hex.gam.MatrixFrameUtils;

import hex.DataInfo;
import hex.gam.GAMModel.GAMParameters;
import hex.gam.GAMModel.GAMParameters.BSType;
import hex.gam.GamSplines.CubicRegressionSplines;
import hex.glm.GLMModel.GLMParameters.MissingValuesHandling;
import hex.util.LinearAlgebraUtils.BMulInPlaceTask;
import water.MRTask;
import water.fvec.Chunk;
import water.fvec.Frame;
import water.fvec.NewChunk;
import water.util.ArrayUtils;

import static hex.gam.MatrixFrameUtils.GamUtils.locateBin;

public class GenerateGamMatrixOneColumn extends MRTask<GenerateGamMatrixOneColumn> {
  BSType _splineType;
  int _numKnots;      // number of knots
  double[][] _bInvD;  // store inv(B)*D
  double _mean;
  double _oneOSigma;
  Frame _gamX;
  boolean _centerGAM; // center model matrix X generated
  double[] _u; // store transpose(X)*1
  public double[][] _ZTransp;  // store Z matrix transpose
  public double[][] _penaltyMat;  // store penalty matrix
  
  public GenerateGamMatrixOneColumn(BSType splineType, int numKnots, double[] knots, Frame gamx, boolean standardize,
                                    boolean centerGam, double penalty_scale) {
    _splineType = splineType;
    _numKnots = numKnots;
    _mean = standardize?gamx.vec(0).mean():0;
    _oneOSigma = 1.0/(standardize?gamx.vec(0).sigma():1);
    CubicRegressionSplines crSplines = new CubicRegressionSplines(numKnots, knots, gamx.vec(0).max(), gamx.vec(0).min());
    _bInvD = crSplines.gen_BIndvD(crSplines._hj);
    _penaltyMat = crSplines.gen_penalty_matrix(ArrayUtils.mult(crSplines._hj, penalty_scale), _bInvD);
    _gamX = gamx;
    _centerGAM = centerGam;
  }

  @Override
  public void map(Chunk[] chk, NewChunk[] newGamCols) {
    _u = new double[_numKnots];
    int chunkRows = chk[0].len(); // number of rows in chunk
    CubicRegressionSplines crSplines = new CubicRegressionSplines(_numKnots, null, _gamX.vec(0).max(), 
            _gamX.vec(0).min());
    double[] basisVals = new double[_numKnots];
    for (int rowIndex=0; rowIndex < chunkRows; rowIndex++) {
     // find index of knot bin where row value belongs to
     double xval = chk[0].atd(rowIndex);
     int binIndex = locateBin(xval,crSplines._knots); // location to update
      // update from F matrix F matrix = [0;invB*D;0] and c functions
      updateFMatrixCFunc(basisVals, xval, binIndex, crSplines, _bInvD);
      // update from a+ and a- functions
      updateAFunc(basisVals, xval, binIndex, crSplines);
      // copy updates to the newChunk row
      for (int colIndex = 0; colIndex < _numKnots; colIndex++) {
        newGamCols[colIndex].addNum(basisVals[colIndex]);
        if (_centerGAM)
          _u[colIndex] += basisVals[colIndex];
      }
    }
  }
  
  @Override
  public void reduce(GenerateGamMatrixOneColumn other) {
    if (_centerGAM) { // only perform reduce during de-centerization
      ArrayUtils.add(_u, other._u);
    }
  }
  
  @Override
  public void postGlobal() {
    if (_centerGAM) { // generate Z matrix
      _ZTransp = new double[_numKnots-1][_numKnots];
      double mag = 0.0;
      for (int index=0; index < _numKnots; index++) // calculate mag and 1/(mag*mag)
        mag += _u[index]*_u[index];
      double twoOmagSq = 2.0/mag;
      mag = Math.sqrt(mag);
      _u[0] -= mag; // form a = u-v and stored back in _u
      for (int rowIndex=0; rowIndex < _numKnots; rowIndex++) {  // form Z matrix transpose here
        for (int colIndex = 1; colIndex < _numKnots; colIndex++) {
          int trueColIndex = colIndex-1;
          _ZTransp[trueColIndex][rowIndex] = colIndex==rowIndex?1:0-_u[colIndex]*_u[rowIndex]*twoOmagSq;
        }
      }
    }
  }
  
  public static void updateAFunc(double[] basisVals, double xval, int binIndex, CubicRegressionSplines splines) {
    int jp1 = binIndex+1;
    basisVals[binIndex] += splines.gen_a_m_j(splines._knots[jp1], xval, splines._hj[binIndex]);
    basisVals[jp1] += splines.gen_a_p_j(splines._knots[binIndex], xval, splines._hj[binIndex]);
  }
  
  public static void updateFMatrixCFunc(double[] basisVals, double xval, int binIndex, CubicRegressionSplines splines,
                                        double[][] binvD) {
    int numKnots = basisVals.length;
    int matSize = binvD.length;
    int jp1 = binIndex+1;
    double cmj = splines.gen_c_m_j(splines._knots[jp1], xval, splines._hj[binIndex]);
    double cpj = splines.gen_c_p_j(splines._knots[binIndex], xval, splines._hj[binIndex]);
    int binIndexM1 = binIndex-1;
    for (int index=0; index < numKnots; index++) {
      if (binIndex == 0) {  // only one part
        basisVals[index] = binvD[binIndex][index] * cpj;
      } else if (binIndex >= matSize) { // update only one part
        basisVals[index] = binvD[binIndexM1][index] * cmj;
      } else { // binIndex > 0 and binIndex < matSize
        basisVals[index] = binvD[binIndexM1][index] * cmj+binvD[binIndex][index] * cpj;
      }
    }
  }
  
  public Frame de_centralize_frame(Frame fr, String colNameStart, GAMParameters parms) {
    int numCols = fr.numCols();
    int ncolExp = numCols-1;
    DataInfo frInfo = new DataInfo(fr, null, 0, false,  DataInfo.TransformType.NONE, DataInfo.TransformType.NONE,
            MissingValuesHandling.Skip == parms._missing_values_handling, 
            (parms._missing_values_handling == MissingValuesHandling.MeanImputation) || 
                    (parms._missing_values_handling == MissingValuesHandling.PlugValues), parms.makeImputer(), 
            false, false, false, false, null);
    for (int index=0; index < ncolExp; index++) {
      fr.add(colNameStart+"_"+index, fr.anyVec().makeZero()); // add numCols-1 columns to fr
    }
    BMulInPlaceTask mulTask = new BMulInPlaceTask(frInfo, _ZTransp, numCols).doAll(fr);
    for (int index=0; index < numCols; index++) { // remove the original gam columns
      fr.remove(0).remove();
    }
    return fr;
  }

 
}


