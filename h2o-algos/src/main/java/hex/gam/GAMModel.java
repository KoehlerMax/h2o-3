package hex.gam;

import hex.*;
import hex.deeplearning.DeepLearningModel;
import hex.glm.GLM;
import hex.glm.GLMModel.GLMParameters.*;
import water.*;
import water.fvec.Frame;
import water.udf.CFuncRef;
import water.util.ArrayUtils;
import water.util.Log;

import java.io.Serializable;
import java.util.Arrays;

import static hex.gam.MatrixFrameUtils.GamUtils.addGAM2Train;
import static hex.gam.MatrixFrameUtils.GamUtils.equalColNames;

public class GAMModel extends Model<GAMModel, GAMModel.GAMParameters, GAMModel.GAMModelOutput> {
  public boolean _centerGAM = false;  // true if centering is needed for training dataset
  public String[][] _gamColnames; // store column names only for GAM columns
  public String[][] _gamColnamesDecenter; // store column names only for GAM columns after decentering
  
  @Override public ModelMetrics.MetricBuilder makeMetricBuilder(String[] domain) {
    return null;
    //return new MetricBuilderGAM(domain);
  }

  public GAMModel(Key<GAMModel> selfKey, GAMParameters parms, GAMModelOutput output) {
    super(selfKey, parms, output);
    assert(Arrays.equals(_key._kb, selfKey._kb));
  }

  ModelMetricsSupervised makeModelMetrics(Frame origFr, Frame adaptFr, String description) {
    Log.info("Making metrics: " + description);
    ModelMetrics.MetricBuilder mb = scoreMetrics(adaptFr);
    ModelMetricsSupervised mm = (ModelMetricsSupervised) mb.makeModelMetrics(this, origFr, adaptFr, null);
    mm._description = description;
    return mm;
  }

  @SuppressWarnings("WeakerAccess")
  public static class GAMParameters extends Model.Parameters {
    // the following parameters will be passed to GLM algos
    public boolean _standardize = false; // pass to GLM algo
    public Family _family;
    public Link _link; 
    public Solver _solver = Solver.AUTO;
    public double _tweedie_variance_power;
    public double _tweedie_link_power;
    public double _theta; // 1/k and is used by negative binomial distribution only
    public double _invTheta;
    public double [] _alpha;
    public double [] _lambda;
    public Serializable _missing_values_handling = MissingValuesHandling.MeanImputation;
    public double _prior = -1;
    public boolean _lambda_search = false;
    public int _nlambdas = -1;
    public boolean _non_negative = false;
    public boolean _exactLambdas = false;
    public double _lambda_min_ratio = -1; // special
    public boolean _use_all_factor_levels = false;
    public int _max_iterations = -1;
    public boolean _intercept = true;
    public double _beta_epsilon = 1e-4;
    public double _objective_epsilon = -1;
    public double _gradient_epsilon = -1;
    public double _obj_reg = -1;
    public boolean _compute_p_values = false;
    public boolean _remove_collinear_columns = false;
    public String[] _interactions=null;
    public StringPair[] _interaction_pairs=null;
    public boolean _early_stopping = true;
    public Key<Frame> _beta_constraints = null;
    public Key<Frame> _plug_values = null;
    // internal parameter, handle with care. GLM will stop when there is more than this number of active predictors (after strong rule screening)
    public int _max_active_predictors = -1;
    public boolean _stdOverride; // standardization override by beta constraints

    // the following parameters are for GAM
    public int[] _k; // array storing number of knots per basis function
    public String[] _gam_X; // array storing which predictor columns are needed
    public BSType[] _bs; // array storing basis functions, only support cr for now
    public double[] _scale;  // array storing scaling values to control wriggliness of fit
    public GLMType _glmType = GLMType.gam; // internal parameter
    public boolean _saveZMatrix = false;  // if asserted will save Z matrix
    public boolean _saveGamCols = false;  // if true will save the keys to gam Columns only
    public boolean _savePenaltyMat = false; // if true will save penalty matrices as tripple array
    
    public String algoName() { return "GAM"; }
    public String fullName() { return "General Additive Model"; }
    public String javaName() { return GAMModel.class.getName(); }

    @Override
    public long progressUnits() {
      return 1;
    }

    public long _seed = -1;
    
    public enum BSType {
      cr  // will support more in the future
    }

    public MissingValuesHandling missingValuesHandling() {
      if (_missing_values_handling instanceof MissingValuesHandling)
        return (MissingValuesHandling) _missing_values_handling;
      assert _missing_values_handling instanceof DeepLearningModel.DeepLearningParameters.MissingValuesHandling;
      switch ((DeepLearningModel.DeepLearningParameters.MissingValuesHandling) _missing_values_handling) {
        case MeanImputation:
          return MissingValuesHandling.MeanImputation;
        case Skip:
          return MissingValuesHandling.Skip;
        default:
          throw new IllegalStateException("Unsupported missing values handling value: " + _missing_values_handling);
      }
    }
    
    public DataInfo.Imputer makeImputer() {
      if (missingValuesHandling() == MissingValuesHandling.PlugValues) {
        if (_plug_values == null || _plug_values.get() == null) {
          throw new IllegalStateException("Plug values frame needs to be specified when Missing Value Handling = PlugValues.");
        }
        return new GLM.PlugValuesImputer(_plug_values.get());
      } else { // mean/mode imputation and skip (even skip needs an imputer right now! PUBDEV-6809)
        return new DataInfo.MeanImputer();
      }
    }
  }
  
  public static class GAMModelOutput extends Model.Output {
    String[] _coefficient_names;
    public int _best_lambda_idx; // lambda which minimizes deviance on validation (if provided) or train (if not)
    public int _lambda_1se = -1; // lambda_best + sd(lambda); only applicable if running lambda search with nfold
    public int _selected_lambda_idx; // lambda which minimizes deviance on validation (if provided) or train (if not)
    double[] _global_beta;
    private double[] _zvalues;
    private double _dispersion;
    private boolean _dispersionEstimated;
    public double[][][] _zTranspose; // Z matrix for de-centralization, can be null
    public double[][][] _penaltyMatrices; // stores t(Z)*t(D)*Binv*D*Z and can be null
    public double[][][] _binvD; // store BinvD for each gam column specified for scoring
    public Key<Frame> _gamTransformedTrain;  // contain key of predictors, all gam columns

    public double dispersion(){ return _dispersion;}
    
    public GAMModelOutput(GAM b, Frame adaptr) { super(b, adaptr); }

    @Override public ModelCategory getModelCategory() {
      return ModelCategory.Regression;
    } // will add others later
  }

  /**
   * This method will massage the input training frame such that it can be used for scoring for a GAM model.
   * 
   * @param test Testing Frame, updated in-place
   * @param expensive Try hard to adapt; this might involve the creation of
   *  whole Vecs and thus get expensive.  If {@code false}, then only adapt if
   *  no warnings and errors; otherwise just the messages are produced.
   *  Created Vecs have to be deleted by the caller (e.g. Scope.enter/exit).
   * @param computeMetrics
   * @return
   */
  @Override
  public String[] adaptTestForTrain(Frame test, boolean expensive, boolean computeMetrics) {
    // compare column names with test frame.  If equal, call adaptTestForTrain.  Otherwise, need to adapt it first
    String[] testNames = test.names();
    if (!equalColNames(testNames, _output._names, _parms._response_column)) {  // shallow check: column number, column names only
      Frame adptedF = new Frame(Key.make(), test._names.clone(), test.vecs().clone()); // clone test dataset
      if (_parms._ignored_columns != null) {
        for (String iname:_parms._ignored_columns) {  // remove ignored columns if test dataset has them
          if (ArrayUtils.contains(testNames, iname)) {
            adptedF.remove(iname);
          }
        }
      }
      String[][] gamColnames = new String[_parms._gam_X.length][];
      String[][] gamColnamesDecenter = new String[_parms._gam_X.length][];
      addGAM2Train(_parms, test, adptedF, null, null, gamColnames, gamColnamesDecenter, 
              false, false);
      return super.adaptTestForTrain(adptedF, expensive, computeMetrics);
    }
    return super.adaptTestForTrain(test, expensive, computeMetrics);
  }
  
  
  
  @Override
  protected Frame predictScoreImpl(Frame fr, Frame adaptFrm, String destination_key, Job j, boolean computeMetrics, 
                                   CFuncRef customMetricFunc) {
    return adaptFrm;  // place holder
  }

  @Override 
  public double[] score0(double[] data, double[] preds) {
    throw new UnsupportedOperationException("GAMModel.score0 should never be called");
  }

  @Override protected Futures remove_impl(Futures fs, boolean cascade) {
    Keyed.remove(_output._gamTransformedTrain, fs, true);
    super.remove_impl(fs, cascade);
    return fs;
  }

  @Override protected AutoBuffer writeAll_impl(AutoBuffer ab) {
    if (_output._gamTransformedTrain!=null)
      ab.putKey(_output._gamTransformedTrain);
    return super.writeAll_impl(ab);
  }

  @Override protected Keyed readAll_impl(AutoBuffer ab, Futures fs) {
    return super.readAll_impl(ab, fs);
  }
}
