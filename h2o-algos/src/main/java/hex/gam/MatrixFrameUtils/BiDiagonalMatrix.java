package hex.gam.MatrixFrameUtils;

import water.MemoryManager;

/***
 * This class denotes a bidiagonal matrix with only non-zero elements on the diagonal, one diagonal above the
 * diagonal, one diagonal below the diagonal.  The matrix must be a square matrix
 */
public class BiDiagonalMatrix {
  double[] _offdiag; // off diagonal of matrix
  double[] _diag;  // diagonal of matrix

  int _size;  // square matrix size which is k-2 where k is number of knots
  
  public BiDiagonalMatrix(int size) {
    assert size>2:"Size of BiDiagonalMatrix must exceed 1 but is "+size;
    _size = size;
    _offdiag = MemoryManager.malloc8d(size-1);
    _diag = MemoryManager.malloc8d(size);
  }
  
  public BiDiagonalMatrix(double[] ldiag, double[] diag) {
    assert (ldiag.length==(diag.length-1)) &&  (diag.length >2);
    _offdiag = ldiag;
    _diag = diag;
    _size = diag.length;
  }
  
  public BiDiagonalMatrix(double[] hj) {
    this(hj.length-1); // hj is of size k-1
    int offDiagSize =_offdiag.length;
    double oneThird = 1.0/3.0;
    double oneSixth = 1.0/6.0;
    for (int index=0; index < _size; index++) {
      _diag[index] = (hj[index]+hj[index+1])*oneThird;
      if (index < offDiagSize) 
        _offdiag[index] = hj[index+1]*oneSixth;
    }
  }
}
