package water.runner;

import org.junit.Ignore;
import water.H2O;
import water.MRTask;

@Ignore
public class CollectInitKeysTask extends MRTask<CollectInitKeysTask>  {

    @Override
    protected void setupLocal() {
        LocalTestRuntime.initKeys.addAll(H2O.localKeySet());
    }
}
