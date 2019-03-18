using Microsoft.ML.Data;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace demo02 {
public class InputModel
    {
        public float Label;

        public float SepalLength;

        public float SepalWidth;

        public float PetalLength;

        public float PetalWidth;
    }
// IrisPrediction is the result returned from prediction operations
    public class PredictionModel
    {
        [ColumnName("PredictedLabel")]
        public uint SelectedClusterId;

        [ColumnName("Score")]
        public float[] Distance;
    }
}