using System;
using Microsoft.ML;

using Microsoft.ML.Transforms;

namespace demo01
{
    class Program
    {
        static void Main(string[] args)
        {
        
           MLContext mlContext = new MLContext();

           var trainingData = mlContext.
           Data.ReadFromTextFile<TrainingInputModel>(path:"../data/titanic/train.csv",hasHeader:true,separatorChar:',');
        
          var trainingDataPreview = trainingData.Preview();

        var pipeline = mlContext.Transforms.DropColumns(
                    nameof(TrainingInputModel.PassengerId),
                    nameof(TrainingInputModel.Name),
                    nameof(TrainingInputModel.Ticket), 
                    nameof(TrainingInputModel.Fare), 
                    nameof(TrainingInputModel.Cabin))

                .Append(mlContext.Transforms.ReplaceMissingValues(
                    nameof(TrainingInputModel.Age), 
                    nameof(TrainingInputModel.Age),
                    MissingValueReplacingTransformer.ColumnInfo.ReplacementMode.Mean))

                .Append(mlContext.Transforms.Categorical.OneHotEncoding(
                    nameof(TrainingInputModel.Gender),
                    nameof(TrainingInputModel.Gender)))

                .Append(mlContext.Transforms.Categorical.OneHotEncoding(
                    nameof(TrainingInputModel.Embarked),
                    nameof(TrainingInputModel.Embarked)))

                .Append(mlContext.Transforms.Categorical.OneHotEncoding(
                    nameof(TrainingInputModel.PassengerClass),
                    nameof(TrainingInputModel.PassengerClass)))

                .Append(mlContext.Transforms.Concatenate("Features", 
                    nameof(TrainingInputModel.PassengerClass),
                    nameof(TrainingInputModel.Gender), 
                    nameof(TrainingInputModel.Age),
                    nameof(TrainingInputModel.SiblingsOrSpouses),
                    nameof(TrainingInputModel.ParentsOrChildren), 
                    nameof(TrainingInputModel.Embarked)))

                .Append(mlContext.BinaryClassification.Trainers.FastForest(nameof(TrainingInputModel.Survived)))

                .Fit(trainingData);



    var statistics = mlContext.
    BinaryClassification.EvaluateNonCalibrated(pipeline.Transform(trainingData),
                       nameof(OutputModel.Survived));
    Console.WriteLine($"Accuracy:{statistics.Accuracy}");
    Console.WriteLine($"AUC:{statistics.Auc}");



        
        }
    }
}
