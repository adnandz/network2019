using System;
using System.IO;
using Microsoft.ML;

using Microsoft.ML.Transforms;

namespace demo01
{
    class Program
    {
        static void Main(string[] args)
        {
        
        //Kreiranje konteksta - katalog sa operacijama
        //seed 0 - uključuje slučajnost
           MLContext mlContext = new MLContext(seed:0);

        //Učitavanje podataka za trening sa diska
           var trainingData = mlContext.
           Data.ReadFromTextFile<TrainingInputModel>(path:"../data/titanic/train.csv",hasHeader:true,separatorChar:',');
        
        //Preview podataka
          var trainingDataPreview = trainingData.Preview();

        //TRENING - Pipeline za učenje koja uključuje:
        //  1. Uklanjanje nepotrebnih kolona (PassengerId,Name,Ticket,Fare,Cabin)
        //  2. Rješavanje problema "nedostajećih vrijednosti" u koloni Age 
        //  3. OneHotEncoding kodiranje (kategorizacija podataka)
        //  4. Spajanje svih kolona koje će se koristiti u "Features" vektor
        //  5. Dodavanje algoritma učenje (konkretno FastForest)
        //  6. Poziv algoritma
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


        //EVALUACIJA - metrika algoritma (tačnost i AUC)
        var statistics = mlContext.
        BinaryClassification.EvaluateNonCalibrated(pipeline.Transform(trainingData),
                        nameof(OutputModel.Survived));
        Console.WriteLine($"Accuracy:{statistics.Accuracy}");
        Console.WriteLine($"AUC:{statistics.Auc}");

        //Snimanje treniranog modela na disk
        Console.WriteLine("Snimanje modela model.bin...");
        using (var modelFileStream = new FileStream("model.bin", FileMode.Create, FileAccess.Write))
        {
            pipeline.SaveTo(mlContext, modelFileStream);
        }

        //Učitavanje modela sa diska
        Console.WriteLine("Ucitavanje modela iz model.bin...");
        var newContext = new MLContext();
       

        using (var modelFileStream = new FileStream("model.bin", FileMode.Open, FileAccess.Read))
        {
            var newPipeline= newContext.Model.Load(modelFileStream);
        
            //TESTIRANJE uz korištenje modela učitanog sa diska

            var predictor = newPipeline.CreatePredictionEngine<PredictionInputModel, OutputModel>(newContext);

            //Učitavanje testnih podataka
            var evalData =
                newContext.Data.ReadFromTextFile<PredictionInputModel>(path:"../data/titanic/test.csv", hasHeader: true, separatorChar: ',');
            
            //Za svaki testni podatak pozvati funkciju predict
            foreach (var row in evalData.Preview().RowView)
            {
                var inputModel = new PredictionInputModel
                {
                    Embarked = row.Values[10].Value.ToString(),
                    PassengerClass = (float)row.Values[1].Value,
                    Gender = row.Values[3].Value.ToString(),
                    Age = (float)row.Values[4].Value,
                    ParentsOrChildren = (float)row.Values[6].Value,
                    SiblingsOrSpouses = (float)row.Values[5].Value,
                    Cabin = row.Values[9].Value.ToString(),
                    Name = row.Values[2].Value.ToString(),
                    Fare = (double)row.Values[8].Value,
                    Ticket = row.Values[7].Value.ToString(),
                    PassengerId = (int)row.Values[0].Value
                };

                var prediction = predictor.Predict(inputModel);

                Console.WriteLine(
                    $"{inputModel.Name}: {(prediction.Survived ? "Živ" : "Nije živ")} ({prediction.Probability:P2})");
            }
        }
            
        }
    }
}
