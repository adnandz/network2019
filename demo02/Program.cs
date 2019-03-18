using System;
using Microsoft.ML;
using System.IO;
using Microsoft.ML.Data;
using Microsoft.Data.DataView;
using Common;

namespace demo02
{
    internal static class Program
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);

        private static string BaseDatasetsRelativePath = @"../../../../data/iris";
        private static string DataSetRealtivePath = $"{BaseDatasetsRelativePath}/iris-full.txt";

        private static string DataPath = GetAbsolutePath(DataSetRealtivePath);

        private static string BaseModelsRelativePath = @"../../../../MLModels";
        private static string ModelRelativePath = $"{BaseModelsRelativePath}/IrisModel.zip";

        private static string ModelPath = GetAbsolutePath(ModelRelativePath);
        private static IDataView trainingDataView;
        private static IDataView testingDataView;

        private static void Main(string[] args)
        {
            //Kreiranje context kataloga sa deterministričkim ponašanjem
            MLContext mlContext = new MLContext(seed: 1); 

            // Učitavanje  podataka           
            IDataView fullData = mlContext.Data.LoadFromTextFile(path: DataPath,
                                                columns:new[]
                                                            {
                                                                new TextLoader.Column(DefaultColumnNames.Label, DataKind.Single, 0),
                                                                new TextLoader.Column(nameof(InputModel.SepalLength), DataKind.Single, 1),
                                                                new TextLoader.Column(nameof(InputModel.SepalWidth), DataKind.Single, 2),
                                                                new TextLoader.Column(nameof(InputModel.PetalLength), DataKind.Single, 3),
                                                                new TextLoader.Column(nameof(InputModel.PetalWidth), DataKind.Single, 4),
                                                            },
                                                hasHeader:true,
                                                separatorChar:'\t');

            //1. Razdravanje podataka : TrainingDataset (80%) i TestDataset (20%)
            TrainCatalogBase.TrainTestData trainTestData = mlContext.Clustering.TrainTestSplit(fullData, testFraction: 0.2);
            trainingDataView = trainTestData.TrainSet;
            testingDataView = trainTestData.TestSet;

            //2. Procesiranje podataka
            var dataProcessPipeline = mlContext.Transforms.Concatenate(DefaultColumnNames.Features, 
                                                                nameof(InputModel.SepalLength), 
                                                                nameof(InputModel.SepalWidth), 
                                                                nameof(InputModel.PetalLength), 
                                                                nameof(InputModel.PetalWidth));

           
            // 3. Kreiranje i treniranje modela   
            var trainer = mlContext.Clustering.Trainers.KMeans(featureColumnName: DefaultColumnNames.Features, clustersCount: 3);
            var trainingPipeline = dataProcessPipeline.Append(trainer);
            var trainedModel = trainingPipeline.Fit(trainingDataView);

            // 4. Evaluacija algoritma
            IDataView predictions = trainedModel.Transform(testingDataView);
            var metrics = mlContext.Clustering.Evaluate(predictions, score: DefaultColumnNames.Score, features: DefaultColumnNames.Features);

            ConsoleHelper.PrintClusteringMetrics(trainer.ToString(), metrics);

            // 5. Snimanje modela
            using (var fs = new FileStream(ModelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(trainedModel, fs);

           
            // Test sa jednostavnim primjerom 
            var sampleInputModel = new InputModel()
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f,
            };
            
            using (var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                ITransformer model = mlContext.Model.Load(stream);
                
                var predEngine = model.CreatePredictionEngine<InputModel, PredictionModel>(mlContext);

                //Pridruzivanje 
                var resultprediction = predEngine.Predict(sampleInputModel);

                Console.WriteLine($"Pridruženi cluster za setosa cvijet je " + resultprediction.SelectedClusterId);
            }

           
            Console.ReadKey();           
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }

}
