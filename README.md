# Build an ML Pipeline for Short-Term Rental Prices in NYC

An ML pipeline used to train a prediction model and upload the artifacts to WandB.

## WandB Project link
Project: nyc_airbnb_project
Link: https://wandb.ai/prashdash112/nyc_airbnb_project/overview?workspace=user-prashdash112

Link to artifacts: https://wandb.ai/prashdash112/nyc_airbnb_project/artifacts/clean_sample/clean_sample.csv/v0

## EDA (Exploratory data analysis)
In this component, We perform the pandas profiling to analyze the data and drop outliers(if any) and plot various graphs. 
We use WandB here to log these graphs and codebases.

## Data Cleaning
In this mlflow componenet, the outliers obtained in previous step to filter the data and obtain a clean data csv file.

```
if "basic_cleaning" in active_steps:
    _ = mlflow.run(
         os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
         "main",
         parameters={
             "input_artifact": "sample.csv:latest",
             "output_artifact": "clean_sample.csv",
             "output_type": "clean_sample",
             "output_description": "Data with outliers and null values removed",
             "min_price": config['etl']['min_price'],
             "max_price": config['etl']['max_price']
         },
     )
```
Run the pipeline. If you go to W&B, you will see the new artifact type clean_sample and within it the clean_sample.csv artifact

## Data testing
After the cleaning, it is a good practice to put some tests that verify that the data does not contain surprises.
One of the tests will compare the distribution of the current data sample with a reference, to ensure that there is no unexpected change.

```
def test_row_count(data):
    assert 15000 < data.shape[0] < 1000000
```
It checks that the size of the dataset is reasonable (not too small, not too large).

## Train Random Forest

Run the ```src/train_random_forest/``` component to train the random forest model.

## Optimize hyperparameters

```
> mlflow run . \
  -P steps=train_random_forest \
  -P hydra_options="modeling.random_forest.max_depth=10,50,100 modeling.random_forest.n_estimators=100,200,500 -m"
```

## Visualize the pipeline
User can now go to W&B, go the Artifacts section, select the model export artifact then click on the Graph view tab to visualize the pipeline.

## License

[License](LICENSE.txt)
