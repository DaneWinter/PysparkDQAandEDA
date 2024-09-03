def testEmpDistributions(train, bench, graphs = False, select = [], ignore = [], sampleRatio = 1.0, bootSamples = 10000, slices = 3328):
    """ Bootstraps empirical samples to compare and identify drift in features.

        Args:
            train (pyspark dataframe): training set which will be compared to the benchmark set
            bench (pyspark dataframe): benchmark data set which will be compared to the training set
            graphs (bool): A flag for if the function will produce visual output
                (default is True)
            select (list of str): specify features/variables in dataframe to analyze
                (default is an empty list)
            ignore (list of str): specify features/variables to remove from analysis
                (default is an empty list)
            sampleRatio (float): proportion of original dataframe that will be utilized for resampling when bootstrapping
                (default is 1.0)
            bootSamples (int): number observation in empirical samples
                (default is 10000)
            slices (int): number of partitions made in dataframe for parallelization
                (default is 3328)
                (value was optimized for a cluster of 26 workers, each with 16 cores & 128 gb of memory)

        Returns:
            graph1: visualized empirical samples from both training & bench sets
            itable1: interactive table of statistics and confidence intervals from empirical samples
    """
    import matplotlib.pyplot as plt

    def createBootSample(dataFrame, feature, sampleRatio = 1.0, bootSamples = 10000, slices = 3328):
        import pandas as pd
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, expr
        import random as rd
        from pyspark.sql.types import StructField, StructType, StringType,IntegerType, FloatType, DoubleType, DecimalType, StringType, Row
        import matplotlib.pyplot as plt
        import math
    
        def bootAggs(data, sampleRatio = 1.0):
            bootSample = data.sample(frac = sampleRatio, replace = True)
            return Row(float(bootSample.mean()),
                       float(bootSample.std()),
                       float(bootSample.quantile(0.5)),
                       float(bootSample.quantile(0.75) - bootSample.quantile(0.25)),
                       float(bootSample.skew()),
                       float(bootSample.kurtosis())
                       )
    
        flatList = dataFrame.select(feature).where(col(feature).isNotNull()).rdd.map(lambda x: x[0]).collect()
    
        data = pd.Series(flatList)
    
        schema = StructType([
            StructField('sample', IntegerType(), False),
            StructField('sampleAggs',
                        StructType([
                            StructField('mean', DoubleType(), False),
                            StructField('sd', DoubleType(), False),
                            StructField('median', DoubleType(), False),
                            StructField('IQR', DoubleType(), False),
                            StructField('skewness', DoubleType(), False),
                            StructField('kurtosis', DoubleType(), False)
                            ]),
                        False
                        )
            ])

        rddtest = spark.sparkContext.parallelize(list(range(1, bootSamples + 1)), numSlices = slices)
        dftest = (rddtest
                  .map(lambda x: (x, bootAggs(data, sampleRatio)))
                  )
    
        return dftest.toDF(schema).select('sampleAggs.*').toPandas()
    
    def makeSampleGraphs(train, bench, feature):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 2, figsize=(14, 16))

        for (ax, col) in (zip(axes.flatten(), train.columns)):
            ax.hist(bench[col],
                    alpha = 0.8,
                    label = 'bench',
                    color = 'black',
                    cumulative = False)
            ax.hist(train[col],
                    alpha = 0.5,
                    label = 'train',
                    color = 'tomato',
                    cumulative = False)
            ax.legend(loc = "upper left")
            ax.set_title(col)
        plt.suptitle("Empirical Sampling Distributions for " + feature)
        plt.tight_layout()
        return fig

    def makeSampleTable(trainSample, benchSample):
        import matplotlib.pyplot as plt
        import pandas as pd
        import math

        diffSample = trainSample.subtract(benchSample)
        trainSample['set'] = 'train'
        benchSample['set'] = 'bench'
        returnTable = (pd.concat([trainSample, benchSample])
                       .groupby('set')
                       .mean()
                       .sort_values('set', ascending = False)
                       .transpose())
        
        returnTable['diffCI'] = testCI = [(
            (diffSample[col].mean() - 2.567 * diffSample[col].std() / math.sqrt(diffSample[col].shape[0])).round(4),
            (diffSample[col].mean() + 2.567 * diffSample[col].std() / math.sqrt(diffSample[col].shape[0])).round(4)
                                          )
                                          for col in diffSample.columns]
        
        returnTable['sig'] = ['' if (returnTable['diffCI'][agg][0] <= 0 and returnTable['diffCI'][agg][1] >= 0)
                              else '***'
                              for agg in returnTable.index]
        
        returnTable['SMD'] = [trainSample[col].mean() - benchSample[col].mean() / math.sqrt((trainSample[col].std() ** 2) + (benchSample[col].std() ** 2)) for col in diffSample.columns]
        
        return returnTable.round(4)
    
    numeric_cols = [c for c, t in train.dtypes if t.startswith(('string', 'timestamp')) == False] # selecting only numeric columns
    if (len(select) != 0): # selecting out user defined columns
        numeric_cols = [c for c in numeric_cols if c in select]
    numeric_cols = [c for c in numeric_cols if c not in ignore]

    for col in numeric_cols:
        trainBootSample = createBootSample(dataFrame = train, feature = col, sampleRatio = sampleRatio, bootSamples = bootSamples, slices = slices)
        benchBootSample = createBootSample(dataFrame = bench, feature = col, sampleRatio = sampleRatio, bootSamples = bootSamples, slices = slices)

        if (graphs == True):
            figure = makeSampleGraphs(trainBootSample, benchBootSample, col)
            plt.show()

        table = makeSampleTable(trainBootSample, benchBootSample)
        displayHTML(table.to_html())
