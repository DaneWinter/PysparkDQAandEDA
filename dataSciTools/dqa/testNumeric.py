def testNumeric(train, bench, graphs = True, select = [], ignore = [], sigCutoff = 0.01, qValue = True, substantiveCutoff = 0.1, sigSubCodes = ['+/+'],
    empiricalSamples = True):
    """Tests numeric features for changes in shape and central tendency.

        Args: 
            train (pyspark dataframe): training set which will be compared to the benchmark set
            bench (pyspark dataframe): benchmark data set which will be compared to the training set
            graphs (bool): A flag for if the function will produce visual output
                (default is True)
            select (list of str): specify features/variables in dataframe to analyze
                (default is an empty list)
            ignore (list of str): specify features/variables to remove from analysis
                (default is an empty list)
            sigCutoff (float): p/qValue cutoff for determining significance when testing for shape change
                (default is 0.01)
            qValue (bool): flag to calculate and utilize q-values (adjusted p-values)
                (default is True)
            substantiveCutoff (float): value to determine if a change is central tendency is substantive
                (default is 0.1)
            sigSubCodes (list of str): list of codes that determine if secondary analysis and visualization should be applied to a feature/variable
                (default is ['+/+'])
            empiricalSamples (bool): flag for if empirical samples should be calculated and compared as secondary analysis for features with appropriate sigSubCodes
                (default is True)
        
        Returns:
            itable1: interactive table of shape & center metrics
            graph1: CDF & PDF graphs showing differences in shape and distribution (or lack thereof)
            graph2: visualized empirical samples from both training & bench sets
            itable2: interactive table of statistics and confidence intervals from empirical samples
    """

    # needed libs
    from pyspark.sql import DataFrame, SparkSession
    import numpy as np
    import pyspark.sql.functions as funcs
    from pyspark.sql.window import Window
    from scipy.stats import distributions
    from pyspark.sql.functions import col, trim, lower
    from scipy.stats.mstats import hdquantiles
    import pandas as pd
    import seaborn as sns
    from itables import show
    import math
    import matplotlib.pyplot as plt

    # needed sub functions

    def testEmpDistributionPreFlattened(train, bench, feature, graphs = False, sampleRatio = 1.0, bootSamples = 10000, slices = 3328):
        
        def createBootSample(flatPd, sampleRatio = 1.0, bootSamples = 10000, slices = 3328):
            from pyspark.sql import SparkSession
            from pyspark.sql.functions import col, expr
            import random as rd
            from pyspark.sql.types import StructField, StructType, StringType,IntegerType, FloatType, DoubleType, DecimalType, StringType, Row
        
            def bootAggs(data, sampleRatio = 1.0):
                bootSample = data.sample(frac = sampleRatio, replace = True)
                return Row(float(bootSample.mean()),
                        float(bootSample.std()),
                        float(bootSample.quantile(0.5)),
                        float(bootSample.quantile(0.75) - bootSample.quantile(0.25)),
                        float(bootSample.skew()),
                        float(bootSample.kurtosis())
                        )
        
            data = flatPd
        
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

            #needed spark instance
            spark = SparkSession.builder.getOrCreate()

            rddtest = spark.sparkContext.parallelize(list(range(1, bootSamples + 1)), numSlices = slices)
            dftest = (rddtest
                    .map(lambda x: (x, bootAggs(data, sampleRatio)))
                    )
        
            return dftest.toDF(schema).select('sampleAggs.*').toPandas()
        
        def makeSampleGraphs(train, bench, feature):
            fig, axes = plt.subplots(3, 2, figsize=(14, 12))

            for (ax, col) in (zip(axes.flatten(), train.columns)):
                ax.hist(bench[col],
                        alpha = 0.35,
                        bins = 100,
                        label = 'bench',
                        color = 'black',
                        cumulative = False)
                ax.hist(train[col],
                        alpha = 0.35,
                        bins = 100,
                        label = 'train',
                        color = 'tomato',
                        cumulative = False)
                ax.legend(loc = "upper left")
                ax.set_title(col)
            plt.suptitle("Empirical Sampling Distributions for " + feature)
            plt.tight_layout()
            return fig

        def makeSampleTable(trainSample, benchSample):
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
            
            return returnTable.round(4)

        trainBootSample = createBootSample(flatPd = train, sampleRatio = sampleRatio, bootSamples = bootSamples, slices = slices)
        benchBootSample = createBootSample(flatPd = bench, sampleRatio = sampleRatio, bootSamples = bootSamples, slices = slices)

        if (graphs == True):
            figure = makeSampleGraphs(trainBootSample, benchBootSample, feature)
            plt.show()

        table = makeSampleTable(trainBootSample, benchBootSample)
        show(table,
             paging = False,
             select = True,
             buttons=["copyHtml5", "csvHtml5", "excelHtml5"],
             caption = (feature + ': Results from Empirical Distributions'),
             style = "table-layout:auto;width:100%;margin:auto;caption-side:top",
             layout = {'bottom1':'buttons'})

    def get_cdf(df, variable, col_name):
        cdf = df.select(variable).na.drop().\
            withColumn(
                col_name,
                funcs.cume_dist().over(Window.orderBy(variable))
            ).distinct()

        return cdf

    ## https://github.com/Davi-Schumacher/KS-2Samp-PySparkSQL/blob/master/ks_2samp_sparksql.py

    def ks_2samp_spark(df1, var1, df2, var2):
        CDF_1 = 'cdf_1'
        CDF_2 = 'cdf_2'
        FILLED_CDF_1 = 'filled_cdf_1'
        FILLED_CDF_2 = 'filled_cdf_2'

        ks_stat = get_cdf(df1, var1, CDF_1).\
            join(
                get_cdf(df2, var2, CDF_2),
                on=df1[var1] == df2[var2],
                how='outer'
            ).\
            withColumn(
                FILLED_CDF_1,
                funcs.last(funcs.col(CDF_1), ignorenulls=True).
                over(Window.rowsBetween(Window.unboundedPreceding, Window.currentRow))
            ).\
            withColumn(
                FILLED_CDF_2,
                funcs.last(funcs.col(CDF_2), ignorenulls=True).
                over(Window.rowsBetween(Window.unboundedPreceding, Window.currentRow))
            ).\
            select(
                funcs.max(
                    funcs.abs(
                        funcs.col(FILLED_CDF_1) - funcs.col(FILLED_CDF_2)
                    )
                )
            ).\
            collect()[0][0]

        # Adapted from scipy.stats ks_2samp
        n1 = df1.select(var1).na.drop().count()
        n2 = df2.select(var2).na.drop().count()
        en = np.sqrt(n1 * n2 / float(n1 + n2))
        try:
            prob = distributions.kstwobign.sf((en + 0.12 + 0.11 / en) * ks_stat)
        except:
            prob = 1.0

        return ks_stat, prob

    output = pd.DataFrame(columns = ["pValue"]) # predefining output dataframe
    numeric_cols = [c for c, t in train.dtypes if t.startswith(('string', 'timestamp')) == False] # selecting only numeric columns
    if (len(select) != 0): # selecting out user defined columns
        numeric_cols = [c for c in numeric_cols if c in select]
    numeric_cols = [c for c in numeric_cols if c not in ignore]   # removing user defined columns  
        
    # loop for main output table (calc: skew, kurtosis, and pval from Kolmogorov-Smirnov Test)
    for col in numeric_cols:
        output.loc[col] = pd.Series({"pValue": ks_2samp_spark(train, col, bench, col)[1]})

    if(qValue == True):        # calc. q-value based on bonferroni adjustment
        output['qValue'] = [(c * output.shape[0]) for c in output['pValue']]

    if(qValue == True):
        sig_cols = [c for c in output.index if output.loc[c, 'qValue'] < sigCutoff]
    else:
        sig_cols = [c for c in output.index if output.loc[c, 'pValue'] < sigCutoff]

    output['robustSMD'] = 0

    for feature in sig_cols:
        from pyspark.sql.functions import col, trim, lower
        trainFlat = pd.Series(train.select(str(feature)).where(col(feature).isNotNull()).rdd.map(lambda x: x[0]).collect())
        benchFlat = pd.Series(bench.select(str(feature)).where(col(feature).isNotNull()).rdd.map(lambda x: x[0]).collect())
        
        trainMedHD = hdquantiles(trainFlat, prob = [0.5])[0] # harrell davis median for training set 
        benchMedHD = hdquantiles(benchFlat, prob = [0.5])[0] # harrell davis median for bench set

        MADxHD = 1.4826 * hdquantiles((trainFlat.subtract(trainMedHD)))[0] # median abs. dispersion for training set
        MADyHD = 1.4826 * hdquantiles((benchFlat.subtract(benchMedHD)))[0] # median abs. dispersion for bench set

        # pooled median abs. dispersion
        PMADHD = math.sqrt( 
            ((trainFlat.shape[0] - 1) * (MADxHD ** 2) + (benchFlat.shape[0] - 1) * (MADyHD ** 2))
            / (trainFlat.shape[0] + benchFlat.shape[0] - 2)
        )

        if ((trainMedHD - benchMedHD) == 0):
            output.loc[feature, 'robustSMD'] = 0
        else:
            output.loc[feature, 'robustSMD'] = (trainMedHD - benchMedHD) / PMADHD

    if (qValue == True):
        output['sig/sub'] = [
            '+/+' if (output.loc[c, 'qValue'] < sigCutoff and np.abs(output.loc[c, 'robustSMD']) >= substantiveCutoff) else
            '-/+' if (output.loc[c, 'qValue'] >= sigCutoff and np.abs(output.loc[c, 'robustSMD']) >= substantiveCutoff) else
            '+/-' if (output.loc[c, 'qValue'] < sigCutoff and np.abs(output.loc[c, 'robustSMD']) < substantiveCutoff) else
            '-/-'
            for c in output.index
                            ]
    else:
        output['sig/sub'] = [
            '+/+' if (output.loc[c, 'pValue'] < sigCutoff and np.abs(output.loc[c, 'robustSMD']) >= substantiveCutoff) else
            '-/+' if (output.loc[c, 'pValue'] >= sigCutoff and np.abs(output.loc[c, 'robustSMD']) >= substantiveCutoff) else
            '+/-' if (output.loc[c, 'pValue'] < sigCutoff and np.abs(output.loc[c, 'robustSMD']) < substantiveCutoff) else
            '-/-'
            for c in output.index
                            ]

    # outputs main table
    output = output.sort_values(['robustSMD', 'pValue'], ascending = [False, True], key = pd.Series.abs)
    show(output.round(5),
     paging = False,
     select = True,
     buttons=["copyHtml5", "csvHtml5", "excelHtml5"],
     scrollCollapse = True,
     scrollY="300px",
     caption = 'Shape & Center Metrics: Kolmogorov–Smirnov test & Robust Standard Center Difference',
     style = "table-layout:auto;width:100%;margin:auto;caption-side:top",
     layout = {"bottom2": "searchBuilder", 'bottom1':'buttons'})

    sigSubFeatures = [c for c in output.index if output.loc[c, 'sig/sub'] in sigSubCodes] # selects features to graphs & bootstrap

    for feature in sigSubFeatures:
        # flattens feature
        flatTrain = pd.Series(train.select(feature).where(col(feature).isNotNull()).rdd.map(lambda x: x[0]).collect())
        flatBench = pd.Series(bench.select(feature).where(col(feature).isNotNull()).rdd.map(lambda x: x[0]).collect())

        #graphs cdf/pdf
        fig, ax = plt.subplots(2, 2, figsize=(14, 12))
        ax[0, 0].hist(flatBench, 
                    alpha = 0.35,
                    label = 'bench',
                    color = 'black',
                    bins = 100,
                    cumulative = False)
        ax[0, 0].hist(flatTrain,
                    alpha = 0.35,
                    label = 'train',
                    color = 'tomato',
                    bins = 100,
                    cumulative = False)
        ax[0, 0].legend(loc = "upper right")
        ax[0, 0].set_title("PDF")
        
        ax[1, 0].hist(flatBench + 1, 
                    alpha = 0.35,
                    label = 'bench',
                    color = 'black',
                    bins = 100,
                    cumulative = False)
        ax[1, 0].hist(flatTrain + 1,
                    alpha = 0.35,
                    label = 'train',
                    color = 'tomato',
                    bins = 100,
                    cumulative = False)
        ax[1, 0].legend(loc = "upper right")
        ax[1, 0].set_xscale('log')
        ax[1, 0].set_title("log PDF")
        
        sns.kdeplot(data = flatBench,
                    alpha = 0.35,
                    label = 'bench',
                    color = 'black',
                    cumulative = True,
                    ax = ax[0, 1])
        sns.kdeplot(data = flatTrain,
                    alpha = 0.35,
                    x = flatBench,
                    label = 'train',
                    color = 'tomato',
                    cumulative = True,
                    ax = ax[0, 1])
        ax[0, 1].legend(loc = "lower right")
        ax[0, 1].set_title('CDF')
        
        sns.kdeplot(data = flatBench,
                    alpha = 0.35,
                    x = flatBench,
                    label = 'bench',
                    color = 'black',
                    cumulative = True,
                    ax = ax[1, 1])
        sns.kdeplot(data = flatTrain,
                    alpha = 0.35,
                    x = flatBench,
                    label = 'train',
                    color = 'tomato',
                    cumulative = True,
                    ax = ax[1, 1])
        ax[1, 1].legend(loc = "lower right")
        ax[1, 1].set_xscale('log')
        ax[1, 1].set_title('log CDF')
        
        if (qValue == True):
            plt.suptitle(feature + ': ' + str(output.loc[feature, 'qValue'].round(3)) + '/' + str(output.loc[feature, 'robustSMD'].round(3)) + ' (significance/substantiveness)')
        else:
            plt.suptitle(feature + ': ' + str(output.loc[feature, 'pValue'].round(3)) + '/' + str(output.loc[feature, 'robustSMD'].round(3)) + ' (significance/substantiveness)')
        plt.tight_layout()
        plt.show()

        if (empiricalSamples == True):
            testEmpDistributionPreFlattened(flatTrain, flatBench, feature, graphs = True)
            