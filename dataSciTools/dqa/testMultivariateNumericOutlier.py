def testMultivariateNumericOutlier(data, graphs = True, select = [], ignore = [], contamination = 0.1, PCA = True):
    """ Utilizes all numeric features in a Pyspark data frame to train an isolation tree to identify anomalous data points.
    
    If specified in function call (PCA = True), the function will also carry out a principal component analysis to reduce all numeric variables used to the first three principal components calculated. After PCA, function will produce a 2d map/graph of the "normal" and "anomalous" data points.

        Args:
            data (pyspark dataframe): data to provide outlier analysis on (original frame is unaffected)
            graphs (bool): flag for if boxplots for each numeric variable by datapoint type are produced
                (default is True)
            select (list of str): specify features/variables in dataframe to analyze
                (default is an empty list)
            ignore (list of str): specify features/variables to remove from analysis
                (default is an empty list)
            contamination (float): contamination of isolation tree (expected proportion of anomalous datapoints)
                (default is 0.1)
            PCA (bool): flag for if PCA should be done to map multidimensional space to 2D
                (default is True)
            
        Return:
            itable1: count of potential outliers
            graph1: boxplots showing distribution of outliers for each feature
            itable2: explained variance of first three principal components
            graph2: 2D mapping of normal and outlier points
    """
    
    # required libs
    from itables import show
    from pyspark.sql.functions import monotonically_increasing_id, array, col
    from pyspark.sql import SparkSession, functions as F, types as T
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    # extracting numeric cols
    numeric_cols = [c for c, t in data.dtypes if t.startswith(('string', 'timestamp')) == False] # selecting only numeric columns
    if (len(select) != 0): # selecting out user defined columns
        numeric_cols = [c for c in numeric_cols if c in select]
    numeric_cols = [c for c in numeric_cols if c not in ignore]   # removing user defined columns 

    data = data.withColumn("functionId", monotonically_increasing_id()) # creating internal ID for later joining

    np.random.seed(42) # setting random state

    # shrinking dataframe
    modelDataSet = data.select(numeric_cols + ['functionId'])

    # instantiate a scaler, an isolation forest classifier and convert the data into the appropriate form
    scaler = StandardScaler()
    classifier = IsolationForest(contamination = contamination, random_state=42, n_jobs=-1)
    x_train = modelDataSet.toPandas()

    # fit on the data
    x_train = scaler.fit_transform(x_train.drop(columns = 'functionId'))
    clf = classifier.fit(x_train)

    #needed spark instance
    spark = SparkSession.builder.getOrCreate()

    # broadcast the scaler and the classifier objects
    # remember: broadcasts work well for relatively small objects
    SCL = spark.sparkContext.broadcast(scaler)
    CLF = spark.sparkContext.broadcast(clf)

    def predict_using_broadcasts(features):
        """
        Scale the feature values and use the model to predict
        :return: 1 if normal, -1 if abnormal 0 if something went wrong
        """
        prediction = 0

        x_test = [features]       
        try:
            x_test = SCL.value.transform(x_test)
            prediction = CLF.value.predict(x_test)[0]
        except ValueError:
            import traceback
            traceback.print_exc()
            print('Cannot predict:', x_test)

        return int(prediction)

    udf_predict_using_broadcasts = F.udf(predict_using_broadcasts, T.IntegerType())

    features = numeric_cols

    modelDataSet = modelDataSet.withColumn(
        'prediction',
        udf_predict_using_broadcasts(array(numeric_cols))
    )

    isolationCount = modelDataSet.groupBy('prediction').count().toPandas()

    show(isolationCount,
        paging = False,
        select = True,
        buttons=["copyHtml5", "csvHtml5", "excelHtml5"],
        scrollCollapse = True,
        scrollY="300px",
        caption = 'Count of Potential Outliers (-1)',
        style = "table-layout:auto;width:100%;margin:auto;caption-side:top",
        layout = {'bottom1':'buttons'}
        )
    
    if (graphs == True):
        pdGraphable = modelDataSet.toPandas()
        print('Distribution of Potential Outliers for Each Numeric Feature')
        for feature in numeric_cols:

            fig, ax = plt.subplots(figsize = (14, 6))
            sns.boxplot(pdGraphable, x = 'prediction', y = feature, hue = 'prediction')
            plt.yscale('log')
            plt.show()

    if (PCA == True):
        from pyspark.ml.feature import PCA
        from pyspark.ml.feature import VectorAssembler

        #vectorizing
        assembler = VectorAssembler(inputCols = numeric_cols, outputCol = 'features')
        prepared_df = assembler.transform(data)

        pca = PCA(k = 3, inputCol = 'features')
        pca.setOutputCol('pca_features')
        pcaModel = pca.fit(prepared_df)

        import pyspark.pandas as ps
        PCAvar = ps.DataFrame(pcaModel.explainedVariance, columns = ['explained_var'])
        PCAvar.insert(0, 'Component',
                    value = ['PC' + str(n) for n in range(1, 4)])

        show(PCAvar.to_pandas(),
            paging = False,
            select = True,
            buttons=["copyHtml5", "csvHtml5", "excelHtml5"],
            scrollCollapse = True,
            scrollY="300px",
            caption = "Principal Components' Explained Variance",
            style = "table-layout:auto;width:100%;margin:auto;caption-side:top",
            layout = {'bottom1':'buttons'}
            )

        pcaModel.setOutputCol('output')
        pca_transformed = pcaModel.transform(prepared_df).select('functionId', 'output')

        from pyspark.ml.functions import vector_to_array
        pca_transformed2 = (pca_transformed
                            .withColumn("pc", vector_to_array("output"))
                            .select([col('functionId')] + [col("pc")[i] for i in range(3)])
                            .join(other = modelDataSet.select('functionId', 'prediction'), on = 'functionId')
                            )

        graphablePCAPD = pca_transformed2.toPandas()

        fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 8))

        ax[0].scatter(x = graphablePCAPD.loc[graphablePCAPD['prediction'] == -1]['pc[0]'],
                        y = graphablePCAPD.loc[graphablePCAPD['prediction'] == -1]['pc[1]'],
                        c = 'tomato',
                        label = 'Outlier',
                        alpha = 0.2)
        ax[0].scatter(x = graphablePCAPD.loc[graphablePCAPD['prediction'] == 1]['pc[0]'],
                        y = graphablePCAPD.loc[graphablePCAPD['prediction'] == 1]['pc[1]'],
                        c = 'black',
                        label = 'Normal',
                        alpha = 0.2)
        ax[0].set_yscale('log')
        ax[0].set_xscale('log')
        ax[0].legend(loc = 'upper right')
        ax[0].set_title('Normal Points Layered on Top')
        ax[0].set_xlabel('1st PC')
        ax[0].set_ylabel('2nd PC')

        ax[1].scatter(x = graphablePCAPD.loc[graphablePCAPD['prediction'] == 1]['pc[0]'],
                        y = graphablePCAPD.loc[graphablePCAPD['prediction'] == 1]['pc[1]'],
                        c = 'black',
                        label = 'Normal',
                        alpha = 0.2)
        ax[1].scatter(x = graphablePCAPD.loc[graphablePCAPD['prediction'] == -1]['pc[0]'],
                        y = graphablePCAPD.loc[graphablePCAPD['prediction'] == -1]['pc[1]'],
                        c = 'tomato',
                        label = 'Outlier',
                        alpha = 0.2)
        ax[1].set_yscale('log')
        ax[1].set_xscale('log')
        ax[1].legend(loc = 'upper right')
        ax[1].set_title('Outliers Layered on Top')
        ax[1].set_xlabel('1st PC')
        ax[1].set_ylabel('2nd PC')

        plt.suptitle('Outliers Graphed via First Two Principal Components')
        plt.tight_layout()
        plt.show()
        