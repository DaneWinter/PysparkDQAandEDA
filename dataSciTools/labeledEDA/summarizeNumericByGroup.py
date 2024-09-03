def summarizeNumericByGroup(dataSet, groupings, select = [], ignore = [], boxplots = True, graphScale = 'linear'):
    """Summarizes a numeric variable by the categories of a grouping variable.

        Args:
            dataSet (pyspark dataframe): dataframe containing the numeric features and grouping variable 
            groupings (str): name of grouping feature/variable 
            select (list of str): specify features/variables in dataframe to analyze
                (default is an empty list)
            ignore (list of str): specify features/variables to remove from analysis
                (default is an empty list)
            boxplots (bool): flag for if the function will produce boxplots
                (default is True)
            graphScale (str): x-scale for boxplots 
                options: 'linear', 'log'
                (default is 'linear')
        
        Return:
            itable1: interactive summary table of statistics about a numeric feature by the defined groupings
            graph1: boxplots illustrating the distributions of the grouping levels within a feature
    """
    from pyspark.sql import functions as F
    import seaborn as sns
    from itables import show
    import matplotlib.pyplot as plt

    numeric_cols = [c for c, t in dataSet.dtypes if t.startswith(('string', 'timestamp')) == False] # selecting only numeric columns
    if (len(select) != 0): # selecting out user defined columns
        numeric_cols = [c for c in numeric_cols if c in select]
    numeric_cols = [c for c in numeric_cols if c not in ignore + [groupings]]

    for col in numeric_cols:
        table = (dataSet
                .select(groupings, col)
                .groupBy(groupings)
                .agg(F.count(col).alias('count'),
                    F.avg(col).alias('mean'),
                    F.stddev(col).alias("sd"),
                    F.min(col).alias('min'),
                    F.percentile(col, 0.25).alias('Q1'),
                    F.median(col).alias('median'),
                    F.percentile(col, 0.75).alias('Q3'),
                    F.max(col).alias('max'),
                    F.skewness(col).alias('skewness'),
                    F.kurtosis(col).alias('kurtosis')
                    )
                .toPandas()
                )

        show(table.set_index('label'),
            paging = False,
            select = True,
            buttons=["copyHtml5", "csvHtml5", "excelHtml5"],
            scrollCollapse = True,
            scrollY="300px",
            caption = str('Summary of ' + col + ' by ' + groupings),
            style = "table-layout:auto;width:100%;margin:auto;caption-side:top",
            layout = {'bottom1':'buttons'}
            )
        
        if (boxplots == True):
            graphAble = dataSet.select(groupings, col).toPandas()

            graphAble[groupings] = graphAble[groupings].astype('string')

            fig, ax = plt.subplots(figsize = (14, 6))
            sns.boxplot(data = graphAble,
                        x = col,
                        y = groupings
                        )
            plt.xscale(graphScale)
            if (graphScale != 'linear'):
                plt.suptitle('Distribution of ' + str(col) + ' by ' + str(groupings) + ' (' + str(graphScale) + ' scale)')
            else:
                plt.suptitle('Distribution of ' + str(col) + ' by ' + str(groupings))
            plt.show()
