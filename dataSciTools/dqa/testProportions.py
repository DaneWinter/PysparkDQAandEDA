def testProportions(train, bench, graphs = False, select = [], ignore = [], cutoffACPD = 0):
    """Calculates differences in category proportions for factor/categorical based variables.

        Args:
            train (pyspark dataframe): training set which will be compared to the benchmark set
            bench (pyspark dataframe): benchmark data set which will be compared to the training set
            graphs (bool): A flag for if the function will produce visual output
                (default is True)
            select (list of str): specify features/variables in dataframe to analyze
                (default is an empty list)
            ignore (list of str): specify features/variables to remove from analysis
                (default is an empty list)
            cutoffACPD (float): ACPD cutoff for determining significance
                (default is 0)

        Returns:
            itable1: interactive table of features' category counts, proportions and differences
            graph: feature categories with the largest proportional differences
            itable2: interactive summary table of all feature's proportional drift statistics 
    """
    # required libs
    import pandas as pd
    from itables import show
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pyspark.sql.functions import lit

    string_cols = [c for c, t in train.dtypes if t.startswith(('string')) == True]
    if (len(select) != 0):
        string_cols = [c for c in string_cols if c in select] 
    string_cols = [c for c in string_cols if c not in ignore]

    ACPDtable = pd.DataFrame(columns = ['categoryCount', 'LargestCategoricalDiff', "ACPD", "AvgACPD"])

    for col in string_cols:
         table = ((train
                 .select(col)
                 .withColumn("dataSet", lit("trainCount"))
                 .union(bench
                        .select(col)
                        .withColumn('dataSet', lit('benchCount'))
                        )
                 .groupBy(col)
                 .pivot("dataSet")
                 .count()
                 .fillna(0)
                 .select(col, 'trainCount', 'benchCount')
                 .toPandas()
                 ))
         table = table.set_index(col)
         table['trainProp'] = table['trainCount'] / table['trainCount'].sum()
         table['benchProp'] = table['benchCount'] / table['benchCount'].sum()
         table['propDiff'] = table['trainProp'] - table['benchProp']
         table = table.sort_values(['propDiff'], ascending = False, key = pd.Series.abs)

         catCount = table.shape[0]

         largestDiff = table['propDiff'].iloc[0]

         ACPD = table['propDiff'].abs().sum()

         AvgACPD = ACPD / catCount

         ACPDtable.loc[col] = pd.Series({'categoryCount': catCount,
                                         'LargestCategoricalDiff': largestDiff,
                                         'ACPD': ACPD,
                                         'AvgACPD': AvgACPD})

         if (ACPD >= cutoffACPD):
             #print('Absolute Categorical Proportional Difference for ' + col + ': ' + str(ACPD))
    
             show(table,
                  paging = False,
                  select = True,
                  buttons=["copyHtml5", "csvHtml5", "excelHtml5"],
                  scrollCollapse = True,
                  scrollY="300px",
                  caption = (col + ': Absolute Categorical Proportional Difference = ' + str(ACPD)),
                  style = "table-layout:auto;width:100%;margin:auto;caption-side:top",
                  layout = {"bottom2": "searchBuilder", 'bottom1':'buttons'})

             if (graphs == True):
                 
                 fig, ax = plt.subplots(figsize = (14, 6))
                 sns.barplot(
                     data = table.head(10),
                     y = 'propDiff',
                     x = table.head(10).index
                    )
                 plt.xlabel(col)
                 plt.ylabel('Proportion Difference')
                 plt.xticks(rotation = 90)
                 if (table.shape[0] > 10):
                     plt.title(col + ':\n' + '10 Largest Proportion Differences [train - bench]')
                 else:
                     plt.title(col + ':\n' + str(table.shape[0]) + ' Largest Proportion Differences [train - bench]')
                 plt.show()

    ACPDtable['sig'] = ['***' if c >= cutoffACPD else '' for c in ACPDtable['ACPD']]

    show(ACPDtable.sort_values(['AvgACPD'], ascending = False),
         paging = False,
         select = True,
         buttons=["copyHtml5", "csvHtml5", "excelHtml5"],
         caption = 'Absolute Categorical Proportional Difference Across Features',
         style = "table-layout:auto;width:100%;margin:auto;caption-side:top",
         layout = {'bottom1':'buttons'}
                   )
    