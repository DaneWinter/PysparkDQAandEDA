def testNull(train, bench, graphs = True, select = [], ignore = [], cutoffDiff = 0):
    """Compares the differences in features' null rates

        Args:
            train (pyspark dataframe): training set which will be compared to the benchmark set
            bench (pyspark dataframe): benchmark data set which will be compared to the training set
            graphs (bool): A flag for if the function will produce visual output
                (default is True)
            select (list of str): specify features/variables in dataframe to analyze
                (default is an empty list)
            ignore (list of str): specify features/variables to remove from analysis
                (default is an empty list)
            cutoffDiff: difference cutoff for determining significance
                (default is 0)

        Returns:
            itable1: summary table of null rate differences
            graph1: proportional barcharts of null rate differences across features
    """

    # required libs
    import pandas as pd
    from itables import show
    import seaborn as sns
    import math
    import matplotlib.pyplot as plt
    
    output = pd.DataFrame(columns = ["trainProp", "benchProp"]) # predefining output dataframe
    cols = [c for c, t in train.dtypes] # selects all columns in training data frame
    if (len(select) != 0):  # selecting out user defined columns
        cols = [c for c in cols if c in select] 
    cols = [c for c in cols if c not in ignore]  # removing user defined columns  
    
    trainLen = train.count() # calc. counts for later computations
    benchLen = bench.count()          
    
    # loop for main output table (calc: prop. of null, diff between training and bench sets, and pval from 2 sample Z-Test)
    for col in cols:
        output.loc[col] = pd.Series({"trainProp": (train.where(train[col].isNull()).count() / trainLen),
                                     "benchProp": (bench.where(bench[col].isNull()).count() / benchLen)})
         
    output['diff'] = output['trainProp'] - output['benchProp']
    output['propDiff'] = [(output.loc[c, 'diff'] / (output.loc[c, 'benchProp'] + 0.1)) if (output.loc[c, 'diff'] == 0) else (output.loc[c, 'diff'] / output.loc[c, 'benchProp']) for c in output.index]

    output = output.sort_values('propDiff', ascending = False, key = pd.Series.abs) 
    # outputs main table
    #output = output.sort_values('pValue', ascending = True)
    show(output,
         paging = False,
         select = True,
         buttons=["copyHtml5", "csvHtml5", "excelHtml5"],
         scrollCollapse = True,
         scrollY="300px",
         caption = 'Differences in Null Rates',
         style = "table-layout:auto;width:100%;margin:auto;caption-side:top",
         layout = {"bottom2": "searchBuilder", 'bottom1':'buttons'}
         )
    
    # produces graphs of features that meet or are under the user defined proporitonal diff cutoff
    if (graphs == True):
        print("Graphs of Features with Dissimular Prop. of Missing Values:")
        for col in output.index:
            if((math.fabs(output.loc[col, "propDiff"]) >= cutoffDiff) and (output.loc[col, 'diff'] != 0)):
                goutput = pd.DataFrame({'test' : train.select(col).toPandas()[col],
                                        'bench' : bench.select(col).toPandas()[col]})
                
                #fig, ax = plt.subplots(figsize = (14, 6))
                sns.displot(
                    data = goutput.isna().melt(value_name = "NaN Values", var_name = "Data Set"),
                    y = "Data Set",
                    hue = "NaN Values",
                    multiple = "fill",
                    palette = ['black', 'tomato'],
                    height = 6,
                    aspect = 2.33 
                    )
                plt.title(col + ': Null Rates')
                plt.show()
                