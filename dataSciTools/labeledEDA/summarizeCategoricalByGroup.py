def summarizeCategoricalByGroup(dataSet, groupings, graphs = True, select = [], ignore = []):
    """Summarizes counts and proportions of a categorical variable's categories by the categories of a grouping variable.

        Args:
            dataSet (pyspark dataframe): dataframe containing the categorical features and grouping variable 
            groupings (str): name of grouping feature/variable 
             graphs (bool): flag for if the function will produce graphs
                (default is True)
            select (list of str): specify features/variables in dataframe to analyze
                (default is an empty list)
            ignore (list of str): specify features/variables to remove from analysis
                (default is an empty list)

        Returns:
            itable1: interactive table of counts and proportions of every category by the grouping variable.
            graph1: barchart illustrating the proportions of the grouping levels in each category within a feature
    """

    # required libs
    from itables import show
    import matplotlib.pyplot as plt

    string_cols = [c for c, t in dataSet.dtypes if t.startswith(('string')) == True]
    if (len(select) != 0):
        string_cols = [c for c in string_cols if c in select] 
    string_cols = [c for c in string_cols if c not in ignore + [groupings]]

    for col in string_cols:
        table = (dataSet
                .select(col, groupings)
                .groupBy(col)
                .pivot(groupings)
                .count()
                .fillna(0)
                .toPandas()
                )

        table = table.set_index(col)

        catLabels = table.columns 

        for r in catLabels:
            table[str(groupings) + str(r) + 'Prop'] = [(table.loc[cat, str(r)] / table.loc[cat].sum()) for cat in table.index]

        show(table,
            paging = False,
            select = True,
            buttons=["copyHtml5", "csvHtml5", "excelHtml5"],
            scrollCollapse = True,
            scrollY="300px",
            caption = str(groupings) + ' Proportions in ' + str(col),
            style = "table-layout:auto;width:100%;margin:auto;caption-side:top",
            layout = {"bottom2": "searchBuilder", 'bottom1':'buttons'}
            )

        for r in catLabels:
            table = table.drop(r, axis = 1)

        if (graphs == True):
            barPlot = table.plot(use_index = True,
                                kind='barh', 
                                stacked=False,
                                title='Proportion of ' + groupings + ' in ' + col,
                                figsize=(14, 6))
            barPlot.invert_yaxis()
            barPlot.set_xlim([0, 1])
            barPlot.legend(loc = 'best')
            plt.show() 
