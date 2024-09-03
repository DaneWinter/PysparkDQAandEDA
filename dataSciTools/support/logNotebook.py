def logNotebook(token, folder, fileName, host = '...', linkTail = '...', logMLflow = False, printLink = True):
    """ Save calling DB notebook to a user specified location in DBFS as an html file and logs download link to MLflow and/or prints link.

        Args:
            token (str): DB token allowing notebook to write as an HTML
            folder (str): DBFS location for HTML file
            filename (str): name of saved report
            host (str): host name (ex. 'https://adb-xxx.azuredatabricks.net)
            linkTail (str): tail of html link (ex. '?oxxxx')
            logMLflow (bool): flag for if HTML link should be saved to MLflow
                (default is True)
            printLink (bool): flag for if function should print HTML link
                (default is True)
    """

    # required libs
    import time
    import datetime as dt
    import base64
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service import workspace

    # creates dbutils obj
    from pyspark.sql import SparkSession
    from pyspark.dbutils import DBUtils

    def get_dbutils():
        spark = SparkSession.builder.getOrCreate()
        
        if spark.conf.get("spark.databricks.service.client.enabled") == "true":
            return DBUtils(spark)
        
        try:
            import IPython
            return IPython.get_ipython().user_ns["dbutils"]
        except ImportError:
            raise ImportError("IPython is not available. Make sure you're not in a non-IPython environment.")
    
    dbutils = get_dbutils() 

    #catches if link will not be logged
    if (logMLflow == False and printLink == False):
        return('ERROR: user must have true for "logMLflow" and/or "printLink"')

    w = WorkspaceClient(host = host, token = token)

    notebook = '/Workspace' + dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()

    export_response = w.workspace.export(format=workspace.ExportFormat.HTML, path=notebook)

    testing_decode = export_response.content
    plaintext_bytes = base64.b64decode(testing_decode)
    plaintext_str = plaintext_bytes.decode()

    #programmatically change file name
    name = fileName + "_" + str(dt.date.today()) + '.html'
    #file_name = "/files/DataQA/" + x
    file_name = "FileStore/" + folder + "/" + name

    dbutils.fs.put(file_name, plaintext_str, True) # get over write code

    file_path = '/files/'+ folder + '/' + name 

    link = host + file_path + linkTail

    if (logMLflow == True):
        import mlflow
        with mlflow.start_run(run_name = fileName + "_" + str(dt.date.today())):
            #Log params
            mlflow.log_param("Notebook Link", link)

        mlflow.end_run()
    
    if (printLink == True):
        return(link)
