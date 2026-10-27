# LQRS: Learned Query Re-optimization Framework for Spark SQL

1. Use python 3.8 and run `pip install -r requirements.txt`.
2. Edit `src/config.py`, there are comments for configurations in the file.
3. `python src/server.py`.
4. Use `conf/lqrs.conf` as the properties file to launch your Spark SQL application. Modify necessary lines in the file if needed (e.g. spark.master).
5. Run your train/test queries. you can see output logs in `logs/date/time.log`.
