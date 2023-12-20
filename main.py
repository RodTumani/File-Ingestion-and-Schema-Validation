import pandas as pd
import dask.dataframe as dd
import modin.pandas as mpd
import yaml
import os
import ray
import time

# Initialize Ray
ray.init()

# Get the current user's home directory
home_directory = os.path.expanduser("~")

# Specify the relative path to the Downloads directory
downloads_directory = "Downloads"

# Specify the file name
file_name = "Accumulation-accounts-2008-2022-provisional.csv"

# Construct the full path to the file
file_path = os.path.join(home_directory, downloads_directory, file_name)

# Print the full path
print("Full path to the file:", file_path)


# Step 1: Read the file using different methods

# Pandas
@ray.remote
def read_with_pandas_parallel(file_path):
    return pd.read_csv(file_path)


def read_with_pandas(file_path):
    return pd.read_csv(file_path)


# Dask
def read_with_dask(file_path):
    dtype = {'Values': 'object'}
    return dd.read_csv(file_path, dtype=dtype)


# Modin
def read_with_modin(file_path):
    return mpd.read_csv(file_path)


# File paths
output_file_path = os.path.join(home_directory, downloads_directory, "Processed_" + file_name + ".csv")
yaml_file_path = 'config.yaml'


# Step 2: Perform basic validation on data columns

def clean_column_names(df):
    df.columns = df.columns.str.replace('[^\w\s]', '').str.strip()
    return df


# Step 3: Create a YAML file

def create_yaml_file(df, separator='|'):
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump({'separator': separator, 'columns': list(df.columns)}, yaml_file, default_flow_style=False)


# Step 4: Validate number of columns and column names with YAML

def validate_with_yaml(df):
    with open(yaml_file_path, 'r') as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    assert len(df.columns) == len(config['columns'])
    assert all(col in df.columns for col in config['columns'])


# Step 5: Write the file in pipe-separated text file (|) in gz format

def write_to_pipe_separated_gz(df, file_path, separator='|'):
    # Ensure the correct file extension is used
    if not file_path.endswith('.gz'):
        file_path += '.gz'

    # Write the DataFrame to a new file in 'append' mode for Dask
    df.to_csv(file_path, sep=separator, compression='gzip', index=False, mode='a')


# Step 6: Create a summary of the file

def create_summary(df, output_file_path):
    total_rows = len(df)
    total_columns = len(df.columns)
    file_size = os.path.getsize(output_file_path)
    print(f'Total number of rows: {total_rows}')
    print(f'Total number of columns: {total_columns}')
    print(f'File size: {file_size} bytes')


# Main script

# Read with Pandas using Ray
df_pandas_parallel = read_with_pandas_parallel.remote(file_path)
# Wait for the result
df_pandas_parallel = ray.get(df_pandas_parallel)
df_pandas_parallel = clean_column_names(df_pandas_parallel)
create_yaml_file(df_pandas_parallel)
validate_with_yaml(df_pandas_parallel)
write_to_pipe_separated_gz(df_pandas_parallel, output_file_path)
create_summary(df_pandas_parallel, output_file_path)

# Read with dask
df_dask = read_with_dask(file_path)
df_dask = clean_column_names(df_dask)
validate_with_yaml(df_dask.compute())
write_to_pipe_separated_gz(df_dask, output_file_path)
create_summary(df_dask, output_file_path)

# Read with modin
df_modin = read_with_modin(file_path)
df_modin = clean_column_names(df_modin)
validate_with_yaml(df_modin)
write_to_pipe_separated_gz(df_modin, output_file_path)
create_summary(df_modin, output_file_path)


# Function to measure the time taken by a specific operation
def measure_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return result, elapsed_time


# Measure time for Pandas
df_pandas_parallel, time_pandas = measure_time(read_with_pandas_parallel.remote, file_path)
df_pandas_parallel = ray.get(df_pandas_parallel)
df_pandas_parallel = clean_column_names(df_pandas_parallel)
create_yaml_file(df_pandas_parallel)
validate_with_yaml(df_pandas_parallel)
write_to_pipe_separated_gz(df_pandas_parallel, output_file_path)
create_summary(df_pandas_parallel, output_file_path)
print(f"Time taken by Pandas: {time_pandas} seconds")

# Measure time for Dask
df_dask, time_dask = measure_time(read_with_dask, file_path)
df_dask = clean_column_names(df_dask)
validate_with_yaml(df_dask.compute())
write_to_pipe_separated_gz(df_dask, output_file_path)
create_summary(df_dask, output_file_path)
print(f"Time taken by Dask: {time_dask} seconds")

# Measure time for Modin
df_modin, time_modin = measure_time(read_with_modin, file_path)
df_modin = clean_column_names(df_modin)
validate_with_yaml(df_modin)
write_to_pipe_separated_gz(df_modin, output_file_path)
create_summary(df_modin, output_file_path)
print(f"Time taken by Modin: {time_modin} seconds")

# Shut down Ray
ray.shutdown()
