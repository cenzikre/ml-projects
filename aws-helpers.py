import boto3
import pickle
import io
import os

# Interact with s3 bucket

def save_pkl_to_s3(obj, file_name, bucket_name=None, folder_name=None):
    pkl_object = pickle.dumps(obj)
    s3 = boto3.client('s3')
    object_key = os.path.join(folder_name, file_name + '.pkl')
    s3.put_object(Bucket=bucket_name, Key=object_key, Body=pkl_object)

def load_pkl_from_s3(file_name, bucket_name=None, folder_name=None):
    s3 = boto3.client('s3')
    object_key = os.path.join(folder_name, file_name + '.pkl')
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    serialized_data = response['Body'].read()
    return pickle.loads(serialized_data)

def save_tensor_to_s3(tensor, file_name, bucket_name=None, folder_name=None):
    import torch
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    s3 = boto3.client('s3')
    object_key = os.path.join(folder_name, file_name + '.pt')
    s3.put_object(Bucket=bucket_name, Key=object_key, Body=buffer.getvalue())

def load_tensor_from_s3(file_name, bucket_name=None, folder_name=None):
    import torch
    s3 = boto3.client('s3')
    object_key = os.path.join(folder_name, file_name + '.pt')
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    buffer = io.BytesIO(response['Body'].read())
    return torch.load(buffer)
