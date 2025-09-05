#!/usr/bin/env python3

import boto3
import os
import re
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
import logging

def list_s3_objects(s3_client, bucket_name, prefix=''):
    """List objects in S3 bucket with optional prefix"""
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
        )
        
        if 'Contents' in response:
            objects = [obj['Key'] for obj in response['Contents']]
            return objects
        else:
            return []
    except Exception as e:
        print(f"Error listing objects: {e}")
        return []

def download_directory(s3_client, bucket, prefix, target_dir, exclude_pattern=None):
    """Download all objects in S3 bucket with given prefix to local directory"""
    exclude_pattern = re.compile(exclude_pattern) if exclude_pattern else None
    objects = list_s3_objects(s3_client, bucket, prefix)
    
    for obj in tqdm(objects, desc=f"Downloading {prefix}"):
        if obj.endswith('/'):
            continue
        if exclude_pattern and exclude_pattern.search(obj):
            continue
        
        target_path = os.path.join(target_dir, os.path.relpath(obj, prefix))
        if os.path.exists(target_path):
            continue  # skip if exists in local
        
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        s3_client.download_file(bucket, obj, target_path)
        print(f"Retrieved {obj} to {target_path}")

def upload_directory(s3_client, bucket, local_dir, s3_prefix, exclude_pattern=None):
    """Upload all files from local directory to S3 with given prefix"""
    exclude_pattern = re.compile(exclude_pattern) if exclude_pattern else None
    
    # Get all files in local directory recursively
    files_to_upload = []
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_dir)
            files_to_upload.append((local_file_path, relative_path))
    
    for local_file_path, relative_path in tqdm(files_to_upload, desc=f"Uploading to {s3_prefix}"):
        if exclude_pattern and exclude_pattern.search(relative_path):
            continue
            
        s3_key = os.path.join(s3_prefix, relative_path).replace('\\', '/')  # Ensure forward slashes
        
        try:
            s3_client.upload_file(local_file_path, bucket, s3_key)
            print(f"Uploaded {local_file_path} to {s3_key}")
        except Exception as e:
            print(f"Error uploading {local_file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Download or upload files from/to S3 based on prefix/path pairs')
    parser.add_argument('file_list', help='File containing s3_prefix_path,local_path pairs')
    parser.add_argument('--bucket', default='runpod', help='S3 bucket name (default: runpod)')
    parser.add_argument('--exclude', help='Regex pattern to exclude files (e.g., ".*\\.npy|.*\\.faiss")')
    parser.add_argument('--upload', action='store_true', help='Upload mode: upload from local_path to s3_prefix (default: download)')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()

    logging.getLogger("s3_sync").info(f"AWS ENDPOINT: {os.getenv('AWS_ENDPOINT')}")
    for key, value in os.environ.items():
        print(f"{key}={value}")

    
    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION'),
        endpoint_url=os.getenv('AWS_ENDPOINT')
    )
    
    print("S3 client initialized successfully")
    mode = "upload" if args.upload else "download"
    print(f"Mode: {mode}")
    
    # Read file list and process each pair
    with open(args.file_list, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            try:
                s3_prefix, local_path = [x.strip() for x in line.split(',', 1)]
                
                if args.upload:
                    print(f"\nProcessing line {line_num}: {local_path} -> {s3_prefix}")
                    upload_directory(s3_client, args.bucket, local_path, s3_prefix, args.exclude)
                else:
                    print(f"\nProcessing line {line_num}: {s3_prefix} -> {local_path}")
                    download_directory(s3_client, args.bucket, s3_prefix, local_path, args.exclude)
            except ValueError:
                print(f"Error parsing line {line_num}: {line}")
                print("Expected format: s3_prefix_path,local_path")
                continue

if __name__ == "__main__":
    main()