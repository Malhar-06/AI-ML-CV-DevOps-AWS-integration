# aws_config.py

import boto3

AWS_ACCESS_KEY_ID = 'your-aws-access-key'
AWS_SECRET_ACCESS_KEY = 'your-aws-secret-key'
AWS_REGION_NAME = 'ap-south-1'

# Create AWS session
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION_NAME
)

ec2 = session.resource('ec2')

# Optionally, you can define other resources or functions related to AWS setup here

