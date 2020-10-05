import logging
import os


# Either security should be set, or the environment variables
# `AWS_ACCESS`, `AWS_SECRET` should be set.
# if using security credentials use:
# ```
# scarlet_extensions.testing.aws.security = {
#   "access_key": <access_key_here>
#   "secret_key": <secret_key_here>
# ```
security= None
region = "us-east-2"


def get_client(service:str):
    """Connect to an AWS service

    :param service: AWS service to connect to
    :return: Client connection to the specified service
    """
    import boto3

    if security is None:
        _security = {
            "aws_access_key_id": os.environ["AWS_ACCESS"],
            "aws_secret_access_key": os.environ["AWS_SECRET"]
        }
    else:
        _security = security

    client = boto3.client(service, region_name=region, **_security)
    return client


def create_bucket(bucket_name:str) -> bool:
    """Create an S3 bucket in a specified region

    If a region is not specified, the bucket is created in the S3 default
    region (us-east-1).

    :param bucket_name: Bucket to create
    :return: True if bucket created, else False
    """
    from botocore.exceptions import ClientError

    # Create bucket
    try:
        client = get_client("s3")
        location = {'LocationConstraint': region}
        client.create_bucket(Bucket=bucket_name,
                             CreateBucketConfiguration=location)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def upload_file(file_name: str, bucket: str, object_name:str) -> bool:
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    from botocore.exceptions import ClientError

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    client = get_client("s3")
    try:
        client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def download_file(bucket:str, file_name:str, object_name:str):
    """Download a file from S3

    :param bucket: The name of the bucket containing the file
    :param file_name: The name of the file to download
    :return:
    """
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    client = get_client("s3")
    return client.download_file(bucket, object_name, file_name)


def table_insert(table_name:str, item:dict):
    table = get_table(table_name)
    with table.batch_writer() as batch:
        batch.put_item(Item=item)


def get_table(table_name:str):
    import boto3

    if security is None:
        _security = {
            "aws_access_key_id": os.environ["AWS_KEY"],
            "aws_secret_access_key": os.environ["AWS_SECRET"]
        }
    else:
        _security = security

    dynamodb = boto3.resource("dynamodb", region_name="us-east-2", **_security)
    return dynamodb.Table(table_name)
