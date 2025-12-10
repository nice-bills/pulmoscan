import boto3
from botocore.exceptions import ClientError
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class StorageService:
    def __init__(self):
        # Ensure endpoint starts with http:// if not present
        endpoint = settings.MINIO_ENDPOINT
        if not endpoint.startswith("http"):
            endpoint = f"http://{endpoint}"

        self.s3_client = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=settings.MINIO_ACCESS_KEY,
            aws_secret_access_key=settings.MINIO_SECRET_KEY,
            config=boto3.session.Config(signature_version='s3v4'),
            region_name="us-east-1" # MinIO default region
        )
        self.bucket = settings.MINIO_BUCKET
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
        except ClientError:
            try:
                self.s3_client.create_bucket(Bucket=self.bucket)
                logger.info(f"Created bucket: {self.bucket}")
            except Exception as e:
                logger.error(f"Failed to create bucket: {e}")

    def upload_file(self, file_obj, object_name):
        """Upload a file-like object to S3."""
        try:
            self.s3_client.upload_fileobj(file_obj, self.bucket, object_name)
            return object_name
        except ClientError as e:
            logger.error(f"Failed to upload file: {e}")
            return None

    def download_file(self, object_name, file_path):
        """Download a file from S3 to local path."""
        try:
            self.s3_client.download_file(self.bucket, object_name, file_path)
            return True
        except ClientError as e:
            logger.error(f"Failed to download file: {e}")
            return False

    def generate_presigned_url(self, object_name, expiration=3600):
        """Generate a presigned URL to share an S3 object."""
        try:
            response = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket, 'Key': object_name},
                ExpiresIn=expiration
            )
            return response
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None

# Singleton instance
storage = StorageService()
