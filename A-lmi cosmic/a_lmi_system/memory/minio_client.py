"""
MinIO Client: Tier 1 Raw Data Lake
Object storage for original files
"""

import logging
from minio import Minio
from minio.error import S3Error
from typing import Optional, BinaryIO

logger = logging.getLogger(__name__)


class MinIOClient:
    """
    Client for MinIO object storage (Tier 1).
    
    Stores raw data files referenced by LightToken raw_data_ref.
    """
    
    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str, secure: bool = False):
        """
        Initialize MinIO client.
        
        Args:
            endpoint: MinIO server endpoint
            access_key: Access key
            secret_key: Secret key
            bucket: Bucket name
            secure: Use HTTPS
        """
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        self.bucket = bucket
        self.connected = False
        
    def connect(self):
        """Establish connection and create bucket if needed."""
        try:
            # Create bucket if it doesn't exist
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info(f"Created bucket: {self.bucket}")
            
            self.connected = True
            logger.info(f"Connected to MinIO at {self.client._base_url}")
        except S3Error as e:
            logger.error(f"Failed to connect to MinIO: {e}")
            raise
    
    def upload_file(self, file_path: str, object_name: str) -> str:
        """
        Upload a file to MinIO.
        
        Args:
            file_path: Local file path
            object_name: Object name in bucket
            
        Returns:
            Object URL (reference)
        """
        if not self.connected:
            raise RuntimeError("Not connected to MinIO")
        
        try:
            from minio.commonconfig import Tags
            from minio.deleteobjects import DeleteObject
            
            # Upload file
            self.client.fput_object(self.bucket, object_name, file_path)
            
            logger.info(f"Uploaded file: {object_name}")
            
            # Return reference (for LightToken raw_data_ref)
            return f"{self.bucket}/{object_name}"
        except S3Error as e:
            logger.error(f"Error uploading file: {e}")
            raise
    
    def upload_bytes(self, data: bytes, object_name: str) -> str:
        """
        Upload bytes data to MinIO.
        
        Args:
            data: Bytes data
            object_name: Object name in bucket
            
        Returns:
            Object URL (reference)
        """
        if not self.connected:
            raise RuntimeError("Not connected to MinIO")
        
        try:
            from io import BytesIO
            from minio import Minio
            
            # Create BytesIO object
            data_stream = BytesIO(data)
            
            # Upload
            self.client.put_object(
                self.bucket,
                object_name,
                data_stream,
                length=len(data)
            )
            
            logger.info(f"Uploaded bytes: {object_name}")
            return f"{self.bucket}/{object_name}"
        except S3Error as e:
            logger.error(f"Error uploading bytes: {e}")
            raise
    
    def download_file(self, object_name: str, file_path: str):
        """
        Download a file from MinIO.
        
        Args:
            object_name: Object name in bucket
            file_path: Local file path
        """
        if not self.connected:
            raise RuntimeError("Not connected to MinIO")
        
        try:
            self.client.fget_object(self.bucket, object_name, file_path)
            logger.info(f"Downloaded file: {object_name}")
        except S3Error as e:
            logger.error(f"Error downloading file: {e}")
            raise
    
    def get_object(self, object_name: str) -> bytes:
        """
        Get object data as bytes.
        
        Args:
            object_name: Object name in bucket
            
        Returns:
            Object data as bytes
        """
        if not self.connected:
            raise RuntimeError("Not connected to MinIO")
        
        try:
            response = self.client.get_object(self.bucket, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            
            return data
        except S3Error as e:
            logger.error(f"Error getting object: {e}")
            raise
    
    def delete_object(self, object_name: str):
        """
        Delete an object from MinIO.
        
        Args:
            object_name: Object name in bucket
        """
        if not self.connected:
            raise RuntimeError("Not connected to MinIO")
        
        try:
            self.client.remove_object(self.bucket, object_name)
            logger.info(f"Deleted object: {object_name}")
        except S3Error as e:
            logger.error(f"Error deleting object: {e}")

