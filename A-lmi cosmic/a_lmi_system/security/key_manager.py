"""
Key Management Service (KMS) Client
HashiCorp Vault integration for secure key storage
"""

import logging
import hvac
from typing import Optional

logger = logging.getLogger(__name__)


class KeyManager:
    """
    Key management service client.
    
    Manages cryptographic keys through HashiCorp Vault.
    """
    
    def __init__(self, vault_addr: str, vault_token: str):
        """
        Initialize KMS client.
        
        Args:
            vault_addr: Vault server address
            vault_token: Vault authentication token
        """
        self.vault_addr = vault_addr
        self.vault_token = vault_token
        self.client: Optional[hvac.Client] = None
        
    def connect(self):
        """Establish connection to Vault."""
        try:
            self.client = hvac.Client(url=self.vault_addr, token=self.vault_token)
            
            if self.client.is_authenticated():
                logger.info(f"Connected to Vault at {self.vault_addr}")
            else:
                raise RuntimeError("Failed to authenticate with Vault")
        except Exception as e:
            logger.error(f"Failed to connect to Vault: {e}")
            raise
    
    def get_secret(self, path: str) -> dict:
        """
        Retrieve secret from Vault.
        
        Args:
            path: Secret path
            
        Returns:
            Secret data as dictionary
        """
        if not self.client:
            raise RuntimeError("Not connected to Vault")
        
        try:
            secret = self.client.secrets.kv.v2.read_secret_version(path=path)
            return secret['data']['data']
        except Exception as e:
            logger.error(f"Error retrieving secret: {e}")
            raise
    
    def set_secret(self, path: str, data: dict):
        """
        Store secret in Vault.
        
        Args:
            path: Secret path
            data: Secret data as dictionary
        """
        if not self.client:
            raise RuntimeError("Not connected to Vault")
        
        try:
            self.client.secrets.kv.v2.create_or_update_secret(path=path, secret=data)
            logger.info(f"Stored secret: {path}")
        except Exception as e:
            logger.error(f"Error storing secret: {e}")
            raise

